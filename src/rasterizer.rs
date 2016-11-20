use cgmath;
use cgmath::SquareMatrix;
use cgmath::{Matrix4, Vector4, Vector3, Vector2};

use hprof;

use std::f32;
use std::mem;

pub struct Rasterizer {
    width: u32,
    height: u32,
    backbuffer: Vec<[u8; 4]>,
    depthbuffer: Vec<u32>,
    view: Matrix4<f32>,
    proj: Matrix4<f32>
}

impl Rasterizer {
    pub fn new(width: u32, height: u32) -> Self {
        Rasterizer {
            width: width,
            height: height,
            backbuffer: vec![[0; 4]; (width * height) as usize],
            depthbuffer: vec![0; (width * height) as usize],
            view: Matrix4::<f32>::identity(),
            proj: Matrix4::<f32>::identity(),
        }
    }

    pub fn draw<I, O>(&mut self, pipeline: &mut Pipeline<I, O>, vertices: &[I], prof: &hprof::Profiler)
            where I: Copy + Clone, O: Copy + Clone + Blend
    {
        pipeline.process(self.width,
                         self.height,
                         vertices,
                         self.view,
                         self.proj,
                         &mut self.backbuffer,
                         &mut self.depthbuffer,
                         &prof);
    }

    pub fn set_view(&mut self, mat: cgmath::Matrix4<f32>) {
        self.view = mat;
    }

    pub fn set_proj(&mut self, mat: cgmath::Matrix4<f32>) {
        self.proj = mat;
    }

    pub fn clear(&mut self) {
        for x in &mut self.backbuffer {
            x[0] = 0;
            x[1] = 0;
            x[2] = 0;
            x[3] = 0;
        }

        for x in &mut self.depthbuffer {
            *x = u32::max_value();
        }
    }

    pub fn backbuffer(&self) -> &Vec<[u8; 4]> {
        &self.backbuffer
    }

    pub fn depthbuffer(&self) -> &Vec<[u8; 4]> {
        unsafe { mem::transmute(&self.depthbuffer) }
    }

}

pub trait Blend {
    fn blend(a: Self, aw: f32, b: Self, bw: f32, c: Self, cw: f32) -> Self;
}

struct Depth(f32);

impl Blend for Depth {
    fn blend(a: Self, aw: f32, b: Self, bw: f32, c: Self, cw: f32) -> Self {
        Depth(a.0 * aw + b.0 * bw + c.0 * cw)
    }
}

pub type VertexShader<I, O> = fn(I, Matrix4<f32>) -> (O, Vector4<f32>);
pub type FragmentShader<I> = fn(Vector2<f32>, I) -> Vector4<f32>;

pub struct Pipeline<I, O> {
    vertex_fn: VertexShader<I, O>,
    frag_fn: FragmentShader<O>,

    frag_cache: Vec<O>,
    vertex_cache: Vec<Vector4<f32>>
}

impl<I, O> Pipeline<I, O>
        where I: Copy + Clone, O: Copy + Clone + Blend {
    pub fn new(vertex: VertexShader<I, O>, frag: FragmentShader<O>) -> Self {
        Pipeline {
            vertex_fn: vertex,
            frag_fn: frag,

            frag_cache: Vec::with_capacity(4096),
            vertex_cache: Vec::with_capacity(4096)
        }
    }

    #[inline]
    fn rasterize(width: u32,
                 height: u32,
                 triangles: &Vec<Vector4<f32>>,
                 attributes: &Vec<O>,
                 backbuffer: &mut Vec<[u8; 4]>,
                 depth: &mut Vec<u32>,
                 frag: FragmentShader<O>)
    {
        let width = width as i32;
        let height = height as i32;

        #[inline]
        fn edge_function(a: Vector2<i32>, b: Vector2<i32>, c: Vector2<i32>) -> i32 {
            (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x)
        }

        #[inline]
        fn edge_step(a: Vector2<i32>, b: Vector2<i32>, step: i32) -> i32 {
            -(b.y-a.y)*step
        }

        #[inline]
        fn map_coord(vec: Vector4<f32>, width: f32, height: f32, near: f32) -> Vector2<i32> {
            let screen = Vector2::new(vec.x / -vec.z, vec.y / -vec.z);
            let ndc = Vector2::new(2f32 * screen.x / width - 1f32,
                                   2f32 * screen.y / height - 1f32);
            Vector2::new(((screen.x + 1f32) / 2f32 * width) as i32,
                         ((1f32 - screen.y) / 2f32 * height) as i32)
        }

        let step = 256;
        let mask = step - 1;

        for triangle in (triangles.chunks(3).zip(attributes.chunks(3))).into_iter() {
            if let (&[a, b, c], &[A, B, C]) = triangle {
                // TODO: return coords as super-sampled coords in 1/256th
                //       resolution

                let p1 = map_coord(a, width as f32, height as f32, 0.1f32);
                let p2 = map_coord(b, width as f32, height as f32, 0.1f32);
                let p3 = map_coord(c, width as f32, height as f32, 0.1f32);

                let mut min = Vector2::<i32>::new(i32::max_value(), i32::max_value());
                let mut max = Vector2::<i32>::new(i32::min_value(), i32::min_value());

                use std::cmp;

                min.x = cmp::min(cmp::min(cmp::min(min.x, p1.x), p2.x), p3.x);
                min.y = cmp::min(cmp::min(cmp::min(min.y, p1.y), p2.y), p3.y);

                max.x = cmp::max(cmp::max(cmp::max(min.x, p1.x), p2.x), p3.x);
                max.y = cmp::max(cmp::max(cmp::max(min.y, p1.y), p2.y), p3.y);

                /*if (p1.x as i32) < min.x { min.x = p1.x as i32; }
                if (p1.y as i32) < min.y { min.y = p1.y as i32; }
                if (p2.x as i32) < min.x { min.x = p2.x as i32; }
                if (p2.y as i32) < min.y { min.y = p2.y as i32; }
                if (p3.x as i32) < min.x { min.x = p3.x as i32; }
                if (p3.y as i32) <https://open.spotify.com/track/6X4mvrDbIckxVQlRm3HhtE min.y { min.y = p3.y as i32; }

                if (p1.x as i32) > max.x { max.x = p1.x as i32; }
                if (p1.y as i32) > max.y { max.y = p1.y as i32; }
                if (p2.x as i32) > max.x { max.x = p2.x as i32; }
                if (p2.y as i32) > max.y { max.y = p2.y as i32; }
                if (p3.x as i32) > max.x { max.x = p3.x as i32; }
                if (p3.y as i32) > max.y { max.y = p3.y as i32; }
*/
                if max.x > width  { max.x = width; }
                if max.y > height { max.y = height; }
                if min.x > width  { min.x = width; }
                if min.y > height { min.y = height; }

                if max.x < 0 { max.x = 0; }
                if max.y < 0 { max.y = 0; }
                if min.x < 0 { min.x = 0; }
                if min.y < 0 { min.y = 0; } 

                //println!("({}, {}), ({}, {})", min.x, min.y, max.x, max.y);
                /*min.x = (min.x + mask) & !mask;
                min.y = (min.y + mask) & !mask;
                max.x = (max.x + mask) & !mask;
                max.y = (max.y + mask) & !mask;*/

                let (w1_step, w2_step, w3_step) = (edge_step(p1, p2, 1),
                                                   edge_step(p2, p3, 1),
                                                   edge_step(p3, p1, 1));

                for x in (min.x..max.x) {
                    for y in (min.y..max.y) {
                        let point = Vector2::new(x, y);

                        let area = edge_function(p1, p2, p3);
                        if area <= 0 {
                            continue;
                        }

                                        let (mut w1, mut w2, mut w3) = (edge_function(p1, p2, point),
                                                edge_function(p2, p3, point),
                                                edge_function(p3, p1, point));


                        if w1 >= 0 && w2 >= 0 && w3 >= 0 {
                            let w1 = w1 as f32 / area as f32;
                            let w2 = w2 as f32 / area as f32;
                            let w3 = w3 as f32 / area as f32;

                            let depth_val = ((f32::min(Blend::blend(Depth(a.z), w1 as f32, Depth(b.z), w2 as f32, Depth(c.z), w3 as f32).0,
                                                       1000f32)
                                                     / 1000f32) * u32::max_value() as f32) as u32;

                            let idx = (x + width * y) as usize;

                            if depth_val < depth[idx] {
                                depth[idx] = depth_val;

                                let color = (frag)(Vector2::new(point.x as f32, point.y as f32), Blend::blend(A, w1 as f32, B, w2 as f32, C, w3 as f32));

                                backbuffer[idx][0] = (color.x * 255f32) as u8;
                                backbuffer[idx][1] = (color.y * 255f32) as u8;
                                backbuffer[idx][2] = (color.z * 255f32) as u8;
                                backbuffer[idx][3] = (color.w * 255f32) as u8;
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn process(&mut self, width: u32, height: u32, vertices: &[I], view: Matrix4<f32>, proj: Matrix4<f32>, mut backbuffer: &mut Vec<[u8; 4]>, mut depth: &mut Vec<u32>, prof: &hprof::Profiler) {
        prof.enter_noguard("transform");
        for &vertex in vertices {
            let (out, pos) = (self.vertex_fn)(vertex, proj * view * Matrix4::from_scale(15f32));

            self.frag_cache.push(out);
            self.vertex_cache.push(pos);
        }
        prof.leave();

        // borrowck can't see iterator borrow as disjoint from rest
        prof.enter_noguard("rasterize");
        Self::rasterize(width,
                        height,
                        &self.vertex_cache,
                        &self.frag_cache,
                        &mut backbuffer,
                        &mut depth,
                        self.frag_fn);
        prof.leave();

        self.frag_cache.clear();
        self.vertex_cache.clear();
    }
}
