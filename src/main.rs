#![feature(slice_patterns)]
#![feature(step_by)]

#[macro_use]
extern crate gfx;
extern crate gfx_app;
extern crate cgmath;
extern crate obj;
extern crate hprof;

use cgmath::SquareMatrix;
use cgmath::{Matrix4, Vector4, Vector3, Vector2, Transform, Point3};

use gfx::{Bundle};
use gfx::texture::Kind;
use gfx::format::{Rgba8, Depth, ChannelType};
use gfx::memory::{Usage, Access};

use gfx_app::ColorFormat;

use std::path::Path;
use std::time::Instant;

mod rasterizer;

use rasterizer::*;

#[derive(Copy, Clone)]
pub struct Color(Vector3<f32>);
#[derive(Copy, Clone)]
pub struct Uv(Vector2<f32>);

impl Blend for Color {
    fn blend(a: Self, aw: f32, b: Self, bw: f32, c: Self, cw: f32) -> Self {
        Color(Vector3::new(a.0.x * aw + b.0.x * bw + c.0.x * cw,
                           a.0.y * aw + b.0.y * bw + c.0.y * cw,
                           a.0.z * aw + b.0.z * bw + c.0.z * cw))
    }
}

impl Blend for Uv {
    fn blend(a: Self, aw: f32, b: Self, bw: f32, c: Self, cw: f32) -> Self {
        Uv(Vector2::new(a.0.x * aw + b.0.x * bw + c.0.x * cw,
                        a.0.y * aw + b.0.y * bw + c.0.y * cw))
    }
}

#[derive(Copy, Clone)]
pub struct Vertex {
    pos: Vector3<f32>,
    uv: Vector2<f32>
}

#[inline]
pub fn vertex_shader(input: Vertex, mat: Matrix4<f32>) -> (Uv, Vector4<f32>) {
    (Uv(input.uv), mat * input.pos.extend(1f32))
}

#[inline]
pub fn fragment_shader(fragment: Vector2<f32>, input: Uv) -> Vector4<f32> {
    Vector4::new(input.0.x, input.0.y, 1f32, 1f32)
}


const SIZE: (u32, u32) = (854, 640);

gfx_defines!{
    vertex BlitVertex {
        pos: [f32; 2] = "a_Pos",
        uv: [f32; 2] = "a_Uv",
    }

    pipeline blit {
        vbuf: gfx::VertexBuffer<BlitVertex> = (),
        blit: gfx::TextureSampler<[f32; 4]> = "t_Blit",
        out: gfx::RenderTarget<gfx::format::Rgba8> = "Target0",
    }
}

impl BlitVertex {
    pub fn new(pos: [f32; 2], uv: [f32; 2]) -> Self {
        BlitVertex {
            pos: pos,
            uv: uv,
        }
    }
}

struct App<R: gfx::Resources> {
    bundle: Bundle<R, blit::Data<R>>,
    blit_texture: gfx::handle::Texture<R, gfx::format::R8_G8_B8_A8>,
    rasterizer: Rasterizer,
    pipeline: Pipeline<Vertex, Uv>,
    model: Vec<Vertex>,
    start_time: Instant,
    prof: hprof::Profiler
}

impl<R> gfx_app::Application<R> for App<R>
        where R: gfx::Resources
{
    fn new<F: gfx::Factory<R>>(mut factory: F, init: gfx_app::Init<R>) -> Self {
        use gfx::traits::FactoryExt;

        let vs = gfx_app::shade::Source {
            glsl_150: include_bytes!("../shader/blit.glslv"),
            .. gfx_app::shade::Source::empty()
        };
        let ps = gfx_app::shade::Source {
            glsl_150: include_bytes!("../shader/blit.glslf"),
            .. gfx_app::shade::Source::empty()
        };

        let vertex_data = [
            BlitVertex::new([-1.0, -1.0], [0.0, 1.0]),
            BlitVertex::new([ 1.0, -1.0], [1.0, 1.0]),
            BlitVertex::new([ 1.0,  1.0], [1.0, 0.0]),

            BlitVertex::new([-1.0, -1.0], [0.0, 1.0]),
            BlitVertex::new([ 1.0,  1.0], [1.0, 0.0]),
            BlitVertex::new([-1.0,  1.0], [0.0, 0.0]),
        ];

        let (vbuf, slice) = factory.create_vertex_buffer_with_slice(&vertex_data, ());

        let texture = factory.create_texture(Kind::D2(SIZE.0 as _, SIZE.1 as _, gfx::texture::AaMode::Single), 1, gfx::SHADER_RESOURCE, Usage::CpuOnly(gfx::memory::WRITE), Some(ChannelType::Unorm)).unwrap();
        let view = factory.view_texture_as_shader_resource::<gfx::format::Rgba8>(&texture, (0, 0), gfx::format::Swizzle::new()).unwrap();

        let sampler = factory.create_sampler_linear();

        let pso = factory.create_pipeline_simple(
            vs.select(init.backend).unwrap(),
            ps.select(init.backend).unwrap(),
            blit::new()
        ).unwrap();

        let data = blit::Data {
            vbuf: vbuf,
            blit: (view.clone(), sampler.clone()),
            out: init.color,
        };

        App {
            bundle: Bundle::new(slice, pso, data),
            blit_texture: texture,

            rasterizer: Rasterizer::new(SIZE.0 as _, SIZE.1 as _),
            pipeline: Pipeline::new(vertex_shader, fragment_shader),
            model: {
                let object = obj::load::<obj::SimplePolygon>(Path::new("./data/spot.obj")).unwrap();
                let indices = object.object_iter().next().unwrap().group_iter().next().unwrap().indices();

                let mut vertices = Vec::new();
                for tri in indices {
                    for &(pos, uv, _) in tri {
                        vertices.push(Vertex {
                            pos: object.position()[pos].into(),
                            uv: object.texture()[uv.unwrap()].into()
                        });
                    }
                }
                vertices
            },
            start_time: Instant::now(),
            prof: hprof::Profiler::new("Rasterizer")
        }
    }

    fn render<C: gfx::CommandBuffer<R>>(&mut self, encoder: &mut gfx::Encoder<R, C>) {
        self.prof.start_frame();

        let elapsed = self.start_time.elapsed();
        let time = elapsed.as_secs() as f32 + elapsed.subsec_nanos() as f32 / 1000_000_000.0;
        let x = (time / 5.0).sin();
        let y = (time / 5.0).cos();
        let view = Transform::look_at(
            Point3::new(x * 32.0,  16.0, y* 32.0),
            Point3::new(0.0, 0.0, 0.0),
            Vector3::unit_z(),
        );

        //self.rasterizer.set_proj(cgmath::ortho(-(SIZE.0 as f32) / 2.0, SIZE.0 as f32 / 2.0, SIZE.1 as f32 / 2.0, -(SIZE.0 as f32) / 2.0, 0.1, 1000.0));
        self.rasterizer.set_proj(cgmath::perspective(cgmath::Deg(60.0f32), (SIZE.0 as f32 / SIZE.1 as f32), 0.1, 1000.0));
        //self.rasterizer.set_proj(Matrix4::identity());
        self.rasterizer.set_view(view);

        self.prof.enter_noguard("clear");
        self.rasterizer.clear();
        self.prof.leave();

        self.prof.enter_noguard("draw");
        self.rasterizer.draw(&mut self.pipeline, &self.model, &self.prof);
        self.prof.leave();

        self.prof.enter_noguard("blit");
        encoder.update_texture::<gfx::format::R8_G8_B8_A8, gfx::format::Rgba8>(&self.blit_texture, None, gfx::texture::NewImageInfo {
            xoffset: 0,
            yoffset: 0,
            zoffset: 0,
            width: SIZE.0 as _,
            height: SIZE.1 as _,
            depth: 0,
            format: (),
            mipmap: 0 as _
        }, self.rasterizer.backbuffer()).unwrap();
        self.prof.leave();

        encoder.clear(&self.bundle.data.out, [0.3, 0.3, 0.3, 1.0]);
        self.bundle.encode(encoder);

        self.prof.end_frame();
        if true {
            self.prof.print_timing();
        }
    }
}

pub fn main() {
    use gfx_app::Application;

    App::launch_default("sw-rast");
}
