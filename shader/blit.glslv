#version 150 core

in vec2 a_Pos;
in vec2 a_Uv;
out vec2 v_TexCoord;

void main() {
	v_TexCoord = a_Uv;
	gl_Position = vec4(a_Pos, 0.0, 1.0);
}
