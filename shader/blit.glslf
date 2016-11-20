#version 150 core

uniform sampler2D t_Blit;
in vec2 v_TexCoord;
out vec4 Target0;

void main() {
	vec4 tex = texture(t_Blit, v_TexCoord);
	Target0 = tex;
}
