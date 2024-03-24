#version 330 core

out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D ourTexture; 

void main()
{
    FragColor = vec4(1.0, 0.5, 0.8, 1.0);
}