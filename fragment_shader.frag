#version 330 core

out vec4 FragColor;

in vec2 TexCoord;
uniform sampler2D ourTexture; 
uniform float tileSize; 

void main()
{
    // Calculate scaled texture coordinates
    vec2 scaledTexCoord = TexCoord / tileSize;
    FragColor = texture(ourTexture, scaledTexCoord); 
}
