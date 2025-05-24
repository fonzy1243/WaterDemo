#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

//shader input
layout (location = 0) in vec3 inColor;
layout (location = 1) in vec2 inUV;

//output write
layout (location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D heightmap;
layout(set = 0, binding = 1) uniform sampler2D normalmap;

struct Vertex {
    vec3 position;
    float uv_x;
    vec3 normal;
    float uv_y;
    vec4 color;
};

layout(buffer_reference, std430) readonly buffer VertexBuffer{
    Vertex vertices[];
};

layout(push_constant) uniform constants
{
    vec4 ambientColor;
    vec4 sunlightDirection;
    vec4 sunlightColor;
    mat4 render_matrix;
    VertexBuffer vertexBuffer;
} PushConstants;

void main()
{
    vec4 normalData = texture(normalmap, inUV);

    vec3 normal = normalize(vec3(-normalData.r, -normalData.g, 1.0));

    vec3 lightDir = normalize(PushConstants.sunlightDirection.xyz);
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * PushConstants.sunlightColor.rgb;

    vec3 lighting = PushConstants.ambientColor.rgb + diffuse;
    outColor = vec4(inColor * lighting * PushConstants.sunlightColor.w, 1.0f);
}