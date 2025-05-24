#version 460
#extension GL_EXT_buffer_reference : require

layout (location = 0) out vec3 outColor;
layout (location = 1) out vec2 outUV;

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

layout (set = 0, binding = 0) uniform sampler2D heightmap;
layout (set = 0, binding = 1) uniform sampler2D normalmap;

//push constants block
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
    //load vertex data from device adress
    Vertex v = PushConstants.vertexBuffer.vertices[gl_VertexIndex];

    vec2 uv = vec2(v.uv_x, v.uv_y);
    float height = texture(heightmap, uv).r;

    vec3 pos = v.position;
    pos.y += height;

    //output data
    gl_Position = PushConstants.render_matrix * vec4(pos, 1.0f);
    outColor = v.color.xyz;
    outUV = uv;
}
