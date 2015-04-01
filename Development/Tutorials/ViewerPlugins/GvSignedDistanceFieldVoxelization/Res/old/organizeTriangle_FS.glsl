#version 420
#extension GL_ARB_shader_atomic_counters : enable
#extension GL_EXT_shader_image_load_store : enable

// To count how many triangles are in the brick
layout (binding = 0, offset = 0) uniform atomic_uint counter;

// To write the IBO
uniform layout( r32ui ) coherent uimageBuffer IBO;

// Length max of the IBO
uniform uint lengthMax;

in Data {
    vec3 normal;
} dataIn;

void main()
{

	// if ( atomicCounterIncrement( counter ) < lengthMax) {
		
	// } else {
		// discard();
	// }
}