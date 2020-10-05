#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define workGroupSize 128

__kernel void sum(__global const unsigned* a,
    __global unsigned* b,
    unsigned int n)
{
    __local unsigned c[workGroupSize];
    const unsigned index = get_global_id(0);
    const unsigned local_index = get_local_id(0);
    const unsigned group_index = get_group_id(0);
    c[local_index] = (index < n) ? a[index] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (index < n)
    {
        if (local_index == 0)
        {
            for (unsigned i = 1; i < workGroupSize; ++i)
                c[0] += c[i];
            b[group_index] = c[0];
        }
    }
}