#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define workGroupSize 128

__kernel void max_prefix_sum(__global const int* sums,
    __global const int* maximums,
    __global const int* indices,
    __global int* sums_prime,
    __global int* maximums_prime,
    __global int* indices_prime,
    int n,
    int block_size)
{
    __local int local_sums[workGroupSize];
    __local int local_maximums[workGroupSize];
    __local int local_indices[workGroupSize];
    
    const unsigned index = get_global_id(0);
    const unsigned local_index = get_local_id(0);
    const unsigned group_index = get_group_id(0);

    local_sums[local_index] = (index < n) ? sums[index] : 0;
    local_maximums[local_index] = (index < n) ? maximums[index] : 0;
    local_indices[local_index] = (index < n) ? indices[index] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    if ((index < n) && (local_index == 0))
    {
        int sum = 0;
        int maximum = 0;
        int index = 0;
        for (int i = 0; i < workGroupSize; ++i)
        {
            int new_sum = sum + local_maximums[i];
            if (new_sum > maximum)
            {
                maximum = new_sum;
                index = local_indices[i] + i * block_size;
            }
            sum += local_sums[i];
        }
        sums_prime[group_index] = sum;
        maximums_prime[group_index] = maximum;
        indices_prime[group_index] = index;
    }
}