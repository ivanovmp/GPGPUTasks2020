#line 2
__kernel void bitonic_left(__global float* as)
{
    __local float f[WORK_GROUP_SIZE << 1];
    unsigned k = get_global_id(0);
    unsigned local_k = get_local_id(0);
    unsigned g = get_group_id(0);
    f[local_k << 1] = as[k << 1];
    f[1 + (local_k << 1)] = as[1 + (k << 1)];
    for (int i = 0; i <= WORK_GROUP_SIZE_LOG; ++i)
        for (int j = i; j >= 0; --j)
        {
            barrier(CLK_LOCAL_MEM_FENCE);
            unsigned x = ((k & ((1 << j) - 1)) ^ ((k >> j) << (j + 1)) ^ (((k >> i) & 1) << j)) & ((WORK_GROUP_SIZE << 1) - 1);
            unsigned y = x ^ (1 << j);
            if (f[x] > f[y])
            {
                float z = f[x];
                f[x] = f[y];
                f[y] = z;
            }
        }
    barrier(CLK_LOCAL_MEM_FENCE);
    as[k << 1] = f[local_k << 1];
    as[1 + (k << 1)] = f[1 + (local_k << 1)];
}
__kernel void bitonic_middle(__global float* as, const unsigned j, const unsigned i)
{
    unsigned k = get_global_id(0);
    unsigned x = (k & ((1 << j) - 1)) ^ ((k >> j) << (j + 1)) ^ (((k >> i) & 1) << j);
    unsigned y = x ^ (1 << j);
    float asx = as[x];
    float asy = as[y];
    if (asx > asy)
    {
        as[x] = asy;
        as[y] = asx;
    }
}
__kernel void bitonic_right(__global float* as, i)
{
    __local float f[WORK_GROUP_SIZE << 1];
    unsigned k = get_global_id(0);
    unsigned local_k = get_local_id(0);
    f[local_k << 1] = as[k << 1];
    f[1 + (local_k << 1)] = as[1 + (k << 1)];
    for (int j = WORK_GROUP_SIZE_LOG; j >= 0; --j)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        unsigned x = ((k & ((1 << j) - 1)) ^ ((k >> j) << (j + 1)) ^ (((k >> i) & 1) << j)) & ((WORK_GROUP_SIZE << 1) - 1);
        unsigned y = x ^ (1 << j);
        if (f[x] > f[y])
        {
            float z = f[x];
            f[x] = f[y];
            f[y] = z;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    as[k << 1] = f[local_k << 1];
    as[1 + (k << 1)] = f[1 + (local_k << 1)];
}
