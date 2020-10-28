#line 2
__kernel void distribute(__global const unsigned* a, __global unsigned* s, const int n, const int m, const int p, const int b)
{
    const unsigned index = get_global_id(0);
    if (index < n)
    {
        unsigned value = a[index];
        unsigned basket = (value >> p) & ((1 << b) - 1);
        for (int i = 0; i < (1 << b); ++i)
            s[index + i * m] = (i == basket);
    }
}
__kernel void distribute_smart(__global const unsigned* a, __global unsigned* s, const int n, const int m, const int p, const int b)
{
    __local unsigned mem[WORK_GROUP_SIZE];
    const unsigned index = get_global_id(0);
    const unsigned group = get_group_id(0);
    const unsigned local_index = get_local_id(0);
    if (index < n)
        mem[local_index] = (a[index] >> p) & ((1 << b) - 1);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_index < (1 << b))
    {
        int u = WORK_GROUP_SIZE, count = 0;
        if (u > n - WORK_GROUP_SIZE * group)
            u = n - WORK_GROUP_SIZE * group;
        for (int i = 0; i < u; ++i)
            if (mem[i] == local_index)
                ++count;
        s[group + local_index * m] = count;
    }
}
__kernel void create_block_sums(__global const unsigned* a, __global unsigned* s, const int n, const int M, const int m)
{
    __local unsigned mem[WORK_GROUP_SIZE];
    const unsigned index = get_global_id(0);
    const unsigned local_index = get_local_id(0);
    const unsigned bits = get_global_id(1);
    if (index < n)
        mem[local_index] = a[index + bits * M];
    else
        mem[local_index] = 0;
    for (int i = 1; i < WORK_GROUP_SIZE; i <<= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if ((local_index & ((i << 1) - 1)) == 0)
            mem[local_index] += mem[local_index + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_index == 0)
        s[get_group_id(0) + bits * m] = mem[0];
}
__kernel void add_zero(__global unsigned* a, const int b, __global unsigned* basket_sums)
{
    const unsigned index = get_global_id(0);
    __local unsigned mem[WORK_GROUP_SIZE];
    __local unsigned ans[WORK_GROUP_SIZE];
    ans[index] = 0;
    if (index < (1 << b))
    {
        unsigned basket_sum = a[WORK_GROUP_SIZE * index];
        a[WORK_GROUP_SIZE * index + 1] = basket_sum;
        if (index != (1 << b) - 1)
            mem[index + 1] = basket_sum;
        else
            mem[0] = 0;
    }
    else
        a[WORK_GROUP_SIZE * index + 1] = 0;
    a[WORK_GROUP_SIZE * index] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    unsigned current_sum = index + 1;
    for (unsigned i = 1; i <= (1 << b); i <<= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (current_sum & i)
            ans[index] += mem[current_sum ^= i];
        barrier(CLK_LOCAL_MEM_FENCE);
        unsigned new_index = 0;
        if (index & i)
            mem[index ^ i] += mem[index];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (index < (1 << b))
        basket_sums[index] = ans[index];
}
__kernel void make_prefix_sums(__global unsigned* a, __global const unsigned* s, const int n, const int M, const int m)
{
    __local unsigned mem[WORK_GROUP_SIZE];
    __local unsigned ans[WORK_GROUP_SIZE];
    const unsigned local_index = get_local_id(0);
    const unsigned index = get_global_id(0);
    const unsigned group = get_group_id(0);
    const unsigned bits = get_global_id(1);
    if (local_index == 0)
        mem[local_index] = s[group + bits * m];
    else if (index <= n)
        mem[local_index] = a[index - 1 + bits * M];
    else
        mem[local_index] = 0;
    ans[local_index] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    unsigned current_sum = local_index + 1;
    for (unsigned i = 1; i <= WORK_GROUP_SIZE; i <<= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (current_sum & i)
            ans[local_index] += mem[current_sum ^= i];
        barrier(CLK_LOCAL_MEM_FENCE);
        unsigned new_index = 0;
        if (local_index & i)
            mem[local_index ^ i] += mem[local_index];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    a[index + bits * M] = ans[local_index];
}
__kernel void radix_sort(__global const unsigned* a, __global unsigned* new_a, __global unsigned* s, __global const unsigned* basket_sums, const int n, const int m, const int p, const int b)
{
    const unsigned index = get_global_id(0);
    const unsigned local_index = get_local_id(0);
    const unsigned group = get_group_id(0);
    __local unsigned mem[WORK_GROUP_SIZE];
    if (index < n)
        mem[local_index] = a[index];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_index < (1 << k))
    {
        unsigned q = basket_sums[local_index] + s[group + local_index * m];
        // printf("basket_sums[%d] = %d, s[%d + %d * %d] = s[%d] = %d\n", local_index, basket_sums[local_index], group, local_index, m, group + local_index * m, s[group + local_index * m]);
        unsigned u = WORK_GROUP_SIZE;
        if (u > n - WORK_GROUP_SIZE * group)
            u = n - WORK_GROUP_SIZE * group;
        for (int i = 0; i < u; ++i)
        {
            unsigned value = mem[i];
            unsigned basket = (value >> p) & ((1 << b) - 1);
            if (basket == local_index)
                new_a[q++] = value;
        }
    }
}