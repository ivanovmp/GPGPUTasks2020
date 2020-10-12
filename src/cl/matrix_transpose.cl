#line 2
#define TILE_SIZE 16
__kernel void matrix_transpose(__global float* a, __global float* at, const unsigned m, const unsigned k)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    __local float tile[TILE_SIZE * TILE_SIZE];
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);
    if (i < k && j < m)
        tile[local_j * TILE_SIZE + local_i] = a[j * k + i];
    barrier(CLK_LOCAL_MEM_FENCE);
    int new_i = i - local_i + local_j, new_j = j - local_j + local_i;
    if (new_i < k && new_j < m)
        at[new_i * m + new_j] = tile[local_i * TILE_SIZE + local_j];
}