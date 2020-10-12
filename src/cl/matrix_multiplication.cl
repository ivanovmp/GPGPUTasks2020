#line 2
#define TILE_SIZE 16
__kernel void matrix_multiplication(__global float* a, __global float* b, __global float* c,
                                    const unsigned m, const unsigned k, const unsigned n)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    __local float tile_a[TILE_SIZE * TILE_SIZE];
    __local float tile_b[TILE_SIZE * TILE_SIZE];
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);
    float sum = 0;
    for (int t = 0; t < k; t += TILE_SIZE)
    {
        barrier(CLK_LOCAL_MEM_FENCE);

        int a_i = k + local_i;
        int a_j = j;
        tile_a[local_j * TILE_SIZE + local_i] = (a_i < k && a_j < m) ? a[a_j * k + a_i] : 0;

        int b_i = i;
        int b_j = k + local_j;
        tile_b[local_j * TILE_SIZE + local_i] = (b_i < n && b_j < k) ? b[b_j * n + b_i] : 0;
        
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k)
            sum += tile_a[local_j * TILE_SIZE + k] * tile_b[k * TILE_SIZE + local_i];
    }
    if (i < n && j < m)
        c[j * n + i] = sum;
}