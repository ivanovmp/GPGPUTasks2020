#line 2
__kernel void merge(__global const float* a, __global float* a_temp, unsigned n, unsigned S)
{
    const unsigned index = get_global_id(0);
    if (index >= n)
        return;
    const unsigned first_array_start = index >> (S + 1) << (S + 1);
    const unsigned second_array_start = first_array_start + (1 << S);
    if (second_array_start >= n)
        a_temp[index] = a[index];
    else
    {
        unsigned second_array_end = second_array_start + (1 << S);
        if (second_array_end > n)
            second_array_end = n;
        unsigned minimum_index = first_array_start;
        unsigned maximum_index = second_array_start + 1;
        unsigned semidifference = (maximum_index - minimum_index) >> 1;
        while (semidifference)
        {
            semidifference += minimum_index;
            unsigned second_index = index + second_array_start - semidifference;
            if (second_index < second_array_start)
                maximum_index = semidifference;
            else if (second_index >= second_array_end)
                minimum_index = semidifference;
            else if (a[semidifference - 1] < a[second_index])
                minimum_index = semidifference;
            else
                maximum_index = semidifference;
            semidifference = (maximum_index - minimum_index) >> 1;
        }
        unsigned second_index = index + second_array_start - minimum_index;
        if (minimum_index >= second_array_start)
            a_temp[index] = a[second_index];
        else if (second_index < second_array_start || second_index >= second_array_end)
            a_temp[index] = a[minimum_index];
        else
            a_temp[index] = min(a[second_index], a[minimum_index]);
    }
}