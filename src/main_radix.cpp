#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <random>


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

unsigned divide_round_up(const unsigned number, const unsigned work_group_size)
{
    return (number + work_group_size - 1) / work_group_size;
}

unsigned round_up(const unsigned number, const unsigned work_group_size)
{
    return divide_round_up(number, work_group_size) * work_group_size;
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)



int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 100;
    std::cout << "We will run " << benchmarkingIters << " iterations in each configuration.\n";
    unsigned int n = 32 * 1024 * 1024; // 32 * 1024 * 1024;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i)
        as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<unsigned int> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = as;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;
    }


    for (unsigned work_group_size = 128; work_group_size <= 256; work_group_size <<= 1)
    {
        std::cout << '\n';
        for (int k = 1; k <= 5; ++k) // No more than six!!!
        {
            const std::string defines = "-D WORK_GROUP_SIZE=" + std::to_string(work_group_size) + " -D k=" + std::to_string(k);
            ocl::Kernel distributor(radix_kernel, radix_kernel_length, "distribute", defines);
            ocl::Kernel smart_distributor(radix_kernel, radix_kernel_length, "distribute_smart", defines);
            ocl::Kernel block_creator(radix_kernel, radix_kernel_length, "create_block_sums", defines);
            ocl::Kernel zero_adder(radix_kernel, radix_kernel_length, "add_zero", defines);
            ocl::Kernel prefix_sums_creator(radix_kernel, radix_kernel_length, "make_prefix_sums", defines);
            ocl::Kernel radix_sorter(radix_kernel, radix_kernel_length, "radix_sort", defines);

            distributor.compile();
            smart_distributor.compile();
            block_creator.compile();
            zero_adder.compile();
            prefix_sums_creator.compile();
            radix_sorter.compile();

            gpu::gpu_mem_32u* as_gpu = new gpu::gpu_mem_32u(), * as_gpu_temp = new gpu::gpu_mem_32u();

            as_gpu->resizeN(n);
            as_gpu_temp->resizeN(n);
            std::vector<gpu::gpu_mem_32u> block_sums;
            std::vector<unsigned> block_numbers;
            std::vector<unsigned> memory_sizes;
            gpu::gpu_mem_32u basket_sums;
            basket_sums.resizeN(1 << k);
            int block_number = n;
            while (true)
            {
                block_sums.push_back(gpu::gpu_mem_32u());
                memory_sizes.push_back(round_up(block_number + 1, work_group_size));
                block_sums.back().resizeN(memory_sizes.back() << k);
                block_numbers.push_back(block_number);
                if (block_number == 1)
                    break;
                block_number = divide_round_up(block_number, work_group_size);
            }
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                as_gpu->writeN(as.data(), n);
                t.restart(); // Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных

                for (int p = 0; p < 32; p += k)
                {
                    int b = std::min(k, 32 - p);
                    // Надо отсортировать as_gpu по битам p, ..., p + b - 1
                    // Для начала надо осуществить подсчёт

                    // // Это способ с большим количеством обращений к внешней памяти. Так нельзя, так что мы будем его ускорять.
                    // distributor.exec(gpu::WorkSize(work_group_size, round_up(n, work_group_size)), *as_gpu, block_sums.front(), n, memory_sizes.front(), p, b);
                    // for (int i = 1; i < block_sums.size(); ++i)
                    //    block_creator.exec(gpu::WorkSize(work_group_size, 1, round_up(block_numbers[i - 1], work_group_size), 1 << b), block_sums[i - 1], block_sums[i], block_numbers[i - 1], memory_sizes[i - 1], memory_sizes[i]);


                    smart_distributor.exec(gpu::WorkSize(work_group_size, round_up(n, work_group_size)), *as_gpu, block_sums[1], n, memory_sizes[1], p, b);
                    for (int i = 2; i < block_sums.size(); ++i)
                        block_creator.exec(gpu::WorkSize(work_group_size, 1, round_up(block_numbers[i - 1], work_group_size), 1 << b), block_sums[i - 1], block_sums[i], block_numbers[i - 1], memory_sizes[i - 1], memory_sizes[i]);
                    zero_adder.exec(gpu::WorkSize(work_group_size, round_up(1 << b, work_group_size)), block_sums.back(), b, basket_sums);
                    for (int i = block_sums.size() - 1; i >= 2; --i)
                        prefix_sums_creator.exec(gpu::WorkSize(work_group_size, 1, memory_sizes[i - 1], 1 << b), block_sums[i - 1], block_sums[i], block_numbers[i - 1], memory_sizes[i - 1], memory_sizes[i]);
                    radix_sorter.exec(gpu::WorkSize(work_group_size, round_up(n, work_group_size)), *as_gpu, *as_gpu_temp, block_sums[1], basket_sums, n, memory_sizes[1], p, b);
                    std::swap(as_gpu, as_gpu_temp);
                }
                t.nextLap();
            }
            block_sums.clear();
            std::cout << "GPU, dividing into blocks of " << k << " bits with workGroup = " << work_group_size << ": " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU, dividing into blocks of " << k << " bits with workGroup = " << work_group_size << ": " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
            as_gpu->readN(as.data(), n);
            // Проверяем корректность результатов
            for (int i = 0; i < n; ++i) {
                EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
            }
            as_gpu->resizeN(1);
            as_gpu_temp->resizeN(1);
            delete as_gpu;
            delete as_gpu_temp;
        }
    }
    system("pause");
    return 0;
}
