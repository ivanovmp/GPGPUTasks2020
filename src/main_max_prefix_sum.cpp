#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/max_prefix_sum_cl.h"


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv)
{
    int benchmarkingIters = 10;
    int max_n = (1 << 24);

    for (int n = 2; n <= max_n; n *= 2) {
        std::cout << "______________________________________________" << std::endl;
        int values_range = std::min(1023, std::numeric_limits<int>::max() / n);
        std::cout << "n=" << n << " values in range: [" << (-values_range) << "; " << values_range << "]" << std::endl;

        std::vector<int> as(n, 0);
        FastRandom r(n);
        for (int i = 0; i < n; ++i) {
            as[i] = (unsigned int) r.next(-values_range, values_range);
        }

        int reference_max_sum;
        int reference_result;
        {
            int max_sum = 0;
            int sum = 0;
            int result = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
                if (sum > max_sum) {
                    max_sum = sum;
                    result = i + 1;
                }
            }
            reference_max_sum = max_sum;
            reference_result = result;
        }
        std::cout << "Max prefix sum: " << reference_max_sum << " on prefix [0; " << reference_result << ")" << std::endl;

        {
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                int max_sum = 0;
                int sum = 0;
                int result = 0;
                for (int i = 0; i < n; ++i) {
                    sum += as[i];
                    if (sum > max_sum) {
                        max_sum = sum;
                        result = i + 1;
                    }
                }
                EXPECT_THE_SAME(reference_max_sum, max_sum, "CPU result should be consistent!");
                EXPECT_THE_SAME(reference_result, result, "CPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "CPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            // TODO: implement on OpenCL
            as.push_back(0);
            int n0 = n + 1;
            gpu::Context context;
            gpu::Device device = gpu::chooseGPUDevice(argc, argv);
            context.init(device.device_id_opencl);
            context.activate();
            gpu::gpu_mem_32i a_gpu, b_gpu, c_gpu;
            std::vector<int> v(n0);
            a_gpu.resizeN(n0);
            a_gpu.writeN(as.data(), n0);
            b_gpu.resizeN(n0);
            b_gpu.writeN(v.data(), n0);
            c_gpu.resizeN(n0);
            c_gpu.writeN(v.data(), n0);
            ocl::Kernel max_prefix_sum(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "max_prefix_sum");
            max_prefix_sum.compile();
            const unsigned int workGroupSize = 128;
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter)
            {
                unsigned int current_size = n0;
                int block_size = 1;
                gpu::gpu_mem_32i* as_gpu = &a_gpu;
                gpu::gpu_mem_32i* bs_gpu = &b_gpu;
                gpu::gpu_mem_32i* cs_gpu = &c_gpu;
                while (current_size > 1)
                {
                    unsigned int new_size = (current_size - 1) / workGroupSize + 1;
                    gpu::gpu_mem_32i* as_gpu_prime = new gpu::gpu_mem_32i();
                    as_gpu_prime->resizeN(new_size);
                    gpu::gpu_mem_32i* bs_gpu_prime = new gpu::gpu_mem_32i();
                    bs_gpu_prime->resizeN(new_size);
                    gpu::gpu_mem_32i* cs_gpu_prime = new gpu::gpu_mem_32i();
                    cs_gpu_prime->resizeN(new_size);
                    unsigned int global_work_size = (n0 + workGroupSize - 1) / workGroupSize * workGroupSize;
                    max_prefix_sum.exec(gpu::WorkSize(workGroupSize, global_work_size), *as_gpu, *bs_gpu, *cs_gpu,
                        *as_gpu_prime, *bs_gpu_prime, *cs_gpu_prime, current_size, block_size);
                    block_size *= workGroupSize;
                    if (n0 != current_size)
                    {
                        delete as_gpu;
                        delete bs_gpu;
                        delete cs_gpu;
                    }
                    as_gpu = as_gpu_prime;
                    bs_gpu = bs_gpu_prime;
                    cs_gpu = cs_gpu_prime;
                    current_size = new_size;
                }
                int sum, max_sum, result;
                as_gpu->readN(&sum, 1);
                bs_gpu->readN(&max_sum, 1);
                cs_gpu->readN(&result, 1);
                if (n0 != current_size)
                {
                    delete as_gpu;
                    delete bs_gpu;
                    delete cs_gpu;
                }
                EXPECT_THE_SAME(reference_max_sum, max_sum, "GPU result should be consistent!");
                EXPECT_THE_SAME(reference_result, result, "GPU result should be consistent!");
                t.nextLap();
            }
            as.pop_back();
            std::cout << "GPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU:     " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }
}
