#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/sum_cl.h"


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char** argv)
{
    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100 * 1000 * 1000;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (unsigned int)r.next(std::numeric_limits<unsigned int>::max() / n);
        reference_sum += as[i];
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU:     " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
#pragma omp parallel for reduction(+:sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        // TODO: implement on OpenCL
        gpu::Context context;
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        context.init(device.device_id_opencl);
        context.activate();
        gpu::gpu_mem_32u* a_gpu = new gpu::gpu_mem_32u();
        a_gpu->resizeN(n);
        a_gpu->writeN(as.data(), n);
        ocl::Kernel sum(sum_kernel, sum_kernel_length, "sum");
        sum.compile();
        const unsigned int workGroupSize = 128;

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter)
        {
            unsigned int current_size = n;
            gpu::gpu_mem_32u* as_gpu = a_gpu;
            while (current_size > 1)
            {
                unsigned int new_size = (current_size - 1) / workGroupSize + 1;
                gpu::gpu_mem_32u* bs_gpu = new gpu::gpu_mem_32u();
                bs_gpu->resizeN(new_size);
                unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
                sum.exec(gpu::WorkSize(workGroupSize, global_work_size), *as_gpu, *bs_gpu, current_size);
                if (n != current_size)
                    delete as_gpu;
                as_gpu = bs_gpu;
                current_size = new_size;
            }
            unsigned sum;
            as_gpu->readN(&sum, 1);
            if (n != current_size)
                delete as_gpu;
            EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "GPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU:     " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }
}
