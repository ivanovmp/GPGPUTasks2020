#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <vector>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

std::string get_endian_little_info(const cl_bool endian_little)
{
    if (endian_little == CL_TRUE)
        return "This OpenCL device is a little endian device";
    else
        return "This OpenCL device is not a little endian device";
}

std::string get_error_message(const int code)
{
    static std::unordered_map<cl_device_type, std::string> m =
    {
        { -30, "CL_INVALID_VALUE" }
    };
    return m[code];
}

std::string join(std::vector<std::string>& strings)
{
    if (strings.size() == 0)
        return "";
    std::string ans = strings[0];
    for (int i = 1; i < strings.size(); ++i)
        ans += std::string() + ", " + strings[i];
    return ans;
}

std::string get_opencl_device_type_name(const cl_device_type device_type)
{
    static std::vector<std::pair<cl_device_type, std::string>> m =
    {
        { CL_DEVICE_TYPE_CPU, "CPU" },
        { CL_DEVICE_TYPE_GPU, "GPU" },
        { CL_DEVICE_TYPE_ACCELERATOR, "accelerator" },
        { CL_DEVICE_TYPE_DEFAULT, "default" },
    };
    std::vector<std::string> strings;
    for (auto device_type_pair : m)
        if (device_type_pair.first & device_type)
            strings.push_back(device_type_pair.second);
    return join(strings);
}

template <typename T>
std::string to_string(T value)
{
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

void reportError(cl_int err, const std::string &filename, int line)
{
    if (CL_SUCCESS == err)
        return;

    // Таблица с кодами ошибок:
    // libs/clew/CL/cl.h:103
    // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
    std::string message = "OpenCL error code " + to_string(err) + " (" + get_error_message(err) + ") encountered at " + filename + ":" + to_string(line);
    throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)


int main()
{
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку libs/clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    // Откройте 
    // https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/
    // Нажмите слева: "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformIDs"
    // Прочитайте документацию clGetPlatformIDs и убедитесь что этот способ узнать сколько есть платформ соответствует документации:
    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;

    // Тот же метод используется для того чтобы получить идентификаторы всех платформ - сверьтесь с документацией, что это сделано верно:
    std::vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));
    bool whined = false;
    for (int platformIndex = 0; platformIndex < platformsCount; ++platformIndex) {
        std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
        cl_platform_id platform = platforms[platformIndex];

        // Откройте документацию по "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformInfo"
        // Не забывайте проверять коды ошибок с помощью макроса OCL_SAFE_CALL
        size_t platformNameSize = 0;
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));
        // TODO 1.1
        // Попробуйте вместо CL_PLATFORM_NAME передать какое-нибудь случайное число - например 239
        // Т.к. это некорректный идентификатор параметра платформы - то метод вернет код ошибки
        // Макрос OCL_SAFE_CALL заметит это, и кинет ошибку с кодом
        // Откройте таблицу с кодами ошибок:
        // libs/clew/CL/cl.h:103
        // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
        // Найдите там нужный код ошибки и ее название
        // Затем откройте документацию по clGetPlatformInfo и в секции Errors найдите ошибку, с которой столкнулись
        // в документации подробно объясняется, какой ситуации соответствует данная ошибка, и это позволит проверив код понять чем же вызвана данная ошибка (не корректным аргументом param_name)
        // Обратите внимание что в этом же libs/clew/CL/cl.h файле указаны всевоможные defines такие как CL_DEVICE_TYPE_GPU и т.п.
        try
        {
            OCL_SAFE_CALL(clGetPlatformInfo(platform, std::hash<std::string>()(std::string("Java — всё ещё ужасный язык")), 0, nullptr, &platformNameSize));
        }
        catch (std::exception e)
        {
            std::cout << e.what() << '\n';
        }
        // TODO 1.2
        // Аналогично тому как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
        std::vector<unsigned char> platformName(platformNameSize, 0);
        // clGetPlatformInfo(...);
        // А вот и не "clGetPlatformInfo(...);"!!! На самом деле " OCL_SAFE_CALL(clGetPlatformInfo(...));".
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, &platformName[0], nullptr));
        std::cout << "    Platform name: " << platformName.data() << std::endl;

        // TODO 1.3
        // Запросите и напечатайте так же в консоль вендора данной платформы
        size_t platformVendorSize = 0;
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &platformVendorSize));
        std::string platformVendor(platformVendorSize, '\0'); // в задании попросили так же, но я хочу попробовать по-другому, через string, а не через vector
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, platformVendorSize, &platformVendor[0], nullptr));
        std::cout << "    Platform vendor: " << platformVendor << '\n'; // endl - зло

        // TODO 2.1
        // Запросите число доступных устройств данной платформы (аналогично тому как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
        std::vector<cl_device_id> devices(devicesCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, &devices[0], nullptr));
        std::cout << "    List of devices:" << '\n';
        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            // TODO 2.2
            // Запросите и напечатайте в консоль:
            // - Название устройства
            // - Тип устройства (видеокарта/процессор/что-то странное)
            // - Размер памяти устройства в мегабайтах
            // - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
            cl_device_id device = devices[deviceIndex];
            std::cout << "        Device " << deviceIndex + 1 << ":" << '\n';
            size_t device_name_size = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &device_name_size));
            std::string device_name(device_name_size, '\0');
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, device_name_size, &device_name[0], nullptr));
            std::cout << "            Name: " << device_name << '\n';

            cl_device_type device_type = CL_DEVICE_TYPE_CPU;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, 32, &device_type, nullptr));
            std::cout << "            Type: " << get_opencl_device_type_name(device_type) << '\n';

            cl_ulong memory_size_in_bytes = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, 64, &memory_size_in_bytes, nullptr));
            std::cout << "            Memory: " << memory_size_in_bytes / 1048576.0 << " MB" << '\n';
            
            if (!whined)
            {
                std::cout << "            Nu zachem echsho paru svoystv? Uzhe naskuchilo." << '\n';
                whined = true;
            }
            std::cout << "            This device has not any other interesting properties." << '\n';
            std::cout << "            Dull properties:" << '\n';
             
            cl_bool endian_little = CL_FALSE;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_ENDIAN_LITTLE, 32, &endian_little, nullptr));
            std::cout << "                " << get_endian_little_info(endian_little) << '\n';

            size_t timer_resolution = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_PROFILING_TIMER_RESOLUTION, 64, &timer_resolution, nullptr));
            std::cout << "                The resolution of device timer: " << timer_resolution << " ns" << '\n';
        }
    }

    return 0;
}