#include "memory_analysis.h"

// Global Memory Kernel
__global__ void globalMemorySearch(int *input, int *found_indexes, int numElements, int numThreads, int search_value)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadId < numThreads)
    {
        int thread_span = numElements / numThreads;
        int offset = threadId * thread_span;

        for (int i = offset; i < offset + thread_span && i < numElements; ++i)
        {
            input[i] += 1;
            if (input[i] == search_value)
            {
                found_indexes[i] = 1;
            }
        }
    }
}

// Constant Memory Kernel
__global__ void constantMemorySearch(int *found_indexes)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadId < constant_num_threads)
    {
        int thread_span = constant_num_elements / constant_num_threads;
        int offset = threadId * thread_span;

        for (int i = offset; i < offset + thread_span && i < constant_num_elements; ++i)
        {
            int value = constant_input[i] + 1;
            if (value == constant_search_value)
            {
                found_indexes[i] = 1;
            }
        }
    }
}

// Shared Memory Kernel
__global__ void sharedMemorySearch(int *input, int *found_indexes, int numElements, int numThreads, int search_value)
{
    extern __shared__ int sharedInput[];
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadId < numThreads)
    {
        int thread_span = numElements / numThreads;
        int offset = threadId * thread_span;

        for (int i = 0; i < thread_span && offset + i < numElements; ++i)
        {
            sharedInput[i] = input[offset + i];
        }
        __syncthreads();

        for (int i = 0; i < thread_span && offset + i < numElements; ++i)
        {
            sharedInput[i] += 1;
            if (sharedInput[i] == search_value)
            {
                found_indexes[offset + i] = 1;
            }
        }
        __syncthreads();

        for (int i = 0; i < thread_span && offset + i < numElements; ++i)
        {
            input[offset + i] = sharedInput[i];
        }
    }
}

// Register Memory Kernel
__global__ void registerMemorySearch(int *input, int *found_indexes, int numElements, int numThreads, int search_value)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadId < numThreads)
    {
        int thread_span = numElements / numThreads;
        int offset = threadId * thread_span;

        for (int i = offset; i < offset + thread_span && i < numElements; ++i)
        {
            int value = input[i] + 1;
            if (value == search_value)
            {
                found_indexes[i] = 1;
            }
        }
    }
}

__host__ int *allocatePageableRandomHostMemory(int numElements)
{
    srand(time(0));
    int *data = new int[numElements];

    for (int i = 0; i < numElements; ++i)
    {
        data[i] = rand() % 255;
    }

    return data;
}

__host__ int *allocateDeviceMemory(int numElements)
{
    int *d_data;
    cudaMalloc(&d_data, numElements * sizeof(int));
    return d_data;
}

__host__ void copyFromHostToDevice(std::string kernelType, int *input, int numElements, int numThreads, int *d_input)
{
    cudaMemcpy(d_input, input, numElements * sizeof(int), cudaMemcpyHostToDevice);

    if (kernelType == "constant")
    {
        int threadSpan = numElements / numThreads;
        cudaMemcpyToSymbol(constant_input, d_input, numElements * sizeof(int));
        cudaMemcpyToSymbol(constant_num_elements, &numElements, sizeof(int));
        cudaMemcpyToSymbol(constant_num_threads, &numThreads, sizeof(int));
        cudaMemcpyToSymbol(constant_thread_span, &threadSpan, sizeof(int));
    }
}

__host__ void executeKernel(int *d_input, int *found_indexes, int numElements, int threadsPerBlock, std::string kernelType)
{
    int search_value = 4; // Example search value
    int totalThreads = threadsPerBlock;

    if (kernelType == "global")
    {
        globalMemorySearch<<<1, threadsPerBlock>>>(d_input, found_indexes, numElements, totalThreads, search_value);
    }
    else if (kernelType == "constant")
    {
        constantMemorySearch<<<1, threadsPerBlock>>>(found_indexes);
    }
    else if (kernelType == "shared")
    {
        sharedMemorySearch<<<1, threadsPerBlock, numElements * sizeof(int)>>>(d_input, found_indexes, numElements, totalThreads, search_value);
    }
    else if (kernelType == "register")
    {
        registerMemorySearch<<<1, threadsPerBlock>>>(d_input, found_indexes, numElements, totalThreads, search_value);
    }
    cudaDeviceSynchronize();
}

__host__ void deallocateMemory(int *d_input, int *found_indexes)
{
    cudaFree(d_input);
    cudaFree(found_indexes);
}

__host__ void cleanUpDevice()
{
    cudaDeviceReset();
}

__host__ std::tuple<int, std::string, int, std::string> parseCommandLineArguments(int argc, char *argv[])
{
    int elementsPerThread = 2;
    int threadsPerBlock = 128;
    std::string currentPartId = "test";
    std::string kernelType = "global";

    for (int i = 1; i < argc; i += 2)
    {
        std::string option(argv[i]);
        std::string value(argv[i + 1]);

        if (option == "-t")
        {
            threadsPerBlock = std::stoi(value);
        }
        else if (option == "-m")
        {
            elementsPerThread = std::stoi(value);
        }
        else if (option == "-p")
        {
            currentPartId = value;
        }
        else if (option == "-k")
        {
            kernelType = value;
        }
    }

    return {elementsPerThread, currentPartId, threadsPerBlock, kernelType};
}

int main(int argc, char *argv[])
{
    auto [elementsPerThread, currentPartId, threadsPerBlock, kernelType] = parseCommandLineArguments(argc, argv);

    int numElements = elementsPerThread * threadsPerBlock;
    int *input = allocatePageableRandomHostMemory(numElements);
    int *d_input = allocateDeviceMemory(numElements);
    int *found_indexes = allocateDeviceMemory(numElements);

    copyFromHostToDevice(kernelType, input, numElements, threadsPerBlock, d_input);

    executeKernel(d_input, found_indexes, numElements, threadsPerBlock, kernelType);

    deallocateMemory(d_input, found_indexes);
    cleanUpDevice();

    std::cout << "partId: " << currentPartId
              << " elements: " << elementsPerThread
              << " threads: " << threadsPerBlock
              << " kernel: " << kernelType << std::endl;

    delete[] input;
    return 0;
}
