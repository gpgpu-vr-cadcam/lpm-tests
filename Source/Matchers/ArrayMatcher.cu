#include "ArrayMatcher.cuh"
#include <thrust/execution_policy.h>
#include <device_launch_parameters.h>
#include <thrust/extrema.h>

__global__ void BuildIPsList(int Count, int* MaxIP, int *MinIP, int *Lenghts, unsigned char *IPData)
{
	int mask = blockIdx.x * blockDim.x + threadIdx.x;

	while (mask < Count)
	{
		int address = 0;
		Lenghts[mask] = IPData[mask * 5 + 4];

		int part;
		for (int i = 0; i < 4; ++i)
		{
			part = IPData[mask * 5 + i];
			address |= part << (8 * (3 - i));
		}

		MinIP[mask] = address;
		MaxIP[mask] = ((1 << (32 - Lenghts[mask])) - 1) & address;

		mask += blockDim.x * gridDim.x;
	}
}

__global__ void FillArray(int *Array, int ArraySize, int *MaxIP, int *MinIP, int *Lenghts, int *indexes, int MaxLenght, int MinLenght, int Count)
{
	int firstIndex = ArraySize / gridDim.x * blockIdx.x;
	int lastIndex = (ArraySize / gridDim.x * (blockIdx.x + 1)) - 1;

	for(int lenght = MaxLenght; lenght >= MinLenght; --lenght)
	{
		int entry = blockIdx.x;
		while(entry < Count)
		{
			if(Lenghts[entry] == lenght)
			{
				int start = MinIP[entry];
				if (start < firstIndex)
					start = firstIndex;

				int end = MaxIP[entry];
				if (end > lastIndex)
					end = lastIndex;

				int index = start + threadIdx.x;
				while (index < end)
				{
					Array[index] = indexes[entry];
					index += blockDim.x;
				}
			}

			entry += gridDim.x;
		}

		__syncthreads();
	}
	
}

void ArrayMatcher::BuildModel(IPSet& set)
{
	Setup = set.Setup;
	GpuAssert(cudaSetDevice(Setup.DeviceID), "Cannot set cuda device in IPSet RandomSubset.");
	Timer timer;
	timer.Start();

	int *MaxIP;
	int *MinIP;
	int *Lenghts;
	GpuAssert(cudaMalloc((void**)&MaxIP, set.Size * sizeof(int)), "Cannot allocate MaxIP memory");
	GpuAssert(cudaMalloc((void**)&MinIP, set.Size * sizeof(int)), "Cannot allocate MinIP memory");
	GpuAssert(cudaMalloc((void**)&Lenghts, set.Size * sizeof(int)), "Cannot allocate Lenghts memory");

	BuildIPsList << <Setup.Blocks, Setup.Threads >> >(set.Size, MaxIP, MinIP, Lenghts, set.d_IPData);
	GpuAssert(cudaGetLastError(), "Error while launching BuildIPsList kernel");
	GpuAssert(cudaDeviceSynchronize(), "Error while running BuildIPsList kernel");

	int *maxLenghtPointer = thrust::max_element(thrust::device, Lenghts, Lenghts + set.Size);
	GpuAssert(cudaMemcpy(&MaxLenght, maxLenghtPointer, sizeof(int), cudaMemcpyDeviceToHost), "Cannot copy max lenght");

	int *minLenghtPointer = thrust::min_element(thrust::device, Lenghts, Lenghts + set.Size);
	GpuAssert(cudaMemcpy(&MinLenght, minLenghtPointer, sizeof(int), cudaMemcpyDeviceToHost), "Cannot copy min lenght");

	ArraySize = (1 << MaxLenght);
	GpuAssert(cudaMalloc((void**)&Array, ArraySize * sizeof(int)), "Cannot allocate Array memory");
	thrust::fill_n(thrust::device, Array, ArraySize, -1);

	int *indexes;
	GpuAssert(cudaMalloc((void**)&indexes, set.Size * sizeof(int)), "Cannot allocate indexes memory");
	thrust::sequence(thrust::device, indexes, indexes + set.Size);

	FillArray << <Setup.Blocks, Setup.Threads >> > (Array, ArraySize, MaxIP, MinIP, Lenghts, indexes, MaxLenght, MinLenght, set.Size);
	GpuAssert(cudaGetLastError(), "Error while launching FillArray kernel");
	GpuAssert(cudaDeviceSynchronize(), "Error while running FillArray kernel");

	//cleanup
	GpuAssert(cudaFree(indexes), "Cannot free indexes memory");
	GpuAssert(cudaFree(MaxIP), "Cannot free MaxIP memory");
	GpuAssert(cudaFree(MinIP), "Cannot free MinIP memory");
	GpuAssert(cudaFree(Lenghts), "Cannot free Lenghts memory");

	ModelBuildTime = timer.Stop();
	GpuAssert(cudaSetDevice(0), "Cannot set cuda device in IPSet RandomSubset.");
}

Result ArrayMatcher::Match(IPSet& set)
{
	Result result(set.Size);
	result.MatchedMaskIndex = new int[set.Size];

	Timer timer;
	timer.Start();

	int *d_Result;
	GpuAssert(cudaMalloc((void**)&d_Result, result.IpsToMatchCount * sizeof(int)), "Cannot allocate memory for Result");
	thrust::fill_n(thrust::device, d_Result, result.IpsToMatchCount, -1);

	//TODO: Implementacja

	GpuAssert(cudaMemcpy(result.MatchedMaskIndex, d_Result, result.IpsToMatchCount * sizeof(int), cudaMemcpyDeviceToHost), "Cannot copy Result data");
	GpuAssert(cudaFree(d_Result), "Cannot free Result memory");

	result.MatchingTime = timer.Stop();
	return result;
}