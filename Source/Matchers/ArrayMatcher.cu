#include "ArrayMatcher.cuh"
#include <thrust/execution_policy.h>
#include <device_launch_parameters.h>
#include <thrust/extrema.h>

#define MASK8  ((1 << 8)-1)

__global__ void BuildIPsList(int Count, unsigned int* MaxIP, unsigned int *MinIP,  int *Lenghts, unsigned int *IPs)
{
	int mask = blockIdx.x * blockDim.x + threadIdx.x;

	while (mask < Count)
	{
		unsigned int address = IPs[mask];

		MinIP[mask] = address;
		MaxIP[mask] = ((1 << (32 - Lenghts[mask])) - 1) | address;

		mask += blockDim.x * gridDim.x;
	}
}

__global__ void FillArray(uchar3 *Array, int ArraySize, unsigned int *MaxIP, unsigned int *MinIP,  int *Lenghts, unsigned int *indexes, int MaxLenght, int MinLenght, int Count, int maskLenght)
{
	int entry = blockIdx.x;
	while(entry < Count)
	{
		if(Lenghts[entry] == maskLenght)
		{
			unsigned int start = MinIP[entry];
			start = (start >> (32 - MaxLenght)) & ((1 << MaxLenght) - 1);

			unsigned int end = MaxIP[entry];
			end = (end >> (32 - MaxLenght)) & ((1 << MaxLenght) - 1);

			unsigned int index = start + threadIdx.x;
			while (index <= end)
			{
				Array[index].x = (indexes[entry] >> 16) & MASK8;
				Array[index].y = (indexes[entry] >> 8) & MASK8;
				Array[index].z = indexes[entry] & MASK8;
				index += blockDim.x;
			}
		}

		entry += gridDim.x;
	}
}

void ArrayMatcher::BuildModel(IPSet& set)
{
	Setup = set.Setup;
	Timer timer;
	timer.Start();

	unsigned int *MaxIP;
	unsigned int *MinIP;
	int *Lenghts;
	GpuAssert(cudaMalloc((void**)&MaxIP, set.Size * sizeof(unsigned int)), "Cannot allocate MaxIP memory");
	GpuAssert(cudaMalloc((void**)&MinIP, set.Size * sizeof(unsigned int)), "Cannot allocate MinIP memory");

	GpuAssert(cudaMalloc((void**)&Lenghts, set.Size * sizeof(unsigned int)), "Cannot allocate Lenghts memory");
	GpuAssert(cudaMemcpy(Lenghts, set.d_Lenghts, set.Size * sizeof(int), cudaMemcpyDeviceToDevice), "Cannot copy Lenghts");

	BuildIPsList << <Setup.Blocks, Setup.Threads >> >(set.Size, MaxIP, MinIP, Lenghts, set.d_IPs);
	GpuAssert(cudaGetLastError(), "Error while launching BuildIPsList kernel");
	GpuAssert(cudaDeviceSynchronize(), "Error while running BuildIPsList kernel");

	int *maxLenghtPointer = thrust::max_element(thrust::device, Lenghts, Lenghts + set.Size);
	GpuAssert(cudaMemcpy(&MaxLenght, maxLenghtPointer, sizeof(int), cudaMemcpyDeviceToHost), "Cannot copy max lenght");

	int *minLenghtPointer = thrust::min_element(thrust::device, Lenghts, Lenghts + set.Size);
	GpuAssert(cudaMemcpy(&MinLenght, minLenghtPointer, sizeof(int), cudaMemcpyDeviceToHost), "Cannot copy min lenght");

	ArraySize = (1 << MaxLenght);
	GpuAssert(cudaMalloc((void**)&Array, ArraySize * sizeof(uchar3)), "Cannot allocate Array memory");
	thrust::fill_n(thrust::device, Array, ArraySize, EMPTY);

	unsigned int *indexes;
	GpuAssert(cudaMalloc((void**)&indexes, set.Size * sizeof(unsigned int)), "Cannot allocate indexes memory");
	thrust::sequence(thrust::device, indexes, indexes + set.Size);

	for(int i = MaxLenght; i >= MinLenght; --i)
	{
		FillArray << <Setup.Blocks, Setup.Threads >> > (Array, ArraySize, MaxIP, MinIP, Lenghts, indexes, MaxLenght, MinLenght, set.Size, i);
		GpuAssert(cudaGetLastError(), "Error while launching FillArray kernel");
		GpuAssert(cudaDeviceSynchronize(), "Error while running FillArray kernel");
	}

	//cleanup
	GpuAssert(cudaFree(indexes), "Cannot free indexes memory");
	GpuAssert(cudaFree(MaxIP), "Cannot free MaxIP memory");
	GpuAssert(cudaFree(MinIP), "Cannot free MinIP memory");
	GpuAssert(cudaFree(Lenghts), "Cannot free Lenghts memory");

	ModelBuildTime = timer.Stop();
}

__global__ void MatchWithArray(uchar3 *Array, int Count, unsigned int *ips, int *result, int MaxLenght)
{
	int ip = blockIdx.x * blockDim.x + threadIdx.x;
	int address;

	while (ip < Count)
	{
		address = ips[ip];
		address = (address >> (32 - MaxLenght)) & ((1 << MaxLenght)-1);

		if (Array[address].x != MASK8 || Array[address].y != MASK8 || Array[address].z != MASK8)
		{
			result[ip] = 0;
			result[ip] = result[ip] | ((unsigned int)Array[address].x) << 16;
			result[ip] = result[ip] | ((unsigned int)Array[address].y) << 8;
			result[ip] = result[ip] | ((unsigned int)Array[address].z);
		}

		ip += gridDim.x * blockDim.x;
	}
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

	MatchWithArray << <Setup.Blocks, Setup.Threads >> > (Array, set.Size, set.d_IPs, d_Result, MaxLenght);
	GpuAssert(cudaGetLastError(), "Error while launching FillArray kernel");
	GpuAssert(cudaDeviceSynchronize(), "Error while running FillArray kernel");

	GpuAssert(cudaMemcpy(result.MatchedMaskIndex, d_Result, result.IpsToMatchCount * sizeof(int), cudaMemcpyDeviceToHost), "Cannot copy Result data");
	GpuAssert(cudaFree(d_Result), "Cannot free Result memory");

	result.MatchingTime = timer.Stop();
	return result;
}