#include "ArrayMatcher.cuh"
#include <thrust/execution_policy.h>
#include <device_launch_parameters.h>

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

void ArrayMatcher::BuildModel(IPSet& set)
{
	Setup = set.Setup;
	GpuAssert(cudaSetDevice(Setup.DeviceID), "Cannot set cuda device in IPSet RandomSubset.");
	Timer timer;
	timer.Start();

	GpuAssert(cudaMalloc((void**)&MaxIP, set.Size * sizeof(int)), "Cannot allocate MaxIP memory");
	GpuAssert(cudaMalloc((void**)&MinIP, set.Size * sizeof(int)), "Cannot allocate MinIP memory");
	GpuAssert(cudaMalloc((void**)&Lenghts, set.Size * sizeof(int)), "Cannot allocate Lenghts memory");

	BuildIPsList << <Setup.Blocks, Setup.Threads >> >(set.Size, MaxIP, MinIP, Lenghts, set.d_IPData);
	GpuAssert(cudaGetLastError(), "Error while launching BuildIPsList kernel");
	GpuAssert(cudaDeviceSynchronize(), "Error while running BuildIPsList kernel");

	//GpuAssert(cudaMalloc((void**)&Array, (~0) * sizeof(int)), "Cannot allocate Array memory");

	int *indexes;
	GpuAssert(cudaMalloc((void**)&indexes, set.Size * sizeof(int)), "Cannot allocate indexes memory");
	thrust::sequence(thrust::device, indexes, indexes + set.Size);

	thrust::sort_by_key(thrust::device, Lenghts, Lenghts + set.Size, indexes);

	GpuAssert(cudaFree(indexes), "Cannot free indexes memory");

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