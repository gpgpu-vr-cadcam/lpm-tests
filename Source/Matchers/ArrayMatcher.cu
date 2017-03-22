#include "ArrayMatcher.cuh"
#include <thrust/execution_policy.h>

void ArrayMatcher::BuildModel(IPSet& set)
{
	Setup = set.Setup;
	GpuAssert(cudaSetDevice(Setup.DeviceID), "Cannot set cuda device in IPSet RandomSubset.");
	Timer timer;
	timer.Start();

	//TODO: Implementacja

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