#include "ArrayMatcher.cuh"
#include <thrust/execution_policy.h>
#include <device_launch_parameters.h>
#include <thrust/extrema.h>

__global__ void BuildIPsList(int Count, unsigned int* MaxIP, unsigned int *MinIP,  int *Lenghts, unsigned char *IPData)
{
	int mask = blockIdx.x * blockDim.x + threadIdx.x;

	while (mask < Count)
	{
		unsigned int address = 0;
		Lenghts[mask] = IPData[mask * 5 + 4];

		unsigned int part;
		for (int i = 0; i < 4; ++i)
		{
			part = IPData[mask * 5 + i];
			address |= part << (8 * (3 - i));
		}

		MinIP[mask] = address;
		MaxIP[mask] = ((1 << (32 - Lenghts[mask])) - 1) | address;

		mask += blockDim.x * gridDim.x;
	}
}

__global__ void FillArray(int *Array, int ArraySize, unsigned int *MaxIP, unsigned int *MinIP,  int *Lenghts, unsigned int *indexes, int MaxLenght, int MinLenght, int Count)
{
	unsigned int firstIndex = (ArraySize / gridDim.x) * blockIdx.x;
	unsigned int lastIndex = ((ArraySize / gridDim.x) * (blockIdx.x + 1)) - 1;

	for(unsigned int lenght = MaxLenght; lenght >= MinLenght; --lenght)
	{
		for(int entry = 0; entry < Count; ++entry)
		{
			if(Lenghts[entry] == lenght)
			{
				unsigned int start = MinIP[entry];
				start = (start >> (32 - MaxLenght)) & ((1 << MaxLenght) - 1);

				if (start < firstIndex)
					start = firstIndex;

				unsigned int end = MaxIP[entry];
				end = (end >> (32 - MaxLenght)) & ((1 << MaxLenght) - 1);

				if (end > lastIndex)
					end = lastIndex;

				unsigned int index = start + threadIdx.x;
				while (index <= end)
				{
					Array[index] = indexes[entry];
					index += blockDim.x;
				}
			}
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

	unsigned int *MaxIP;
	unsigned int *MinIP;
	int *Lenghts;
	GpuAssert(cudaMalloc((void**)&MaxIP, set.Size * sizeof(unsigned int)), "Cannot allocate MaxIP memory");
	GpuAssert(cudaMalloc((void**)&MinIP, set.Size * sizeof(unsigned int)), "Cannot allocate MinIP memory");
	GpuAssert(cudaMalloc((void**)&Lenghts, set.Size * sizeof(unsigned int)), "Cannot allocate Lenghts memory");

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

	unsigned int *indexes;
	GpuAssert(cudaMalloc((void**)&indexes, set.Size * sizeof(unsigned int)), "Cannot allocate indexes memory");
	thrust::sequence(thrust::device, indexes, indexes + set.Size);

	FillArray << <Setup.Blocks, Setup.Threads >> > (Array, ArraySize, MaxIP, MinIP, Lenghts, indexes, MaxLenght, MinLenght, set.Size);
	GpuAssert(cudaGetLastError(), "Error while launching FillArray kernel");
	GpuAssert(cudaDeviceSynchronize(), "Error while running FillArray kernel");

	//cleanup
	//unsigned int *maxIP = new unsigned int[set.Size];
	//unsigned int *minIP = new unsigned int[set.Size];
	//int *len = new int[set.Size];
	//cudaMemcpy(maxIP, MaxIP, set.Size * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	//cudaMemcpy(minIP, MinIP, set.Size * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	//cudaMemcpy(len, Lenghts, set.Size * sizeof(int), cudaMemcpyDeviceToHost);

	//for (int i = 0; i < set.Size; ++i)
	//	cout << maxIP[i] << "  " << minIP[i] << "   " << len[i] << endl;

	//delete[] minIP;
	//delete[] maxIP;
	//delete[] len;

	//int *a = new int[ArraySize];
	//int c = 0;
	//cudaMemcpy(a, Array, ArraySize * sizeof(int), cudaMemcpyDeviceToHost);

	//for (int i = 0; i < ArraySize; ++i)
	//{
	//	if (a[i] != -1)
	//	{
	//		//cout << a[i] << "  " << i << endl;
	//		++c;
	//	}
	//}
	//delete[] a;
	//cout << "Counter:   " << c << endl;

	GpuAssert(cudaFree(indexes), "Cannot free indexes memory");
	GpuAssert(cudaFree(MaxIP), "Cannot free MaxIP memory");
	GpuAssert(cudaFree(MinIP), "Cannot free MinIP memory");
	GpuAssert(cudaFree(Lenghts), "Cannot free Lenghts memory");

	ModelBuildTime = timer.Stop();
	GpuAssert(cudaSetDevice(0), "Cannot set cuda device in IPSet RandomSubset.");
}

__global__ void MatchWithArray(int *Array, int Count, unsigned char *ips, int *result, int MaxLenght)
{
	int ip = blockIdx.x * blockDim.x + threadIdx.x;
	int address, part;

	while (ip < Count)
	{
		address = 0;
		address |= ips[ip * 5 + 0] << (8 * (3 - 0));
		address |= ips[ip * 5 + 1] << (8 * (3 - 1));
		address |= ips[ip * 5 + 2] << (8 * (3 - 2));
		address |= ips[ip * 5 + 3] << (8 * (3 - 3));

		address = (address >> (32 - MaxLenght)) & ((1 << MaxLenght)-1);

		result[ip] = Array[address];

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

	MatchWithArray << <Setup.Blocks, Setup.Threads >> > (Array, set.Size, set.d_IPData, d_Result, MaxLenght);
	GpuAssert(cudaGetLastError(), "Error while launching FillArray kernel");
	GpuAssert(cudaDeviceSynchronize(), "Error while running FillArray kernel");

	GpuAssert(cudaMemcpy(result.MatchedMaskIndex, d_Result, result.IpsToMatchCount * sizeof(int), cudaMemcpyDeviceToHost), "Cannot copy Result data");
	GpuAssert(cudaFree(d_Result), "Cannot free Result memory");

	result.MatchingTime = timer.Stop();
	return result;
}