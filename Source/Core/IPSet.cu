#include "IPSet.cuh"
#include <fstream>
#include <string>
#include <vector>

#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

#include "device_launch_parameters.h"

IPSet& IPSet::operator=(const IPSet& other)
{
	if (this == &other)
		return *this;
	Size = other.Size;
	Setup = other.Setup;

	GpuAssert(cudaMalloc(reinterpret_cast<void**>(&d_IPs), Size * sizeof(unsigned int)), "Cannot init IPs device memory.");
	GpuAssert(cudaMalloc(reinterpret_cast<void**>(&d_Lenghts), Size * sizeof(int)), "Cannot init Lenghts device memory.");

	GpuAssert(cudaMemcpy(d_IPs, other.d_IPs, Size * sizeof(unsigned int), cudaMemcpyDeviceToDevice), "Cannot copy IPs to device memory in = operator.");
	GpuAssert(cudaMemcpy(d_Lenghts, other.d_Lenghts, Size * sizeof(int), cudaMemcpyDeviceToDevice), "Cannot copy Lenghts to device memory in = operator.");

	return *this;
}

IPSet::IPSet(const IPSet& other)
{
	if (this == &other)
		return;
	Size = other.Size;
	Setup = other.Setup;

	GpuAssert(cudaMalloc(reinterpret_cast<void**>(&d_IPs), Size * sizeof(unsigned int)), "Cannot init IPs device memory.");
	GpuAssert(cudaMalloc(reinterpret_cast<void**>(&d_Lenghts), Size * sizeof(int)), "Cannot init Lenghts device memory.");

	GpuAssert(cudaMemcpy(d_IPs, other.d_IPs, Size * sizeof(unsigned int), cudaMemcpyDeviceToDevice), "Cannot copy IPs to device memory in copy operator.");
	GpuAssert(cudaMemcpy(d_Lenghts, other.d_Lenghts, Size * sizeof(int), cudaMemcpyDeviceToDevice), "Cannot copy Lenghts to device memory in copy operator.");
}

void IPSet::Dispose()
{
	if(d_IPs != NULL)
	{
		GpuAssert(cudaFree(d_IPs), "Cannot free IPs memory in IPSet destructor");
		GpuAssert(cudaFree(d_Lenghts), "Cannot free Lenghts memory in IPSet destructor");
		d_IPs = NULL;
		d_Lenghts = NULL;
	}
}

__global__ void BuildIPs(unsigned char * ipData, unsigned int * ips, int *lenghts, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned char b1, b2, b3, b4;

	while (i < size)
	{
		b1 = ipData[i * 5];
		b2 = ipData[i * 5 + 1];
		b3 = ipData[i * 5 + 2];
		b4 = ipData[i * 5 + 3];

		ips[i] = (b1 << 24) + (b2 << 16) + (b3 << 8) + b4;
		lenghts[i] = ipData[i * 5 + 4];

		i += blockDim.x * gridDim.x;
	}
}

void IPSet::Load(GpuSetup &setup, string path, int count)
{
	Setup = setup;
	ifstream file(path);
	string line;
	string delims = ";.";
	vector<string> parts;
	int pos;

	int iteration = 0;
	while (!file.eof() && iteration < count)
	{
		file >> line;
		line = line.substr(4, line.size());

		pos = line.find(".");
		parts.push_back(line.substr(0, pos));
		line = line.substr(pos+1, line.size());

		pos = line.find(".");
		parts.push_back(line.substr(0, pos));
		line = line.substr(pos+1, line.size());

		pos = line.find(".");
		parts.push_back(line.substr(0, pos));
		line = line.substr(pos+1, line.size());

		pos = line.find(";");
		parts.push_back(line.substr(0, pos));
		line = line.substr(pos+1, line.size());

		parts.push_back(line);

		++iteration;
	}

	file.close();

	Size = parts.size() / 5;
	unsigned char *IPData = new unsigned char[Size * 5];

	for(int i = 0; i < Size * 5; ++i)
		IPData[i] = static_cast<unsigned char>(stoi(parts[i]));

	unsigned char *d_IPData;
	GpuAssert(cudaMalloc(reinterpret_cast<void**>(&d_IPData), 5 * Size * sizeof(unsigned char)), "Cannot init ip masks device memory.");
	GpuAssert(cudaMemcpy(d_IPData, IPData, 5 * Size * sizeof(unsigned char), cudaMemcpyHostToDevice), "Cannot copy ip masks to device memory.");

	GpuAssert(cudaMalloc(reinterpret_cast<void**>(&d_IPs), Size * sizeof(unsigned int)), "Cannot init IPs device memory.");
	GpuAssert(cudaMalloc(reinterpret_cast<void**>(&d_Lenghts), Size * sizeof(int)), "Cannot init Lenghts device memory.");

	BuildIPs << < Setup.Blocks, Setup.Threads >> > (d_IPData, d_IPs, d_Lenghts, Size);
	GpuAssert(cudaPeekAtLastError(), "Error while launching BuildIPs kernel");
	GpuAssert(cudaDeviceSynchronize(), "Error while running BuildIPs kernel");

	delete[] IPData;
	GpuAssert(cudaFree(d_IPData), "Cannot free d_IPData in Load");
}

void IPSet::Generate(GpuSetup& setup, int count)
{
	Setup = setup;
	Size = count;

	unsigned char *IPData = new unsigned char[Size * 5];

	int maskLenght;
	int mask;

	for (int i = 0; i < Size; ++i)
	{
		if ((rand() % 100) <= 60)
			maskLenght = 24;
		else if ((rand() % 100) <= 25)
			maskLenght = 16;
		else if ((rand() % 100) <= 33)
			maskLenght = 8;
		else
			maskLenght = rand() % 32;

		mask = (rand() | (rand() << 16)) << (32 - maskLenght);

		IPData[i * 5] = (mask >> 24) & 0xFF;
		IPData[i * 5 + 1] = (mask >> 16) & 0xFF;
		IPData[i * 5 + 2] = (mask >> 8) & 0xFF;
		IPData[i * 5 + 3] = mask & 0xFF;
		IPData[i * 5 + 4] = maskLenght;
	}

	unsigned char *d_IPData;
	GpuAssert(cudaMalloc(reinterpret_cast<void**>(&d_IPData), 5 * Size * sizeof(unsigned char)), "Cannot init ip masks device memory.");
	GpuAssert(cudaMemcpy(d_IPData, IPData, 5 * Size * sizeof(unsigned char), cudaMemcpyHostToDevice), "Cannot copy ip masks to device memory.");

	GpuAssert(cudaMalloc(reinterpret_cast<void**>(&d_IPs), Size * sizeof(unsigned int)), "Cannot init IPs device memory.");
	GpuAssert(cudaMalloc(reinterpret_cast<void**>(&d_Lenghts), Size * sizeof(int)), "Cannot init Lenghts device memory.");

	BuildIPs << < Setup.Blocks, Setup.Threads >> > (d_IPData, d_IPs, d_Lenghts, Size);
	GpuAssert(cudaPeekAtLastError(), "Error while launching BuildIPs kernel");
	GpuAssert(cudaDeviceSynchronize(), "Error while running BuildIPs kernel");

	delete[] IPData;
	GpuAssert(cudaFree(d_IPData), "Cannot free d_IPData in Load");
}

__global__ void CopyIPsToSubset(int *indexes, int subsetSize, 
	unsigned int *IPs, int *Lenghts, unsigned int *sourceSetIPs, int * sourceSetLenghts)
{
	int iteration = blockIdx.x * blockDim.x + threadIdx.x;

	while(iteration < subsetSize)
	{
		int sourceInd = indexes[iteration];

		IPs[iteration] = sourceSetIPs[sourceInd];
		Lenghts[iteration] = sourceSetLenghts[sourceInd];

		iteration += blockDim.x * gridDim.x;
	}
}

void IPSet::RandomSubset(int subsetSize, IPSet& sourceSet)
{
	Setup = sourceSet.Setup;
	Size = subsetSize;

	int *d_Indexes;
	int *d_RandomValues;
	
	GpuAssert(cudaMalloc(reinterpret_cast<void**>(&d_IPs), Size * sizeof(unsigned int)), "Cannot init IPs device memory.");
	GpuAssert(cudaMalloc(reinterpret_cast<void**>(&d_Lenghts), Size * sizeof(int)), "Cannot init Lenghts device memory.");

	int maxSize = subsetSize > sourceSet.Size ? subsetSize : sourceSet.Size;
	
	GpuAssert(cudaMalloc(reinterpret_cast<void**>(&d_Indexes), maxSize * sizeof(int)), "Cannot init indexes device memory.");
	GpuAssert(cudaMalloc(reinterpret_cast<void**>(&d_RandomValues), maxSize * sizeof(int)), "Cannot init random values device memory.");

	thrust::sequence(thrust::device, d_Indexes, d_Indexes + sourceSet.Size, 0);
	thrust::generate_n(thrust::device, d_Indexes + sourceSet.Size, maxSize - sourceSet.Size, Rnd(0, sourceSet.Size));

	thrust::generate_n(thrust::device, d_RandomValues, maxSize, Rnd(0, maxSize));
	thrust::stable_sort_by_key(thrust::device, d_RandomValues, d_RandomValues + maxSize, d_Indexes);
	thrust::sort(thrust::device, d_Indexes, d_Indexes + Size);

	CopyIPsToSubset << < Setup.Blocks, Setup.Threads >> > (d_Indexes, Size, d_IPs, d_Lenghts, sourceSet.d_IPs, sourceSet.d_Lenghts);
	GpuAssert(cudaPeekAtLastError(), "Error while launching CopyIPsToSubset kernel");
	GpuAssert(cudaDeviceSynchronize(), "Error while running CopyIPsToSubset kernel");

	GpuAssert(cudaFree(d_Indexes), "Cannot free indexes memory.");
	GpuAssert(cudaFree(d_RandomValues), "Cannot free random values memory.");
}

void IPSet::Sort()
{
	thrust::sort_by_key(thrust::device, d_IPs, d_IPs + Size, d_Lenghts);
}

void IPSet::Randomize()
{
	int *d_Indexes;
	int *d_RandomValues;

	GpuAssert(cudaMalloc(reinterpret_cast<void**>(&d_Indexes), Size * sizeof(int)), "Cannot init indexes device memory.");
	GpuAssert(cudaMalloc(reinterpret_cast<void**>(&d_RandomValues), Size * sizeof(int)), "Cannot init random values device memory.");

	thrust::sequence(thrust::device, d_Indexes, d_Indexes + Size, 0);
	thrust::generate_n(thrust::device, d_RandomValues, Size, Rnd(0, Size));

	unsigned int *new_IPs;
	int *new_Lenghts;
	GpuAssert(cudaMalloc(reinterpret_cast<void**>(&new_IPs), Size * sizeof(unsigned int)), "Cannot init IPs device memory.");
	GpuAssert(cudaMalloc(reinterpret_cast<void**>(&new_Lenghts), Size * sizeof(int)), "Cannot init Lenghts device memory.");

	CopyIPsToSubset << < Setup.Blocks, Setup.Threads >> > (d_Indexes, Size, new_IPs, new_Lenghts, d_IPs, d_Lenghts);
	GpuAssert(cudaPeekAtLastError(), "Error while launching CopyIPsToSubset kernel");
	GpuAssert(cudaDeviceSynchronize(), "Error while running CopyIPsToSubset kernel");

	GpuAssert(cudaFree(d_IPs), "Cannot free IPs memory.");
	GpuAssert(cudaFree(d_Lenghts), "Cannot free Lenghts values memory.");

	d_IPs = new_IPs;
	d_Lenghts = new_Lenghts;

	GpuAssert(cudaFree(d_Indexes), "Cannot free indexes memory.");
	GpuAssert(cudaFree(d_RandomValues), "Cannot free random values memory.");
}

IPSet operator+(IPSet& l, IPSet& r)
{
	IPSet set;

	if (l.Setup.DeviceID != r.Setup.DeviceID)
		throw runtime_error("Cannot add set from different devices");

	set.Size = l.Size + r.Size;
	set.Setup = l.Setup;

	GpuAssert(cudaMalloc(reinterpret_cast<void**>(&set.d_IPs), set.Size * sizeof(unsigned int)), "Cannot init IPs device memory.");
	GpuAssert(cudaMalloc(reinterpret_cast<void**>(&set.d_Lenghts), set.Size * sizeof(int)), "Cannot init Lenghts device memory.");

	GpuAssert(cudaMemcpy(set.d_IPs, l.d_IPs, l.Size * sizeof(unsigned int), cudaMemcpyDeviceToDevice), "Cannot copy IPs to device memory in + operator.");
	GpuAssert(cudaMemcpy(set.d_Lenghts, l.d_Lenghts, l.Size * sizeof(int), cudaMemcpyDeviceToDevice), "Cannot copy Lenghts to device memory in + operator.");

	GpuAssert(cudaMemcpy(set.d_IPs + l.Size, r.d_IPs, r.Size * sizeof(unsigned int), cudaMemcpyDeviceToDevice), "Cannot copy IPs to device memory in + operator.");
	GpuAssert(cudaMemcpy(set.d_Lenghts + l.Size, r.d_Lenghts, r.Size * sizeof(int), cudaMemcpyDeviceToDevice), "Cannot copy Lenghts to device memory in + operator.");

	return set;
}
