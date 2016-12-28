#include "IPSet.cuh"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

#include "device_launch_parameters.h"
#include <iomanip>

IPSet& IPSet::operator=(const IPSet& other)
{
	if (this == &other)
		return *this;
	Size = other.Size;
	Setup = other.Setup;

	GpuAssert(cudaSetDevice(Setup.DeviceID), "Cannot set cuda device in IPSet = operator.");
	GpuAssert(cudaMalloc(reinterpret_cast<void**>(&d_IPData), 5 * Size * sizeof(unsigned char)), "Cannot init ip masks device memory in = operator.");
	GpuAssert(cudaMemcpy(d_IPData, other.d_IPData, 5 * Size * sizeof(unsigned char), cudaMemcpyDeviceToDevice), "Cannot copy ip masks to device memory in = operator.");
	GpuAssert(cudaSetDevice(0), "Cannot set cuda device in IPSet = operator.");

	return *this;
}

IPSet::IPSet(const IPSet& other)
{
	if (this == &other)
		return;
	Size = other.Size;
	Setup = other.Setup;

	GpuAssert(cudaSetDevice(Setup.DeviceID), "Cannot set cuda device in IPSet = operator.");
	GpuAssert(cudaMalloc(reinterpret_cast<void**>(&d_IPData), 5 * Size * sizeof(unsigned char)), "Cannot init ip masks device memory in = operator.");
	GpuAssert(cudaMemcpy(d_IPData, other.d_IPData, 5 * Size * sizeof(unsigned char), cudaMemcpyDeviceToDevice), "Cannot copy ip masks to device memory in = operator.");
	GpuAssert(cudaSetDevice(0), "Cannot set cuda device in IPSet = operator.");
}

void IPSet::Dispose()
{
	if (d_IPData != NULL)
	{
		GpuAssert(cudaFree(d_IPData), "Cannot free device memory in IPSet destructor.");
		d_IPData = NULL;
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


	GpuAssert(cudaSetDevice(Setup.DeviceID), "Cannot set cuda device in IPSet Load.");
	GpuAssert(cudaMalloc(reinterpret_cast<void**>(&d_IPData), 5 * Size * sizeof(unsigned char)), "Cannot init ip masks device memory.");
	GpuAssert(cudaMemcpy(d_IPData, IPData, 5 * Size * sizeof(unsigned char), cudaMemcpyHostToDevice), "Cannot copy ip masks to device memory.");
	GpuAssert(cudaSetDevice(0), "Cannot reset cuda device in IPSet Load.");

	delete[] IPData;
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

	GpuAssert(cudaSetDevice(Setup.DeviceID), "Cannot set cuda device in IPSet Generate.");
	GpuAssert(cudaMalloc(reinterpret_cast<void**>(&d_IPData), 5 * Size * sizeof(unsigned char)), "Cannot init ip masks device memory.");
	GpuAssert(cudaMemcpy(d_IPData, IPData, 5 * Size * sizeof(unsigned char), cudaMemcpyHostToDevice), "Cannot copy ip masks to device memory.");
	GpuAssert(cudaSetDevice(0), "Cannot reset cuda device in IPSet Generate.");

	delete[] IPData;
}

__global__ void CopyIPsToSubset(unsigned char *dstIPs, unsigned char *srcIPs, int *indexes, int subsetSize)
{
	int iteration = blockIdx.x * blockDim.x + threadIdx.x;

	while(iteration < subsetSize)
	{
		int sourceInd = indexes[iteration];

		dstIPs[5 * iteration] = srcIPs[5 * sourceInd];
		dstIPs[5 * iteration + 1] = srcIPs[5 * sourceInd + 1];
		dstIPs[5 * iteration + 2] = srcIPs[5 * sourceInd + 2];
		dstIPs[5 * iteration + 3] = srcIPs[5 * sourceInd + 3];
		dstIPs[5 * iteration + 4] = srcIPs[5 * sourceInd + 4];

		iteration += blockDim.x * gridDim.x;
	}
}

IPSet IPSet::RandomSubset(int subsetSize, GpuSetup &setup)
{
	IPSet subset = RandomSubset(subsetSize);
	subset.Setup = setup;

	if (subset.Setup.DeviceID == Setup.DeviceID)
		return subset;

	unsigned char *d_OldData = subset.d_IPData;
	GpuAssert(cudaSetDevice(subset.Setup.DeviceID), "Cannot set cuda device in IPSet RandomSubset.");
	GpuAssert(cudaMalloc(reinterpret_cast<void**>(&subset.d_IPData), 5 * subset.Size * sizeof(unsigned char)), "Cannot malloc memory on other device for subset.");
	GpuAssert(cudaMemcpy(subset.d_IPData, d_OldData, 5 * subset.Size * sizeof(unsigned char), cudaMemcpyDeviceToDevice), "Cannot copy data to other device.");
	GpuAssert(cudaSetDevice(Setup.DeviceID), "Cannot set cuda device in IPSet RandomSubset.");
	GpuAssert(cudaFree(d_OldData), "Cannot free old data of random subset.");
	GpuAssert(cudaSetDevice(0), "Cannot set cuda device in IPSet RandomSubset.");
	return subset;
}

IPSet IPSet::RandomSubset(int subsetSize)
{
	IPSet subset;
	subset.Setup = Setup;
	subset.Size = subsetSize;

	GpuAssert(cudaSetDevice(Setup.DeviceID), "Cannot set cuda device in IPSet RandomSubset.");

	int *d_Indexes;
	GpuAssert(cudaMalloc(reinterpret_cast<void**>(&d_Indexes), subset.Size * sizeof(int)), "Cannot init indexes device memory.");
	int *d_RandomValues;
	GpuAssert(cudaMalloc(reinterpret_cast<void**>(&d_RandomValues), subset.Size * sizeof(int)), "Cannot init random values device memory.");
	GpuAssert(cudaMalloc(reinterpret_cast<void**>(&subset.d_IPData), 5 * subset.Size * sizeof(unsigned char)), "Cannot init ip masks device memory.");

	thrust::sequence(thrust::device, d_Indexes, d_Indexes + Size, 0);
	if(subset.Size > Size)
		thrust::generate_n(thrust::device, d_Indexes + Size, subsetSize - Size, Rnd(0, Size));
	thrust::generate_n(thrust::device, d_RandomValues, subset.Size, Rnd(0, subset.Size));
	thrust::stable_sort_by_key(thrust::device, d_RandomValues, d_RandomValues + subset.Size, d_Indexes);
	thrust::sort(thrust::device, d_Indexes, d_Indexes + subset.Size);

	CopyIPsToSubset << < Setup.Blocks, Setup.Threads >> > (subset.d_IPData, d_IPData, d_Indexes, subset.Size);
	GpuAssert(cudaPeekAtLastError(), "Error while launching CopyIPsToSubset kernel");
	GpuAssert(cudaDeviceSynchronize(), "Error while running CopyIPsToSubset kernel");

	GpuAssert(cudaFree(d_Indexes), "Cannot free indexes memory.");
	GpuAssert(cudaFree(d_RandomValues), "Cannot free random values memory.");

	GpuAssert(cudaSetDevice(0), "Cannot set cuda device in IPSet RandomSubset.");

	return subset;
}

std::ostream& operator<<(std::ostream& os, const IPSet& obj)
{
	unsigned char *ips = new unsigned char[obj.Size * 5];

	GpuAssert(cudaSetDevice(obj.Setup.DeviceID), "Cannot set cuda device in IPSet << operator.");
	GpuAssert(cudaMemcpy(ips, obj.d_IPData, obj.Size * 5 * sizeof(unsigned char), cudaMemcpyDeviceToHost), "Cannot copy ip masks in << operator.");
	GpuAssert(cudaSetDevice(0), "Cannot set cuda device in IPSet << operator.");

	for(int i = 0; i < obj.Size; ++i)
		os << static_cast<int>(ips[5 * i + 4]) << setw(10) << static_cast<int>(ips[5 * i]) << "." << static_cast<int>(ips[5 * i + 1]) << "." 
			<< static_cast<int>(ips[5 * i + 2]) << "." << static_cast<int>(ips[5 * i + 3])  << endl;

	delete[] ips;
	return os;
}

IPSet operator+(IPSet& l, IPSet& r)
{
	IPSet set;

	if (l.Setup.DeviceID != r.Setup.DeviceID)
		throw runtime_error("Cannot add set from different devices");

	set.Size = l.Size + r.Size;
	set.Setup = l.Setup;

	GpuAssert(cudaSetDevice(set.Setup.DeviceID), "Cannot set cuda device in IPSet add.");
	GpuAssert(cudaMalloc(reinterpret_cast<void**>(&set.d_IPData), 5 * set.Size * sizeof(unsigned char)), "Cannot init ip masks device memory.");
	GpuAssert(cudaMemcpy(set.d_IPData, l.d_IPData, 5 * l.Size * sizeof(unsigned char), cudaMemcpyDeviceToDevice), "Cannot copy ip masks to device memory.");
	GpuAssert(cudaMemcpy(set.d_IPData + 5 * l.Size * sizeof(unsigned char), r.d_IPData, 5 * r.Size * sizeof(unsigned char), cudaMemcpyDeviceToDevice), "Cannot copy ip masks to device memory.");
	GpuAssert(cudaSetDevice(0), "Cannot reset cuda device in IPSet Generate.");

	return set;
}
