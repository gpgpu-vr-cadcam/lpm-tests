#include "IPSet.cuh"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

void IPSet::Load(GpuSetup &setup, string &path)
{
	//TODO: This may be faster, but less clean

	Setup = setup;
	ifstream file(path);
	string line;
	string delims = ";.";
	vector<string> parts;

	while (!file.eof())
	{
		file >> line;
		line = line.substr(4, line.size());
		SplitLine(line, delims, parts);
	}

	file.close();

	Size = parts.size() / 5;
	int *IPData = new int[Size * 5];

	for(int i = 0; i < Size * 5; ++i)
		IPData[i] = static_cast<unsigned char>(stoi(parts[i]));

	GpuAssert(cudaSetDevice(Setup.DeviceID), "Cannot set cuda device in IPSet Load.");
	GpuAssert(cudaMalloc(reinterpret_cast<void**>(&d_IPData), 5 * Size * sizeof(unsigned char)), "Cannot init ip masks device memory.");
	GpuAssert(cudaMemcpy(d_IPData, IPData, 5 * Size * sizeof(unsigned char), cudaMemcpyHostToDevice), "Cannot copy ip masks to device memory.");
	GpuAssert(cudaSetDevice(0), "Cannot set cuda device in IPSet Load.");

	delete[] IPData;
}
