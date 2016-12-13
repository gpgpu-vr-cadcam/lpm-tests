#pragma once

#include "cuda_runtime.h"
#include <iterator>
#include <stdexcept>
#include <ostream>

using namespace std;

inline void GpuAssert(cudaError_t code, const char *msg)
{
	if (code != cudaSuccess)
	{
		string m(msg);
		m.append(": ");
		m.append(cudaGetErrorString(code));
		throw std::runtime_error(m.c_str());
	}
		
}

class GpuSetup
{
public:
	int Blocks;
	int Threads;
	int DeviceID;


	GpuSetup(int blocks, int threads, int deviceID)
		: Blocks(blocks),
		  Threads(threads),
		  DeviceID(deviceID) {}


	friend std::ostream& operator<<(std::ostream& os, const GpuSetup& obj)
	{
		return os
			<< "Blocks: " << obj.Blocks
			<< " Threads: " << obj.Threads
			<< " DeviceID: " << obj.DeviceID;
	}
};
