#pragma once

#include "cuda_runtime.h"
#include <stdexcept>
#include <ostream>
#include <vector>

using namespace std;

inline void GpuAssert(cudaError_t code, const char *msg)
{
	if (code != cudaSuccess)
	{
		string m(msg);
		m.append(": ");
		m.append(cudaGetErrorString(code));
		throw runtime_error(m.c_str());
	}
		
}

class GpuSetup
{
public:
	int Blocks;
	int Threads;
	int DeviceID;


	GpuSetup()
		: Blocks(-1),
		Threads(-1),
		DeviceID(-1) {}

	GpuSetup(int blocks, int threads, int deviceID)
		: Blocks(blocks),
		  Threads(threads),
		  DeviceID(deviceID) {}

	friend ostream& operator<<(ostream& os, const GpuSetup& obj)
	{
		return os
			<< "Blocks: " << obj.Blocks
			<< " Threads: " << obj.Threads
			<< " DeviceID: " << obj.DeviceID;
	}


	GpuSetup(const GpuSetup& other)
		: Blocks(other.Blocks),
		  Threads(other.Threads),
		  DeviceID(other.DeviceID)
	{
	}

	GpuSetup& operator=(const GpuSetup& other)
	{
		if (this == &other)
			return *this;
		Blocks = other.Blocks;
		Threads = other.Threads;
		DeviceID = other.DeviceID;
		return *this;
	}

	friend bool operator==(const GpuSetup& lhs, const GpuSetup& rhs)
	{
		return lhs.Blocks == rhs.Blocks
			&& lhs.Threads == rhs.Threads
			&& lhs.DeviceID == rhs.DeviceID;
	}

	friend bool operator!=(const GpuSetup& lhs, const GpuSetup& rhs)
	{
		return !(lhs == rhs);
	}
};

void SplitLine(const string& str, const string& delim, vector<string>& parts);
