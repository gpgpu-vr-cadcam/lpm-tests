#pragma once

#include "cuda_runtime.h"
#include <stdexcept>
#include <ostream>
#include <vector>

#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <iostream>

using namespace std;

class LPMException : public exception
{
public:
	explicit LPMException()
		: exception() {}
};

inline void GpuAssert(cudaError_t code, const char *msg)
{
	if (code != cudaSuccess)
	{
		string m(msg);
		m.append(": ");
		m.append(cudaGetErrorString(code));
		cerr << m << "   Error code:" << code << endl;
		throw LPMException();
	}
}

class GpuSetup
{
public:
	int Blocks;
	int Threads;
	int DeviceID;

	GpuSetup()
		: Blocks(-1), Threads(-1), DeviceID(-1) {}

	GpuSetup(int blocks, int threads, int deviceID)
		: Blocks(blocks), Threads(threads), DeviceID(deviceID) {}

	friend ostream& operator<<(ostream& os, const GpuSetup& obj)
	{
		return os << obj.Blocks << ";" << obj.Threads << ";" << obj.DeviceID << ";";
	}


	GpuSetup(const GpuSetup& other)
		: Blocks(other.Blocks), Threads(other.Threads), DeviceID(other.DeviceID) {}

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

struct Rnd
{
	int Min, Max;

	__host__ __device__ Rnd(int min = 0, int max = 10000) : Min(min), Max(max) {};

	__host__ __device__ int operator()() const
	{
		thrust::default_random_engine rng;
		thrust::uniform_int_distribution<int> dist(Min, Max);

		return dist(rng);
	}
};

void SplitLine(const string& str, const string& delim, vector<string>& parts);

class Timer
{
private:
	cudaEvent_t StartEvent, StopEvent;

public:
	void Start()
	{
		GpuAssert(cudaEventCreate(&StartEvent), "Cannot create StartEvent");
		GpuAssert(cudaEventCreate(&StopEvent), "Cannot create StopEvent");
		GpuAssert(cudaEventRecord(StartEvent, 0), "Cannot record StartEvent");
	}

	float Stop()
	{
		GpuAssert(cudaEventRecord(StopEvent, 0), "Cannot record StopEvent");
		GpuAssert(cudaEventSynchronize(StopEvent), "Cannot synchronize StopEvent");

		float time;
		GpuAssert(cudaEventElapsedTime(&time, StartEvent, StopEvent), "Cannot get elapsed time");
		
		return time;
	}
};
