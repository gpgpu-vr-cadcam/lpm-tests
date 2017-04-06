#pragma once

#include "../Utils/Utils.h"
#include <ostream>

class IPSet
{
public:

	int Size;
	GpuSetup Setup;

	unsigned int *d_IPs;
	int *d_Lenghts;

	IPSet() :
		Size(0), d_IPs(NULL), d_Lenghts(NULL) {}

	~IPSet()
	{
		Dispose();
	}

	IPSet& operator=(const IPSet& other);

	IPSet(const IPSet& other);

	void Dispose();
	void Load(GpuSetup &setup, string path, int count);
	void Generate(GpuSetup &setup, int count);
	void RandomSubset(int subsetSize, IPSet &sourceSet);
	void Sort();
	void Randomize();

	friend IPSet operator+(IPSet &l, IPSet &r);
};
