#pragma once

#include "../Utils/Utils.h"
#include <ostream>

class IPSet
{
public:

	int Size;
	unsigned char *d_IPData;
	GpuSetup Setup;

	IPSet() :
		Size(0),
		d_IPData(NULL) {}

	~IPSet()
	{
		Dispose();
	}

	IPSet& operator=(const IPSet& other);

	IPSet(const IPSet& other);

	void Dispose();
	void Load(GpuSetup &setup, string path, int count);
	void Generate(GpuSetup &setup, int count);
	IPSet RandomSubset(int subsetSize);
	IPSet RandomSubset(int subsetSize, GpuSetup &setup);
	friend std::ostream& operator<<(std::ostream& os, const IPSet& obj);
	friend IPSet operator+(IPSet &l, IPSet &r);
	
};
