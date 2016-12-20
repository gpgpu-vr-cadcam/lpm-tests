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
	void Load(GpuSetup &setup, string &path, int count);
	IPSet RandomSubset(int subsetSize);
	friend std::ostream& operator<<(std::ostream& os, const IPSet& obj);
	
};
