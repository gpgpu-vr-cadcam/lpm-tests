#pragma once

#include "../Utils/Utils.h"

class IPSet
{
public:

	int Size;
	char *d_IPData;
	GpuSetup Setup;

	IPSet() :
		Size(0),
		d_IPData(NULL) {}

	~IPSet()
	{
		if (d_IPData != NULL)
			GpuAssert(cudaFree(d_IPData), "Cannot free device memory in IPSet destructor.");
	}

	void Load(GpuSetup &setup, string &path);
};