#pragma once

#include "../Core/IPSet.cuh"

class BaseMatcher
{
public:

	GpuSetup Setup;

	virtual ~BaseMatcher() {}

	virtual void BuildModel(IPSet set) = 0;
	virtual void Match(IPSet set) = 0;
};