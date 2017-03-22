#include "../Utils/Utils.h"
#include "../Core/IPSet.cuh"
#include "../Core/Result.cuh"

class ArrayMatcher
{
public:
	GpuSetup Setup;
	float ModelBuildTime;

	ArrayMatcher()
		: ModelBuildTime(0)
	{
	}

	void BuildModel(IPSet &set);
	Result Match(IPSet &set);
};
