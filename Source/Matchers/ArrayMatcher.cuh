#include "../Utils/Utils.h"
#include "../Core/IPSet.cuh"
#include "../Core/Result.cuh"

class ArrayMatcher
{
public:
	uchar3 EMPTY;

	GpuSetup Setup;
	float ModelBuildTime;

	uchar3 *Array;

	int MaxLenght;
	int MinLenght;
	int ArraySize;

	ArrayMatcher()
		: ModelBuildTime(0)
	{
		Array = NULL;
		EMPTY.x = EMPTY.y = EMPTY.z = ~0;
	}

	~ArrayMatcher()
	{
		if (Array != NULL)
		{
			GpuAssert(cudaFree(Array), "Cannot free Array memory");
			Array = NULL;
		}
	}

	void BuildModel(IPSet &set);
	Result Match(IPSet &set);
};
