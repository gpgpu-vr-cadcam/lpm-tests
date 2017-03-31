#include "../Utils/Utils.h"
#include "../Core/IPSet.cuh"
#include "../Core/Result.cuh"

class ArrayMatcher
{
public:
	GpuSetup Setup;
	float ModelBuildTime;

	int *Array;	//TODO: Zamiana Array na uchar3 w celu oszczêdnoœci pamiêci

	int MaxLenght;
	int MinLenght;
	int ArraySize;

	ArrayMatcher()
		: ModelBuildTime(0)
	{
		Array = NULL;
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
