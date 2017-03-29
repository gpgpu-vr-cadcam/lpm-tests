#include "../Utils/Utils.h"
#include "../Core/IPSet.cuh"
#include "../Core/Result.cuh"

class ArrayMatcher
{
public:
	GpuSetup Setup;
	float ModelBuildTime;

	int *MaxIP;
	int *MinIP;
	int *Lenghts;
	int *Array;

	ArrayMatcher()
		: ModelBuildTime(0)
	{
		MaxIP = NULL;
		MinIP = NULL;
		Array = NULL;
		Lenghts = NULL;
	}

	~ArrayMatcher()
	{
		if(MaxIP != NULL)
		{
			GpuAssert(cudaFree(MaxIP), "Cannot free MaxIP memory");
			MaxIP = NULL;
		}

		if (MinIP != NULL)
		{
			GpuAssert(cudaFree(MaxIP), "Cannot free MinIP memory");
			MinIP = NULL;
		}

		if(Lenghts != NULL)
		{
			GpuAssert(cudaFree(Lenghts), "Cannot free Lenghts memory");
			Lenghts = NULL;
		}

		if (Array != NULL)
		{
			GpuAssert(cudaFree(Array), "Cannot free Array memory");
			Array = NULL;
		}
	}

	void BuildModel(IPSet &set);
	Result Match(IPSet &set);
};
