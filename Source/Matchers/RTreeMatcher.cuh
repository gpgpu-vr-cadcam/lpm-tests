
#include "../Utils/Utils.h"
#include "../Core/IPSet.cuh"
#include "../Core/BaseResult.h"

class RTreeModel
{
	
};

class RTreeResult : public BaseResult
{
	void PrintResult() override;
	int CountMatched() override;
};

class RTreeMatcher
{
public:
	GpuSetup Setup;
	RTreeModel Model;
	int R;
	float ModelBuildTime;


	RTreeMatcher(int r)
		: ModelBuildTime(0)
	{
		if (r != 2 && r != 4 && r != 8 && r != 16)
			throw runtime_error("Invalid R for RTree");

		R = r;
	}

	void BuildModel(IPSet set);
	RTreeResult Match(IPSet set);
};
