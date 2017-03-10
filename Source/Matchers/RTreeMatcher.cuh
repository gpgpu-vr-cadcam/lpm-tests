
#include "../Utils/Utils.h"
#include "../Core/IPSet.cuh"
#include "../Core/BaseResult.h"

class RTreeModel
{
public:
	//Number of masks stored in tree structure
	int Count;

	//Bits per level
	vector<int> h_R;
	int *R;
	int *rSums;
	int *rPreSums;

	//Number of levels
	int L;

	//TODO: Rename pointers to d_
	//Masks stored in tree structure
	int** Masks;
	int* Lenghts;

	//Number of nodes on each tree level
	int* LevelsSizes;

	//Number of possible children of tree node
	int *ChildrenCount;

	//Children arrays for tree nodes
	int** Children;
	int** h_Children;

	//Lists items (indexes to Masks array)
	int* ListItems;

	//Start indexes of lists (in ListsItems array) of tree nodes
	int** ListsStarts;
	int** h_ListsStarts;

	//Lenghts of lists of tree nodes
	int** ListsLenghts;
	int** h_ListsLenghts;

	//Building model
	void Build(IPSet set, GpuSetup setup);

	void Dispose();
	~RTreeModel()
	{
		Dispose();
	}
};

class RTreeResult : public BaseResult
{
public:
	int *MatchedMaskIndex;
	int IpsToMatchCount;


	RTreeResult(int ips_to_match_count)
		: IpsToMatchCount(ips_to_match_count),
		  MatchedMaskIndex(NULL)
	{
	}

	~RTreeResult()
	{
		if(MatchedMaskIndex != NULL)
		{
			delete[] MatchedMaskIndex;
			MatchedMaskIndex = NULL;
		}
	}

	void PrintResult() override;
	int CountMatched() override;
};

class RTreeMatcher
{
public:
	GpuSetup Setup;
	RTreeModel Model;

	float ModelBuildTime;


	RTreeMatcher(vector<int> r)
		: ModelBuildTime(0)
	{
		Model.h_R = r;
	}

	void BuildModel(IPSet set);
	RTreeResult Match(IPSet set);
};
