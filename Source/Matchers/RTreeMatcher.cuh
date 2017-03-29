
#include "../Utils/Utils.h"
#include "../Core/IPSet.cuh"
#include "../Core/Result.cuh"

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
	//MaxIP stored in tree structure
	int** Masks;
	int* Lenghts;

	//Number of nodes on each tree level
	int* LevelsSizes;

	//Number of possible children of tree node
	int *ChildrenCount;

	//Children arrays for tree nodes
	int** Children;
	int** h_Children;

	//Lists items (indexes to MaxIP array)
	int* ListItems;

	//Start indexes of lists (in ListsItems array) of tree nodes
	int** ListsStarts;
	int** h_ListsStarts;

	//Lenghts of lists of tree nodes
	int** ListsLenghts;
	int** h_ListsLenghts;

	int *totalListItemsPerLevel;

	//Building model
	void Build(IPSet &set, GpuSetup setup);

	int GetMinListLenght(int level);
	int GetMaxListLenght(int level);

	void Dispose();
	~RTreeModel()
	{
		Dispose();
	}
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

	void BuildModel(IPSet &set);
	Result Match(IPSet &set);
};
