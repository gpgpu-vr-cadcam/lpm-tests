#pragma once

#include "../Utils/Utils.h"
#include "../Core/IPSet.cuh"
#include "../Core/BaseResult.h"

class TreeModel
{
public:
	unsigned int * d_SubnetsIndx;
	int * d_PrefixesStart;
	int * d_PrefixesEnd;
	int ** d_Tree;
	char * d_SortedSubnetBits;
	int * d_Level8;
	int * d_Level16;
	int * d_Level24;
	int * d_LevelNodes;
	int MinPrefix;
	int Size;
	GpuSetup Setup;

	~TreeModel()
	{
		if(!Disposed)
			Dispose();
	}
	void Dispose();
	bool Disposed = true;
};

class TreeResult : public BaseResult
{
public:
	int * MatchedIndexes = NULL;
	char * SortedSubnetsBits = NULL;
	unsigned int * IPsList = NULL;
	int IPsListSize = 0;
	const int PrintJump = 100;
	float PresortingTime = 0;

	virtual ~TreeResult()
	{
		if (MatchedIndexes != NULL)
			delete[] MatchedIndexes;
		if (SortedSubnetsBits != NULL)
			delete[] SortedSubnetsBits;
		if (IPsList != NULL)
			delete[] IPsList;
	}

	void PrintResult() override;
	void ConvertFromBits(char * inBits, unsigned char * outByte);
	int CountMatched() override;
};

class TreeMatcher
{
public:
	GpuSetup Setup;
	bool UseMidLevels = false;
	bool UsePresorting = false;

	float ModelBuildTime;
	const int MaxSetSize;

	TreeMatcher(int maxSetSize)
		: ModelBuildTime(0),
	      MaxSetSize(maxSetSize) {}

	TreeModel Tree;
	void BuildModel(IPSet set);
	TreeResult Match(IPSet set);
};
