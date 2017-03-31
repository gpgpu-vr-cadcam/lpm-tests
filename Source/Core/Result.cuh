#pragma once
#include "BaseResult.h"

class Result : public BaseResult
{
public:
	int *MatchedMaskIndex;
	int IpsToMatchCount;


	Result(int ips_to_match_count)
		: IpsToMatchCount(ips_to_match_count),
		MatchedMaskIndex(NULL)
	{
	}

	~Result()
	{
		if (MatchedMaskIndex != NULL)
		{
			delete[] MatchedMaskIndex;
			MatchedMaskIndex = NULL;
		}
	}

	void PrintResult() override;
	int CountMatched() override;
};
