#include "Result.cuh"

void Result::PrintResult()
{
}

int Result::CountMatched()
{
	int result = 0;

	for (int i = 0; i < IpsToMatchCount; ++i)
		if (MatchedMaskIndex[i] != -1)
			++result;
	//else
	//	printf("%d\n", i);

	return result;
}