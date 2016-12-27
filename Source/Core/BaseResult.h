#pragma once
#include <cstddef>

class BaseResult
{
public:
	unsigned int * ThreadTimeStart = NULL;
	unsigned int * ThreadTimeEnd = NULL;
	bool ThreadTimeRecorded = false;

	float MatchingTime;

	virtual ~BaseResult()
	{
		if (ThreadTimeStart != NULL)
			delete[] ThreadTimeStart;
		if (ThreadTimeEnd != NULL)
			delete[] ThreadTimeEnd;
		ThreadTimeStart = ThreadTimeEnd = NULL;
	}

	virtual void PrintResult() = 0;
	virtual int CountMatched() = 0;
};
