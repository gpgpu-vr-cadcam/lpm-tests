#pragma once
#include <cstddef>

class BaseResult
{
public:
	unsigned int * ThreadTimeStart = NULL;
	unsigned int * ThreadTimeEnd = NULL;
	unsigned int * ThreadTime = NULL;
	bool ThreadTimeRecorded = false;

	float MatchingTime;

	virtual ~BaseResult()
	{
		if (ThreadTime != NULL)
			delete[] ThreadTime;
		if (ThreadTimeStart != NULL)
			delete[] ThreadTimeStart;
		if (ThreadTimeEnd != NULL)
			delete[] ThreadTimeEnd;
		ThreadTime = ThreadTimeStart = ThreadTimeEnd = NULL;
	}

	virtual void PrintResult() = 0;
	virtual int CountMatched() = 0;
};
