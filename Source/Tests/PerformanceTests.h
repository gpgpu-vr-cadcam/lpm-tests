#pragma once
#include "Tests.h"
#include <ostream>

class PerformanceTest
{
public:

	int Seed;
	TestFile SourceSet;
	vector<int> MatchSubsetSize;
	vector<float> RandomMasksSetSize;
	vector<bool> PresortMatchSet;
	GpuSetup Setup;


	PerformanceTest(int seed, const TestFile& source_set, const vector<int>& match_subset_size, const vector<float>& random_masks_set_size, const vector<bool>& presort_match_set, const GpuSetup& setup)
		: Seed(seed),
		  SourceSet(source_set),
		  MatchSubsetSize(match_subset_size),
		  RandomMasksSetSize(random_masks_set_size),
		  PresortMatchSet(presort_match_set),
		  Setup(setup) {}

	friend std::ostream& operator<<(std::ostream& os, const PerformanceTest& obj)
	{
		return os << obj.Seed << ";" << obj.SourceSet << obj.Setup;
	}

	int MaxMatchSetSize()
	{
		int max = MatchSubsetSize[0];
		for (int i = 1; i < MatchSubsetSize.size(); ++i)
			if (max < MatchSubsetSize[i])
				max = MatchSubsetSize[i];

		return max;
	}
};

class TreeMatcherPerformanceTestCase : public PerformanceTest
{
public:
	bool UseMidLevels;

	TreeMatcherPerformanceTestCase(PerformanceTest &performanceTest, bool use_mid_levels)
		: PerformanceTest(performanceTest.Seed, performanceTest.SourceSet, performanceTest.MatchSubsetSize, 
			performanceTest.RandomMasksSetSize, performanceTest.PresortMatchSet, performanceTest.Setup),
		UseMidLevels(use_mid_levels) {}

	friend std::ostream& operator<<(std::ostream& os, const TreeMatcherPerformanceTestCase& obj)
	{
		return os << static_cast<const PerformanceTest&>(obj);
	}
};

class RTreeMatcherPerformanceTestCase : public PerformanceTest
{
public:
	vector<int> R;

	RTreeMatcherPerformanceTestCase(PerformanceTest &performanceTest, const vector<int>& r)
		: PerformanceTest(performanceTest.Seed, performanceTest.SourceSet, performanceTest.MatchSubsetSize,
			performanceTest.RandomMasksSetSize, performanceTest.PresortMatchSet, performanceTest.Setup),
		R(r) {}


	friend std::ostream& operator<<(std::ostream& os, const RTreeMatcherPerformanceTestCase& obj)
	{
		return os << static_cast<const PerformanceTest&>(obj);
	}
};