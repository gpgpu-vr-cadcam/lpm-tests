#pragma once
#include "PerformanceTests.h"
#include <fstream>

class Environment
{
public:

	string TestDataPath = "../../../TestData/";
	vector<TestFile> Files;
	vector<TestFile> ShortMasksFiles;

	vector<GpuSetup> PerfSetups;
	vector<GpuSetup> Setups;
	vector<IPSetTest> IPSetTests;
	vector<IPSetLoadTest> IPSetLoadTests;
	vector<IPSubsetTest> SubsetTests;
	vector<RTreeMatcherTest> RTreeMatcherTests;

	vector<PerformanceTest> PerformanceTests;
	vector<TreeMatcherPerformanceTestCase> TreeMatcherPerformanceTests;
	vector<RTreeMatcherPerformanceTestCase> RTreeMatcherPerformanceTests;

	ofstream ArrayResultsFile;
	ofstream TreeResultsFile;
	ofstream RTreeResultsFile;

	ofstream ThreadsFile;
	int ThreadsFileLines;

	void InitFiles()
	{
		Files.push_back(TestFile("data-raw-table_australia_012016.txt", 565949));
		Files.push_back(TestFile("data-raw-table_australia_092016.txt", 420206));
		Files.push_back(TestFile("data-raw-table_honkkong_012016.txt", 565907));
		Files.push_back(TestFile("data-raw-table_honkkong_092016.txt", 602812));
		Files.push_back(TestFile("data-raw-table_london_012016.txt", 564426));
		Files.push_back(TestFile("data-raw-table_london_092016.txt", 601489));
		Files.push_back(TestFile("data-raw-table_tokyo_012016.txt", 576846));
		Files.push_back(TestFile("data-raw-table_tokyo_092016.txt", 611460));
		Files.push_back(TestFile("data-raw-table_usa_012016.txt", 561755));
		Files.push_back(TestFile("data-raw-table_usa_092016.txt", 598473));

		ShortMasksFiles.push_back(TestFile("data-raw-table_australia_012016_short_masks.txt", 565726));
		ShortMasksFiles.push_back(TestFile("data-raw-table_australia_092016_short_masks.txt", 420098));
		ShortMasksFiles.push_back(TestFile("data-raw-table_honkkong_012016_short_masks.txt", 565470));
		ShortMasksFiles.push_back(TestFile("data-raw-table_honkkong_092016_short_masks.txt", 602206));
		ShortMasksFiles.push_back(TestFile("data-raw-table_london_012016_short_masks.txt", 563495));
		ShortMasksFiles.push_back(TestFile("data-raw-table_london_092016_short_masks.txt", 600444));
		ShortMasksFiles.push_back(TestFile("data-raw-table_tokyo_012016_short_masks.txt", 575282));
		ShortMasksFiles.push_back(TestFile("data-raw-table_tokyo_092016_short_masks.txt", 610033));
		ShortMasksFiles.push_back(TestFile("data-raw-table_usa_012016_short_masks.txt", 561519));
		ShortMasksFiles.push_back(TestFile("data-raw-table_usa_092016_short_masks.txt", 598235));
	}

	void InitGenerateSetups()
	{
		vector<int> masksToLoad = { 10000, 20000, 30000, 40000, 50000 };

		for (auto s : Setups)
			for (auto m : masksToLoad)
				IPSetTests.push_back(IPSetTest(s, m));

	}

	void InitSetups()
	{
		vector<int> devices = { 0 };
		vector<int> blocks = { 512 };
		vector<int> threads = { 512 };

		for (auto d : devices)
			for (auto b : blocks)
				for (auto t : threads)
					Setups.push_back(GpuSetup(b, t, d));
	}

	void InitPerfSetups()
	{
		vector<int> devices = { 0 };
		vector<int> blocks = { 128, 256, 512, 1024 };
		vector<int> threads = { 128, 256, 512, 1024 };

		for (auto d : devices)
			for (auto b : blocks)
				for (auto t : threads)
					PerfSetups.push_back(GpuSetup(b, t, d));
	}

	void InitIPSetTests()
	{
		for (auto f : ShortMasksFiles)
			for (auto t : IPSetTests)
				IPSetLoadTests.push_back(IPSetLoadTest(f, t.Setup, t.MasksToLoad));
	}

	void InitSubsetTests()
	{
		vector<int> subsetSizes = { 5000, 12500 };

		for (auto t : IPSetLoadTests)
			for (auto s : subsetSizes)
				SubsetTests.push_back(IPSubsetTest(t.File, t.Setup, t.MasksToLoad, s));
	}

	void InitRTreeMatcherTests()
	{
		vector<vector<int>> rs =
		{
			{ 8, 8, 8, 8 },
			{ 8, 8, 16 },
			{ 5, 5, 6, 8, 8 },
			{ 4, 5, 4, 5, 6, 4, 2 },
		};

		for (auto t : SubsetTests)
			for (auto r : rs)
				RTreeMatcherTests.push_back(RTreeMatcherTest(t.File, t.Setup, t.MasksToLoad, t.SubsetSize, r));
	}

	void InitPerformanceTests()
	{
		vector<int> Seeds = { 5236, 1090, 9876, 4254, 7648, 5983, 5867, 3329, 7249 };
		vector<int> MatchSetSize = { 250000, 500000, 750000, 1000000, 2500000, 5000000, 7500000, 10000000 };
		vector<bool> UsePresorting = { false, true };
		vector<float> RandomMasks = { 0, 0.25, 0.5, 0.75, 1 };

		for (auto s : Seeds)
			for (auto testFile : ShortMasksFiles)
				for (auto setup : PerfSetups)
					PerformanceTests.push_back(PerformanceTest(s, testFile, MatchSetSize, RandomMasks, UsePresorting, setup));

		ArrayResultsFile.open("ArrayTestResults.txt");

#ifndef NO_THREADS_TRACE
		ThreadsFile.open("ThreadsTimes.txt");
		ThreadsFileLines = 0;
#endif
	}

	void InitTreeMatcherPerformanceTests()
	{
		vector<bool> UseMidLevels = { false, true };

		for (auto perfTest : PerformanceTests)
			for (auto useMidLevels : UseMidLevels)
				TreeMatcherPerformanceTests.push_back(TreeMatcherPerformanceTestCase(perfTest, useMidLevels));

		TreeResultsFile.open("TreeTestResults.txt");
	}

	void InitRTreeMatcherPerformanceTests()
	{
		vector<vector<int>> rs =
		{
			{ 4, 4, 4, 4, 4, 4, 8 },
			{ 4, 4, 4, 4, 8, 8 },
			{ 8, 4, 4, 4, 4, 8 },
			{ 8, 2, 2, 2, 2, 2, 2, 2, 2, 8 },
			{ 8, 4, 4, 2, 2, 2, 2, 8 },
			{ 8, 4, 4, 4, 2, 2, 8 },
			{ 8, 8, 8, 8 },
			{ 8, 8, 4, 4, 8 },
			{ 8, 8, 2, 2, 2, 2, 8 },
			{ 8, 8, 4, 2, 2, 8 }
		};

		for (auto perfTest : PerformanceTests)
			for (auto r : rs)
				RTreeMatcherPerformanceTests.push_back(RTreeMatcherPerformanceTestCase(perfTest, r));

		RTreeResultsFile.open("RTreeTestResults.txt");
	}


	Environment()
	{
		InitFiles();
		InitSetups();
		InitPerfSetups();
		InitGenerateSetups();
		InitIPSetTests();
		InitSubsetTests();
		InitRTreeMatcherTests();

#ifdef PERF_TEST
		InitPerformanceTests();
		InitTreeMatcherPerformanceTests();
		InitRTreeMatcherPerformanceTests();
#endif
	}
};

extern Environment ENV;