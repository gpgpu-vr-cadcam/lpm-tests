#pragma once
#include "../Utils/Utils.h"
#include <vector>
#include <string>
#include <ostream>
#include <fstream>
using namespace std;

class TestFile
{
public:
	int Size;
	string FileName;

	TestFile(const string& file_name, int size)
		: Size(size),
		FileName(file_name) {}

	friend std::ostream& operator<<(std::ostream& os, const TestFile& obj)
	{
		return os
			<< "Size: " << obj.Size
			<< " FileName: " << obj.FileName;
	}
};

class IPSetTest
{
public:
	GpuSetup Setup;
	int MasksToLoad;


	IPSetTest(const GpuSetup& setup, int masksToLoad)
		: Setup(setup),
		  MasksToLoad(masksToLoad) {}


	friend std::ostream& operator<<(std::ostream& os, const IPSetTest& obj)
	{
		return os
			<< "Setup: " << obj.Setup
			<< " MasksToLoad: " << obj.MasksToLoad;
	}
};

class IPSetLoadTest : public IPSetTest
{
public:
	TestFile File;

	IPSetLoadTest(const TestFile& file, const GpuSetup& setup, int masksToLoad)
		: IPSetTest(setup, masksToLoad),
		  File(file) {}

	friend std::ostream& operator<<(std::ostream& os, const IPSetLoadTest& obj)
	{
		return os
			<< "File: " << obj.File
			<< " Setup: " << obj.Setup
			<< " MasksToLoad: " << obj.MasksToLoad;
	}
};

class IPSubsetTest : public IPSetLoadTest
{
public:
	int SubsetSize;

	IPSubsetTest(const TestFile& file, const GpuSetup& setup, int masksToLoad, int subset_size)
		: IPSetLoadTest(file, setup, masksToLoad),
		  SubsetSize(subset_size) {}

	friend std::ostream& operator<<(std::ostream& os, const IPSubsetTest& obj)
	{
		return os
			<< static_cast<const IPSetLoadTest&>(obj)
			<< " SubsetSize: " << obj.SubsetSize;
	}
};

class RTreeMatcherTest : public IPSubsetTest
{
public:
	vector<int> R;

	RTreeMatcherTest(const TestFile& file, const GpuSetup& setup, int masksToLoad, int subset_size, vector<int> r)
		: IPSubsetTest(file, setup, masksToLoad, subset_size),
		  R(r) {}

	friend std::ostream& operator<<(std::ostream& os, const RTreeMatcherTest& obj)
	{
		os << static_cast<const IPSubsetTest&>(obj) << " {";
		for (int i = 0; i < obj.R.size(); ++i)
			os << obj.R[i] << ",";
		os << " } ";

		return os;
	}
};

class PerformanceTest
{
public:
	int Seed;
	TestFile SourceSet;
	int ModelSubsetSize;
	int MatchSubsetSize;
	int RandomMasksSetSize;
	int Blocks;
	int Threads;
	int DeviceID;

	PerformanceTest(int seed, const TestFile& source_set, int model_subset_size, int match_subset_size, int random_masks_set_size, int blocks, int threads, int device_id)
		: Seed(seed),
		  SourceSet(source_set),
		  ModelSubsetSize(model_subset_size),
		  MatchSubsetSize(match_subset_size),
		  RandomMasksSetSize(random_masks_set_size),
		  Blocks(blocks),
		  Threads(threads),
		  DeviceID(device_id) {}


	friend std::ostream& operator<<(std::ostream& os, const PerformanceTest& obj)
	{
		return os
			<< "Seed: " << obj.Seed
			<< " SourceSet: " << obj.SourceSet
			<< " ModelSubsetSize: " << obj.ModelSubsetSize
			<< " MatchSubsetSize: " << obj.MatchSubsetSize
			<< " RandomMasksSetSize: " << obj.RandomMasksSetSize
			<< " Blocks: " << obj.Blocks
			<< " Threads: " << obj.Threads
			<< " DeviceID: " << obj.DeviceID;
	}
};

class TreeMatcherPerformanceTestCase : public PerformanceTest
{
public:
	bool UsePresorting;
	bool UseMidLevels;


public:
	TreeMatcherPerformanceTestCase(int seed, const TestFile& source_set, int model_subset_size, int match_subset_size, int random_masks_set_size, int blocks, int threads, int device_id, bool use_presorting, bool use_mid_levels)
		: PerformanceTest(seed, source_set, model_subset_size, match_subset_size, random_masks_set_size, blocks, threads, device_id),
		  UsePresorting(use_presorting),
		  UseMidLevels(use_mid_levels)
	{
	}


	friend std::ostream& operator<<(std::ostream& os, const TreeMatcherPerformanceTestCase& obj)
	{
		return os
			<< static_cast<const PerformanceTest&>(obj)
			<< " UsePresorting: " << obj.UsePresorting
			<< " UseMidLevels: " << obj.UseMidLevels;
	}
};

class RTreeMatcherPerformanceTestCase : public PerformanceTest
{
public:
	vector<int> R;


	RTreeMatcherPerformanceTestCase(int seed, const TestFile& source_set, int model_subset_size, int match_subset_size, int random_masks_set_size, int blocks, int threads, int device_id, const vector<int>& is)
		: PerformanceTest(seed, source_set, model_subset_size, match_subset_size, random_masks_set_size, blocks, threads, device_id),
		  R(is)
	{
	}


	friend std::ostream& operator<<(std::ostream& os, const RTreeMatcherPerformanceTestCase& obj)
	{
		os << static_cast<const PerformanceTest&>(obj) << " {";
		for (int i = 0; i < obj.R.size(); ++i)
			os << obj.R[i] << ",";
		os << " } ";

		return os;
	}
};

class Environment
{
public:

	string TestDataPath = "../../../TestData/";
	vector<TestFile> Files;
	vector<GpuSetup> Setups;
	vector<IPSetTest> IPSetTests;
	vector<IPSetLoadTest> IPSetLoadTests;
	vector<IPSubsetTest> SubsetTests;
	vector<RTreeMatcherTest> RTreeMatcherTests;

	vector<PerformanceTest> PerformanceTests;
	vector<TreeMatcherPerformanceTestCase> TreeMatcherPerformanceTests;
	vector<RTreeMatcherPerformanceTestCase> RTreeMatcherPerformanceTests;

	ofstream ResultsFile;
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
	}

	void InitGenerateSetups()
	{
		vector<int> masksToLoad = { 10000, 20000 };

		for (auto s : Setups)
			for (auto m : masksToLoad)
				IPSetTests.push_back(IPSetTest(s, m));

	}

	void InitSetups()
	{
		vector<int> devices = { 0 };
		vector<int> blocks = { 1000 };
		vector<int> threads = { 512, 1024 };

		for (auto d : devices)
			for (auto b : blocks)
				for (auto t : threads)
					Setups.push_back(GpuSetup(b, t, d));
	}

	void InitIPSetTests()
	{
		for (auto f : Files)
			for(auto t : IPSetTests)
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
				{ 5, 5, 6, 8, 8},
				{ 4, 5, 4, 5, 6, 4, 2 },
			};

		for (auto t : SubsetTests)
			for (auto r : rs)
				RTreeMatcherTests.push_back(RTreeMatcherTest(t.File, t.Setup, t.MasksToLoad, t.SubsetSize, r));
	}

	void InitPerformanceTests()
	{
		vector<int> Seeds = { 2341};
		vector<int> ModelSetSize = { 100000 };
		vector<int> MatchSet1Size = { 1000000 };
		vector<int> MatchSet2Size = { 1000000 };
		vector<int> Blocks = { 1024 };
		vector<int> Threads = { 1024 };
		vector<int> Devices = { 0 };
		

		for (auto s : Seeds)
			for (auto b : Blocks)
				for (auto t : Threads)
					for (auto d : Devices)
						for (auto modelSS : ModelSetSize)
							for (auto matchSS : MatchSet1Size)
								for (auto rndSS : MatchSet2Size)
									for (auto f : Files)
										PerformanceTests.push_back(PerformanceTest(s, f, modelSS, matchSS, rndSS, b, t, d));

		ResultsFile.open("TestResults.txt");

#ifndef NO_THREADS_TRACE
		ThreadsFile.open("ThreadsTimes.txt");
		ThreadsFileLines = 0;
#endif
	}

	void InitTreeMatcherPerformanceTests()
	{
		vector<bool> UsePresorting = { false, true };
		vector<bool> UseMidLevels = { false, true };

		for (auto t : PerformanceTests)
			for (auto ps : UsePresorting)
				for (auto ml : UseMidLevels)
					TreeMatcherPerformanceTests.push_back(TreeMatcherPerformanceTestCase(t.Seed, t.SourceSet, t.ModelSubsetSize, t.MatchSubsetSize, t.RandomMasksSetSize, t.Blocks, t.Threads, t.DeviceID, ps, ml));
	}

	void InitRTreeMatcherPerformanceTests()
	{
		//vector<vector<int>> rs =
		//{
		//	{ 8, 8, 8, 8 },
		//	{ 5, 5, 6, 8, 8 },
		//	{ 8, 8, 4, 4, 8 },
		//};
		vector<vector<int>> rs =
		{
			{ 4, 4, 4, 4, 4, 4, 4, 4 },
			{ 5, 5, 6, 8, 8 },
		};

		for (auto t : PerformanceTests)
			for (auto r : rs)
					RTreeMatcherPerformanceTests.push_back(RTreeMatcherPerformanceTestCase(t.Seed, t.SourceSet, t.ModelSubsetSize, t.MatchSubsetSize, t.RandomMasksSetSize, t.Blocks, t.Threads, t.DeviceID, r));
	}

	Environment()
	{
		InitFiles();
		InitSetups();
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
