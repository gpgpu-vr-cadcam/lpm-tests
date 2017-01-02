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
	bool UsePresorting;
	bool UseMidLevels;


	PerformanceTest(int seed, const TestFile& source_set, int model_subset_size, int match_subset_size, int random_masks_set_size, int blocks, int threads, int device_id, bool use_presorting, bool use_mid_levels)
		: Seed(seed),
		  SourceSet(source_set),
		  ModelSubsetSize(model_subset_size),
		  MatchSubsetSize(match_subset_size),
		  RandomMasksSetSize(random_masks_set_size),
		  Blocks(blocks),
		  Threads(threads),
		  DeviceID(device_id),
		  UsePresorting(use_presorting),
		  UseMidLevels(use_mid_levels) {}


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
			<< " DeviceID: " << obj.DeviceID
			<< " UsePresorting: " << obj.UsePresorting
			<< " UseMidLevels: " << obj.UseMidLevels;
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

	vector<PerformanceTest> PerformanceTests;
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

	void InitPerformanceTests()
	{
		vector<int> Seeds = { 2341};
		vector<int> ModelSetSize = { 400000 };
		vector<int> MatchSet1Size = { 150000 };
		vector<int> MatchSet2Size = { 150000 };
		vector<int> Blocks = { 1024 };
		vector<int> Threads = { 1024 };
		vector<int> Devices = { 0 };
		vector<bool> UsePresorting = { false, true };
		vector<bool> UseMidLevels = { false, true };

		for (auto s : Seeds)
			for (auto b : Blocks)
				for (auto t : Threads)
					for (auto d : Devices)
						for (auto modelSS : ModelSetSize)
							for (auto matchSS : MatchSet1Size)
								for (auto rndSS : MatchSet2Size)
									for(auto ps : UsePresorting)
										for(auto ml : UseMidLevels)
											for (auto f : Files)
												PerformanceTests.push_back(PerformanceTest(s, f, modelSS, matchSS, rndSS, b, t, d, ps, ml));

		ResultsFile.open("TestResults.txt");
		ThreadsFile.open("ThreadsTimes.txt");
		ThreadsFileLines = 0;
	}

	Environment()
	{
		InitFiles();
		InitSetups();
		InitGenerateSetups();
		InitIPSetTests();
		InitSubsetTests();

		#ifdef PERF_TEST
			InitPerformanceTests();
		#endif
	}
};

extern Environment ENV;
