#pragma once
#include "../Utils/Utils.h"
#include <vector>
#include <string>
#include <ostream>

using namespace std;

class TestFile
{
public:
	string Path;
	int Size;

	TestFile(const string& path, int size)
		: Path(path),
		  Size(size) {}

	friend std::ostream& operator<<(std::ostream& os, const TestFile& obj)
	{
		return os
			<< "Path: " << obj.Path
			<< " Size: " << obj.Size;
	}
};

class IPSetTest
{
public:
	TestFile File;
	GpuSetup Setup;
	int MasksToLoad;


	IPSetTest(const TestFile& file, const GpuSetup& setup, int masksToLoad)
		: File(file),
		  Setup(setup),
		  MasksToLoad(masksToLoad) {}


	friend std::ostream& operator<<(std::ostream& os, const IPSetTest& obj)
	{
		return os
			<< "File: " << obj.File
			<< " Setup: " << obj.Setup
			<< " MasksToLoad: " << obj.MasksToLoad;
	}
};

class IPSubsetTest : public IPSetTest
{
public:
	int SubsetSize;


	IPSubsetTest(const TestFile& file, const GpuSetup& setup, int masksToLoad, int subset_size)
		: IPSetTest(file, setup, masksToLoad),
		  SubsetSize(subset_size) {}

	friend std::ostream& operator<<(std::ostream& os, const IPSubsetTest& obj)
	{
		return os
			<< static_cast<const IPSetTest&>(obj)
			<< " SubsetSize: " << obj.SubsetSize;
	}
};

class Environment
{
public:

	vector<TestFile> Files;
	vector<GpuSetup> Setups;
	vector<IPSetTest> IPSetTests;
	vector<IPSubsetTest> SubsetTests;

	void InitFiles()
	{
		Files.push_back(TestFile("../../../TestData/data-raw-table_australia_012016.txt", 565949));
		Files.push_back(TestFile("../../../TestData/data-raw-table_australia_092016.txt", 420206));
		Files.push_back(TestFile("../../../TestData/data-raw-table_honkkong_012016.txt", 565907));
		Files.push_back(TestFile("../../../TestData/data-raw-table_honkkong_092016.txt", 602812));
		Files.push_back(TestFile("../../../TestData/data-raw-table_london_012016.txt", 564426));
		Files.push_back(TestFile("../../../TestData/data-raw-table_london_092016.txt", 601489));
		Files.push_back(TestFile("../../../TestData/data-raw-table_tokyo_012016.txt", 576846));
		Files.push_back(TestFile("../../../TestData/data-raw-table_tokyo_092016.txt", 611460));
		Files.push_back(TestFile("../../../TestData/data-raw-table_usa_012016.txt", 561755));
		Files.push_back(TestFile("../../../TestData/data-raw-table_usa_092016.txt", 598473));
	}

	void InitSetups()
	{
		vector<int> devices = { 0 };
		vector<int> blocks = { 100, 200 };
		vector<int> threads = { 100, 200 };

		for (auto d : devices)
			for (auto b : blocks)
				for (auto t : threads)
					Setups.push_back(GpuSetup(b, t, d));
	}

	void InitIPSetTests()
	{
		vector<int> masksToLoad = { 100000, 200000 };

		for (auto f : Files)
			for (auto s : Setups)
				for(auto m : masksToLoad)
					IPSetTests.push_back(IPSetTest(f, s, m));
	}

	void InitSubsetTests()
	{
		vector<int> subsetSizes = { 50000, 75000 };

		for (auto t : IPSetTests)
			for (auto s : subsetSizes)
				SubsetTests.push_back(IPSubsetTest(t.File, t.Setup, t.MasksToLoad, s));
	}

	Environment()
	{
		InitFiles();
		InitSetups();
		InitIPSetTests();
		InitSubsetTests();
	}
};

static Environment ENV;
