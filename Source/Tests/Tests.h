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

	IPSetTest(TestFile& file, GpuSetup& setup)
		: File(file),
		  Setup(setup) {}

	friend std::ostream& operator<<(std::ostream& os, const IPSetTest& obj)
	{
		return os
			<< "File: " << obj.File
			<< " Setup: " << obj.Setup;
	}
};

class Environment
{
public:

	vector<TestFile> Files;
	vector<GpuSetup> Setups;
	vector<IPSetTest> IPSetTests;

	void InitFiles()
	{
		Files.push_back(TestFile("../../../TestData/data-raw-table_australia_012016.txt", 565949));
		Files.push_back(TestFile("../../../TestData/data-raw-table_australia_092016.txt", 420207));
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
		for (auto f : Files)
			for (auto s : Setups)
				IPSetTests.push_back(IPSetTest(f, s));
	}

	Environment()
	{
		InitFiles();
		InitSetups();
		InitIPSetTests();
	}
};

static Environment ENV;
