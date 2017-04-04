#pragma once
#include "../Utils/Utils.h"

#include <vector>
#include <string>
#include <ostream>

class TestFile
{
public:
	int Size;
	string FileName;

	TestFile(const string& file_name, int size)
		: Size(size), FileName(file_name) {}

	friend std::ostream& operator<<(std::ostream& os, const TestFile& obj)
	{
		return os << obj.Size << ";" << obj.FileName << ";";
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
