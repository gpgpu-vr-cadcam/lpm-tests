#include "Tests.h"
#include "../Matchers/TreeMatcher.cuh"
#include "../Matchers/RTreeMatcher.cuh"
#include <gtest/gtest.h>

struct TreeMatcherPerformanceTest : testing::Test, testing::WithParamInterface<TreeMatcherPerformanceTestCase> {};
TEST_P(TreeMatcherPerformanceTest, For)
{
	//given
	TreeMatcherPerformanceTestCase testCase = GetParam();

	srand(testCase.Seed);
	GpuSetup setup(testCase.Blocks, testCase.Threads, testCase.DeviceID);

	IPSet modelSet;
	modelSet.Load(setup, ENV.TestDataPath + testCase.SourceSet.FileName, testCase.ModelSubsetSize);

	IPSet matchSet1 = modelSet.RandomSubset(testCase.MatchSubsetSize);
	IPSet matchSet2;
	matchSet2.Generate(setup, testCase.RandomMasksSetSize);
	IPSet matchSet = matchSet1 + matchSet2;

	TreeMatcher matcher(max(modelSet.Size, matchSet.Size));
	matcher.UsePresorting = testCase.UsePresorting;
	matcher.UseMidLevels = testCase.UseMidLevels;

	//when
	matcher.BuildModel(modelSet);
	TreeResult result = matcher.Match(matchSet);

	//then
#ifndef NO_THREADS_TRACE
	int startLine = ENV.ThreadsFileLines;
	ENV.ThreadsFileLines += matchSet.Size;
#endif

	ENV.ResultsFile << testCase.DeviceID << ";" << testCase.Blocks << ";" << testCase.Threads << ";"
		<< testCase.ModelSubsetSize << ";" << testCase.MatchSubsetSize << ";" << testCase.RandomMasksSetSize << ";"
		<< testCase.SourceSet.FileName << ";" << testCase.UsePresorting << ";" << testCase.UseMidLevels << ";"
		<< matcher.ModelBuildTime << ";" << result.PresortingTime << ";" << result.MatchingTime;

#ifndef NO_THREADS_TRACE
		ENV.ResultsFile << ";" << startLine << ";" << ENV.ThreadsFileLines - 1;
#endif
		
		 ENV.ResultsFile << endl;
	
#ifndef NO_THREADS_TRACE
	for (int i = 0; i < matchSet.Size; ++i)
		ENV.ThreadsFile << i << ";" << result.ThreadTimeStart[i] << ";" << result.ThreadTimeEnd[i] << endl;
#endif

	cout << "Model build time:" << matcher.ModelBuildTime << endl << "Matching time:" << result.MatchingTime << endl << "Presorting time:" << result.PresortingTime;

	//cleanup
	GpuAssert(cudaSetDevice(setup.DeviceID), "Cannot set cuda device.");
	modelSet.Dispose();
	matchSet1.Dispose();
	matchSet2.Dispose();
	matchSet.Dispose();
	matcher.Tree.Dispose();
	GpuAssert(cudaDeviceReset(), "Reseting device in test failed");
	GpuAssert(cudaSetDevice(0), "Cannot set cuda device.");
}
//INSTANTIATE_TEST_CASE_P(Given_ProperSettings_TreeMatcherPerformanceMeasured, TreeMatcherPerformanceTest, testing::ValuesIn(ENV.TreeMatcherPerformanceTests));

struct RTreeMatcherPerformanceTest : testing::Test, testing::WithParamInterface<RTreeMatcherPerformanceTestCase> {};
TEST_P(RTreeMatcherPerformanceTest, For)
{
	//given
	RTreeMatcherPerformanceTestCase testCase = GetParam();

	srand(testCase.Seed);
	GpuSetup setup(testCase.Blocks, testCase.Threads, testCase.DeviceID);

	IPSet modelSet;
	modelSet.Load(setup, ENV.TestDataPath + testCase.SourceSet.FileName, testCase.ModelSubsetSize);

	IPSet matchSet1 = modelSet.RandomSubset(testCase.MatchSubsetSize);
	IPSet matchSet2;
	matchSet2.Generate(setup, testCase.RandomMasksSetSize);
	IPSet matchSet = matchSet1 + matchSet2;

	RTreeMatcher matcher(testCase.R);

	//when
	matcher.BuildModel(modelSet);
	Result result = matcher.Match(matchSet);

	//then
	cout << "Model build time:" << matcher.ModelBuildTime << endl << "Matching time:" << result.MatchingTime << endl;

	ENV.ResultsFile << testCase.Seed << ";" << testCase.SourceSet.FileName << ";" << testCase.ModelSubsetSize << ";" << testCase.MatchSubsetSize << ";" << testCase.RandomMasksSetSize << ";" 
		<< testCase.Blocks << ";" << testCase.Threads << ";" << testCase.DeviceID << ";" << "{";
		for (int i = 0; i < testCase.R.size(); ++i)
			ENV.ResultsFile << testCase.R[i] << ";";
	ENV.ResultsFile << "}" << ";" << matcher.ModelBuildTime << ";" << result.MatchingTime << endl;

}
//INSTANTIATE_TEST_CASE_P(Given_ProperSettings_TreeMatcherPerformanceMeasured, RTreeMatcherPerformanceTest, testing::ValuesIn(ENV.RTreeMatcherPerformanceTests));

struct RTreeMatcherListsLenghtsTest : testing::Test, testing::WithParamInterface<RTreeListsLenghtsTestCase> {};
TEST_P(RTreeMatcherListsLenghtsTest, For)
{
	//given
	RTreeListsLenghtsTestCase testCase = GetParam();

	GpuSetup setup(testCase.Blocks, testCase.Threads, 0);

	IPSet modelSet;
	modelSet.Load(setup, ENV.TestDataPath + testCase.File.FileName, testCase.File.Size);
	RTreeMatcher matcher(testCase.R);

	srand(1234);
	IPSet matchSet1 = modelSet.RandomSubset(200000);
	srand(1234);
	IPSet matchSet2;
	matchSet2.Generate(setup, 200000);

	//when
	matcher.BuildModel(modelSet);
	auto result1 = matcher.Match(modelSet);
	auto result2 = matcher.Match(matchSet1);
	auto result3 = matcher.Match(matchSet2);
	
	ENV.ListLenghtsFile << testCase.File.FileName << ";" << testCase.File.Size << ";" << setup.Blocks << ";" << setup.Threads << ";" << matcher.Model.L << ";" << "{";
	for (int i = 0; i < testCase.R.size(); ++i)
		ENV.ListLenghtsFile << testCase.R[i] << ";";
	ENV.ListLenghtsFile << "}" << ";" << matcher.ModelBuildTime << ";";
	ENV.ListLenghtsFile << result1.MatchingTime << ";";
	ENV.ListLenghtsFile << result2.MatchingTime << ";";
	ENV.ListLenghtsFile << result3.MatchingTime << ";";


	for(int i = 0; i < matcher.Model.L; ++i)
	{
		ENV.ListLenghtsFile << "{" << matcher.Model.GetMinListLenght(i) << ";" << matcher.Model.GetMaxListLenght(i) << ";" << matcher.Model.totalListItemsPerLevel[i] / matcher.Model.LevelsSizes[i] << ";" << matcher.Model.LevelsSizes[i] << "};";
	}

	ENV.ListLenghtsFile << endl;

}
INSTANTIATE_TEST_CASE_P(Given_ProperSettings_TreeMatcherPerformanceMeasured, RTreeMatcherListsLenghtsTest, testing::ValuesIn(ENV.RTreeListsLenghtsTests));
