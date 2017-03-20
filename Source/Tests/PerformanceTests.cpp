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
INSTANTIATE_TEST_CASE_P(Given_ProperSettings_TreeMatcherPerformanceMeasured, TreeMatcherPerformanceTest, testing::ValuesIn(ENV.TreeMatcherPerformanceTests));

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
	RTreeResult result = matcher.Match(matchSet);

	//then
	cout << "Model build time:" << matcher.ModelBuildTime << endl << "Matching time:" << result.MatchingTime << endl;

	//ENV.ResultsFile << testCase.DeviceID << ";" << testCase.Blocks << ";" << testCase.Threads << ";"
	//	<< testCase.ModelSubsetSize << ";" << testCase.MatchSubsetSize << ";" << testCase.RandomMasksSetSize << ";"
	//	<< testCase.SourceSet.FileName << ";" << matcher.ModelBuildTime << ";" << result.MatchingTime;
	//for (int i = 0; i < testCase.R.size(); ++i)
	//	ENV.ResultsFile << testCase.R[i] << ",";
	//ENV.ResultsFile << " } ";

	//ENV.ResultsFile << endl;

}
INSTANTIATE_TEST_CASE_P(Given_ProperSettings_TreeMatcherPerformanceMeasured, RTreeMatcherPerformanceTest, testing::ValuesIn(ENV.RTreeMatcherPerformanceTests));
