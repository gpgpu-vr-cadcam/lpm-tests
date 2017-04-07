#include "TestEnv.h"
#include "../Matchers/TreeMatcher.cuh"
#include "../Matchers/RTreeMatcher.cuh"
#include "../Matchers/ArrayMatcher.cuh"
#include <gtest/gtest.h>

struct TreeMatcherPerformanceTest : testing::Test, testing::WithParamInterface<TreeMatcherPerformanceTestCase> {};
TEST_P(TreeMatcherPerformanceTest, For)
{
	size_t totalMemory, freeMemory1, freeMemory2;
	GpuAssert(cudaDeviceReset(), "Reseting device in test failed");
	TreeMatcherPerformanceTestCase testCase = GetParam();
	srand(testCase.Seed);

	IPSet modelSet;
	modelSet.Load(testCase.Setup, ENV.TestDataPath + testCase.SourceSet.FileName, testCase.SourceSet.Size);

	TreeMatcher matcher(max(testCase.MaxMatchSetSize(), modelSet.Size));
	matcher.UseMidLevels = testCase.UseMidLevels;

	GpuAssert(cudaMemGetInfo(&freeMemory1, &totalMemory), "Cannot check memory usage");
	matcher.BuildModel(modelSet);
	GpuAssert(cudaMemGetInfo(&freeMemory2, &totalMemory), "Cannot check memory usage");

	for (auto matchSetSize : testCase.MatchSubsetSize)
		for (auto randomSize : testCase.RandomMasksSetSize)
			for (auto usePresorting : testCase.PresortMatchSet)
			{
				int maskSubsetSize = (1 - randomSize) * matchSetSize;
				int rndSetSize = randomSize * matchSetSize;
				float presortingTime = 0;

				IPSet matchSet;
				if (maskSubsetSize == 0)
				{
					IPSet matchSet2;
					matchSet2.Generate(testCase.Setup, rndSetSize);
					matchSet = matchSet2;
				}

				if (rndSetSize == 0)
				{
					IPSet matchSet1;
					matchSet1.RandomSubset(maskSubsetSize, modelSet);
					matchSet = matchSet1;
				}

				if (maskSubsetSize != 0 && rndSetSize != 0)
				{
					IPSet matchSet1;
					matchSet1.RandomSubset(maskSubsetSize, modelSet);

					IPSet matchSet2;
					matchSet2.Generate(testCase.Setup, rndSetSize);
					matchSet = matchSet1 + matchSet2;
				}

				if (usePresorting)
				{
					Timer timer;
					timer.Start();
					matchSet.Sort();
					presortingTime = timer.Stop();
				}
				else
					matchSet.Randomize();

				TreeResult result = matcher.Match(matchSet);

				ENV.TreeResultsFile << testCase << maskSubsetSize << ";" << rndSetSize << ";" << testCase.UseMidLevels << ";" << matcher.ModelBuildTime << ";";
				ENV.TreeResultsFile << usePresorting << ";" << presortingTime << ";" << result.MatchingTime << ";" << freeMemory1 - freeMemory2 << ";" << endl;
			}

	//cleanup
	modelSet.Dispose();
	matcher.Tree.Dispose();
	GpuAssert(cudaDeviceReset(), "Reseting device in test failed");
}
INSTANTIATE_TEST_CASE_P(Test, TreeMatcherPerformanceTest, testing::ValuesIn(ENV.TreeMatcherPerformanceTests));

struct ArrayMatcherPerformanceTest : testing::Test, testing::WithParamInterface<PerformanceTest> {};
TEST_P(ArrayMatcherPerformanceTest, For)
{
	size_t totalMemory, freeMemory1, freeMemory2;
	PerformanceTest testCase = GetParam();
	GpuAssert(cudaDeviceReset(), "Reseting device in test failed");
	srand(testCase.Seed);

	IPSet modelSet;
	modelSet.Load(testCase.Setup, ENV.TestDataPath + testCase.SourceSet.FileName, testCase.SourceSet.Size);

	ArrayMatcher matcher;
	GpuAssert(cudaMemGetInfo(&freeMemory1, &totalMemory), "Cannot check memory usage");
	matcher.BuildModel(modelSet);
	GpuAssert(cudaMemGetInfo(&freeMemory2, &totalMemory), "Cannot check memory usage");

	for (auto matchSetSize : testCase.MatchSubsetSize)
		for (auto randomSize : testCase.RandomMasksSetSize)
			for (auto usePresorting : testCase.PresortMatchSet)
			{
				int maskSubsetSize = (1 - randomSize) * matchSetSize;
				int rndSetSize = randomSize * matchSetSize;
				float presortingTime = 0;

				IPSet matchSet;
				if (maskSubsetSize == 0)
				{
					IPSet matchSet2;
					matchSet2.Generate(testCase.Setup, rndSetSize);
					matchSet = matchSet2;
				}

				if (rndSetSize == 0)
				{
					IPSet matchSet1;
					matchSet1.RandomSubset(maskSubsetSize, modelSet);
					matchSet = matchSet1;
				}

				if (maskSubsetSize != 0 && rndSetSize != 0)
				{
					IPSet matchSet1;
					matchSet1.RandomSubset(maskSubsetSize, modelSet);

					IPSet matchSet2;
					matchSet2.Generate(testCase.Setup, rndSetSize);
					matchSet = matchSet1 + matchSet2;
				}

				if (usePresorting)
				{
					Timer timer;
					timer.Start();
					matchSet.Sort();
					presortingTime = timer.Stop();
				}
				else
					matchSet.Randomize();

				Result result = matcher.Match(matchSet);

				ENV.ArrayResultsFile << testCase << maskSubsetSize << ";" << rndSetSize << ";" << matcher.ModelBuildTime << ";";
				ENV.ArrayResultsFile << usePresorting << ";" << presortingTime << ";" << result.MatchingTime << ";" << freeMemory1 - freeMemory2 << ";" << endl;
			}
}
INSTANTIATE_TEST_CASE_P(Test, ArrayMatcherPerformanceTest, testing::ValuesIn(ENV.PerformanceTests));

struct RTreeMatcherPerformanceTest : testing::Test, testing::WithParamInterface<RTreeMatcherPerformanceTestCase> {};
TEST_P(RTreeMatcherPerformanceTest, For)
{
	size_t totalMemory, freeMemory1, freeMemory2;
	GpuAssert(cudaDeviceReset(), "Reseting device in test failed");
	RTreeMatcherPerformanceTestCase testCase = GetParam();
	srand(testCase.Seed);

	IPSet modelSet;
	modelSet.Load(testCase.Setup, ENV.TestDataPath + testCase.SourceSet.FileName, testCase.SourceSet.Size);

	RTreeMatcher matcher(testCase.R);

	GpuAssert(cudaMemGetInfo(&freeMemory1, &totalMemory), "Cannot check memory usage");
	matcher.BuildModel(modelSet);
	GpuAssert(cudaMemGetInfo(&freeMemory2, &totalMemory), "Cannot check memory usage");

	for(auto matchSetSize : testCase.MatchSubsetSize)
		for(auto randomSize : testCase.RandomMasksSetSize)
			for(auto usePresorting : testCase.PresortMatchSet)
			{
				int maskSubsetSize = (1 - randomSize) * matchSetSize;
				int rndSetSize = randomSize * matchSetSize;
				float presortingTime = 0;

				IPSet matchSet;
				if(maskSubsetSize == 0)
				{
					IPSet matchSet2;
					matchSet2.Generate(testCase.Setup, rndSetSize);
					matchSet = matchSet2;
				}

				if(rndSetSize == 0)
				{
					IPSet matchSet1;
					matchSet1.RandomSubset(maskSubsetSize, modelSet);
					matchSet = matchSet1;
				}

				if(maskSubsetSize != 0 && rndSetSize != 0)
				{
					IPSet matchSet1;
					matchSet1.RandomSubset(maskSubsetSize, modelSet);

					IPSet matchSet2;
					matchSet2.Generate(testCase.Setup, rndSetSize);
					matchSet = matchSet1 + matchSet2;
				}

				if (usePresorting)
				{
					Timer timer;
					timer.Start();
					matchSet.Sort();
					presortingTime = timer.Stop();
				}
				else
					matchSet.Randomize();

				Result result = matcher.Match(matchSet);

				ENV.RTreeResultsFile << testCase << maskSubsetSize << ";" << rndSetSize << ";" << matcher.ModelBuildTime << ";";
				ENV.RTreeResultsFile << usePresorting << ";" << presortingTime << ";" << result.MatchingTime << ";" << freeMemory1 - freeMemory2 << ";" << matcher.Model.L << ";";

				ENV.RTreeResultsFile << "{";
				for (int i = 0; i < testCase.R.size(); ++i)
					ENV.RTreeResultsFile << testCase.R[i] << ",";
				ENV.RTreeResultsFile << "}" << ";";

				for (int i = 0; i < matcher.Model.L; ++i)
					ENV.RTreeResultsFile << "{" << matcher.Model.GetMinListLenght(i) << "," << matcher.Model.GetMaxListLenght(i) << "," << ((matcher.Model.LevelsSizes[i] > 0) ? (matcher.Model.totalListItemsPerLevel[i] / matcher.Model.LevelsSizes[i]) : 0) << "," << matcher.Model.LevelsSizes[i] << "};";
				ENV.RTreeResultsFile << endl;
			}

}
INSTANTIATE_TEST_CASE_P(Test, RTreeMatcherPerformanceTest, testing::ValuesIn(ENV.RTreeMatcherPerformanceTests));