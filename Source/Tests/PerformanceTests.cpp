#include "TestEnv.h"
#include "../Matchers/TreeMatcher.cuh"
#include "../Matchers/RTreeMatcher.cuh"
#include "../Matchers/ArrayMatcher.cuh"
#include <gtest/gtest.h>

struct RTreeMatcherPerformanceTest : testing::Test, testing::WithParamInterface<RTreeMatcherPerformanceTestCase> {};
TEST_P(RTreeMatcherPerformanceTest, For)
{
	size_t totalMemory, freeMemory1, freeMemory2;

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
//INSTANTIATE_TEST_CASE_P(Test, RTreeMatcherPerformanceTest, testing::ValuesIn(ENV.RTreeMatcherPerformanceTests));