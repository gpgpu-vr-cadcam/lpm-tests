#include "../Matchers/RTreeMatcher.cuh"
#include "Tests.h"
#include <gtest/gtest.h>

struct RTreeMatcherBuildModelTest : testing::Test, testing::WithParamInterface<RTreeMatcherTest> {};
TEST_P(RTreeMatcherBuildModelTest, For)
{
	//given
	RTreeMatcherTest testCase = GetParam();
	IPSet set;
	set.Load(testCase.Setup, ENV.TestDataPath + testCase.File.FileName, testCase.MasksToLoad);

	//when
	RTreeMatcher matcher(testCase.R);
	matcher.BuildModel(set);

	//then

	cout << "Model build time:" << matcher.ModelBuildTime << endl;
}
INSTANTIATE_TEST_CASE_P(Given_ProperIPSet_When_BuildModelCalled_Then_ModelCreated, RTreeMatcherBuildModelTest, testing::ValuesIn(ENV.RTreeMatcherTests));
