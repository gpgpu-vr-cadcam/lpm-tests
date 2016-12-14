#include "../Matchers/TreeMatcher.cuh"
#include "Tests.h"
#include <gtest/gtest.h>

struct TreeMatcherTest : testing::Test, testing::WithParamInterface<IPSubsetTest> {};
TEST_P(TreeMatcherTest, For)
{
	//given
	IPSubsetTest testCase = GetParam();
	IPSet set;
	set.Load(testCase.Setup, testCase.File.Path, testCase.MasksToLoad);

	//when
	TreeMatcher matcher;
	matcher.BuildModel(set);

	//cout << subset;
}
INSTANTIATE_TEST_CASE_P(Given_ProperIPSet_When_BuildModelCalled_Then_ModelCreated, TreeMatcherTest, testing::ValuesIn(ENV.SubsetTests));