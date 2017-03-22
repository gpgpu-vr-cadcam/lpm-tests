#include "../Matchers/ArrayMatcher.cuh"
#include "Tests.h"
#include <gtest/gtest.h>

struct ArrayMatcherBuildModelTest : testing::Test, testing::WithParamInterface<IPSubsetTest> {};
TEST_P(ArrayMatcherBuildModelTest, For)
{
	//given
	IPSubsetTest testCase = GetParam();
	IPSet set;
	set.Load(testCase.Setup, ENV.TestDataPath + testCase.File.FileName, testCase.MasksToLoad);

	//when
	ArrayMatcher matcher;
	matcher.BuildModel(set);

	//then
	cout << "Model build time:" << matcher.ModelBuildTime << endl;
}
INSTANTIATE_TEST_CASE_P(Given_ProperIPSet_When_BuildModelCalled_Then_ModelCreated, ArrayMatcherBuildModelTest, testing::ValuesIn(ENV.SubsetTests));

struct ArrayMatcherBasicMatchTest : testing::Test, testing::WithParamInterface<IPSubsetTest> {};
TEST_P(ArrayMatcherBasicMatchTest, For)
{
	//given
	IPSubsetTest testCase = GetParam();
	IPSet set;
	set.Load(testCase.Setup, ENV.TestDataPath + testCase.File.FileName, testCase.MasksToLoad);

	ArrayMatcher matcher;
	matcher.BuildModel(set);

	//when
	Result result = matcher.Match(set);

	//then
	EXPECT_TRUE(result.MatchedMaskIndex != NULL);
	EXPECT_EQ(result.IpsToMatchCount, set.Size);
	//EXPECT_TRUE(result.ThreadTimeEnd != NULL);
	//EXPECT_TRUE(result.ThreadTimeStart != NULL);
	//EXPECT_TRUE(result.ThreadTimeRecorded);
	EXPECT_EQ(result.CountMatched(), set.Size);

	//result.PrintResult();

	cout << "Model build time:" << matcher.ModelBuildTime << endl << "Matching time:" << result.MatchingTime << endl;
}
INSTANTIATE_TEST_CASE_P(Given_ProperTreeMatcher_When_MatchCalled_Then_IPsMatched, ArrayMatcherBasicMatchTest, testing::ValuesIn(ENV.SubsetTests));