#include "../Matchers/RTreeMatcher.cuh"
#include "TestEnv.h"
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

struct RTreeMatcherBasicMatchTest : testing::Test, testing::WithParamInterface<RTreeMatcherTest> {};
TEST_P(RTreeMatcherBasicMatchTest, For)
{
	//given
	RTreeMatcherTest testCase = GetParam();
	IPSet set;
	set.Load(testCase.Setup, ENV.TestDataPath + testCase.File.FileName, testCase.MasksToLoad);

	RTreeMatcher matcher(testCase.R);
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
INSTANTIATE_TEST_CASE_P(Given_ProperTreeMatcher_When_MatchCalled_Then_IPsMatched, RTreeMatcherBasicMatchTest, testing::ValuesIn(ENV.RTreeMatcherTests));
