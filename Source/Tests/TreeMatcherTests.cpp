#include "../Matchers/TreeMatcher.cuh"
#include "Tests.h"
#include <gtest/gtest.h>

struct TreeMatcherBuildModelTest : testing::Test, testing::WithParamInterface<IPSubsetTest>{};
TEST_P(TreeMatcherBuildModelTest, For)
{
	//given
	IPSubsetTest testCase = GetParam();
	IPSet set;
	set.Load(testCase.Setup, testCase.File.Path, testCase.MasksToLoad);

	//when
	TreeMatcher matcher(set.Size);
	matcher.BuildModel(set);

	//then
	EXPECT_TRUE(matcher.Tree.d_Level16 != NULL);
	EXPECT_TRUE(matcher.Tree.d_Level24 != NULL);
	EXPECT_TRUE(matcher.Tree.d_Level8 != NULL);
	EXPECT_TRUE(matcher.Tree.d_LevelNodes != NULL);
	EXPECT_TRUE(matcher.Tree.d_PrefixesEnd != NULL);
	EXPECT_TRUE(matcher.Tree.d_PrefixesStart != NULL);
	EXPECT_TRUE(matcher.Tree.d_SortedSubnetBits != NULL);

	//cleanup due to setting cudaLimitMallocHeapSize in this matcher
	matcher.Tree.Dispose();
	set.Dispose();
	GpuAssert(cudaDeviceReset(), "Reseting device in test failed");

	cout << "Model build time:" << matcher.ModelBuildTime << endl;
}
INSTANTIATE_TEST_CASE_P(Given_ProperIPSet_When_BuildModelCalled_Then_ModelCreated, TreeMatcherBuildModelTest, testing::ValuesIn(ENV.SubsetTests));

struct TreeMatcherBasicMatchTest : testing::Test, testing::WithParamInterface<IPSubsetTest>{};
TEST_P(TreeMatcherBasicMatchTest, For)
{
	//given
	IPSubsetTest testCase = GetParam();
	IPSet set;
	set.Load(testCase.Setup, testCase.File.Path, testCase.MasksToLoad);

	TreeMatcher matcher(set.Size);
	matcher.BuildModel(set);

	//when
	TreeResult result = matcher.Match(set);

	//then
	EXPECT_TRUE(result.MatchedIndexes != NULL);
	EXPECT_TRUE(result.IPsList != NULL);
	EXPECT_TRUE(result.SortedSubnetsBits != NULL);
	EXPECT_EQ(result.IPsListSize, set.Size);
	EXPECT_TRUE(result.ThreadTime != NULL);
	EXPECT_TRUE(result.ThreadTimeEnd != NULL);
	EXPECT_TRUE(result.ThreadTimeStart != NULL);
	EXPECT_TRUE(result.ThreadTimeRecorded);
	EXPECT_EQ(result.CountMatched(), set.Size);

	//result.PrintResult();

	//cleanup due to setting cudaLimitMallocHeapSize in this matcher
	matcher.Tree.Dispose();
	set.Dispose();
	GpuAssert(cudaDeviceReset(), "Reseting device in test failed");

	cout << "Model build time:" << matcher.ModelBuildTime << endl << "Matching time:" << result.MatchingTime << endl;
}
INSTANTIATE_TEST_CASE_P(Given_ProperTreeMatcher_When_MatchCalled_Then_IPsMatched, TreeMatcherBasicMatchTest, testing::ValuesIn(ENV.SubsetTests));

struct TreeMatcherBasicMatchTestWithMidLevels : testing::Test, testing::WithParamInterface<IPSubsetTest> {};
TEST_P(TreeMatcherBasicMatchTestWithMidLevels, For)
{
	//given
	IPSubsetTest testCase = GetParam();
	IPSet set;
	set.Load(testCase.Setup, testCase.File.Path, testCase.MasksToLoad);

	TreeMatcher matcher(set.Size);
	matcher.UseMidLevels = true;
	matcher.BuildModel(set);

	//when
	TreeResult result = matcher.Match(set);

	//then
	EXPECT_TRUE(result.MatchedIndexes != NULL);
	EXPECT_TRUE(result.IPsList != NULL);
	EXPECT_TRUE(result.SortedSubnetsBits != NULL);
	EXPECT_EQ(result.IPsListSize, set.Size);
	EXPECT_TRUE(result.ThreadTime != NULL);
	EXPECT_TRUE(result.ThreadTimeEnd != NULL);
	EXPECT_TRUE(result.ThreadTimeStart != NULL);
	EXPECT_TRUE(result.ThreadTimeRecorded);
	EXPECT_EQ(result.CountMatched(), set.Size);

	//result.PrintResult();

	//cleanup due to setting cudaLimitMallocHeapSize in this matcher
	matcher.Tree.Dispose();
	set.Dispose();
	GpuAssert(cudaDeviceReset(), "Reseting device in test failed");

	cout << "Model build time:" << matcher.ModelBuildTime << endl << "Matching time:" << result.MatchingTime << endl;
}
INSTANTIATE_TEST_CASE_P(Given_ProperTreeMatcherWithMidLevels_When_MatchCalled_Then_IPsMatched, TreeMatcherBasicMatchTestWithMidLevels, testing::ValuesIn(ENV.SubsetTests));

struct TreeMatcherBasicMatchTestWithPresorting : testing::Test, testing::WithParamInterface<IPSubsetTest> {};
TEST_P(TreeMatcherBasicMatchTestWithPresorting, For)
{
	//given
	IPSubsetTest testCase = GetParam();
	IPSet set;
	set.Load(testCase.Setup, testCase.File.Path, testCase.MasksToLoad);

	TreeMatcher matcher(set.Size);
	matcher.UsePresorting = true;
	matcher.BuildModel(set);

	//when
	TreeResult result = matcher.Match(set);

	//then
	EXPECT_TRUE(result.MatchedIndexes != NULL);
	EXPECT_TRUE(result.IPsList != NULL);
	EXPECT_TRUE(result.SortedSubnetsBits != NULL);
	EXPECT_EQ(result.IPsListSize, set.Size);
	EXPECT_TRUE(result.ThreadTime != NULL);
	EXPECT_TRUE(result.ThreadTimeEnd != NULL);
	EXPECT_TRUE(result.ThreadTimeStart != NULL);
	EXPECT_TRUE(result.ThreadTimeRecorded);
	EXPECT_EQ(result.CountMatched(), set.Size);

	//result.PrintResult();

	//cleanup due to setting cudaLimitMallocHeapSize in this matcher
	matcher.Tree.Dispose();
	set.Dispose();
	GpuAssert(cudaDeviceReset(), "Reseting device in test failed");

	cout << "Model build time:" << matcher.ModelBuildTime << endl << "Matching time:" << result.MatchingTime << endl;
}
INSTANTIATE_TEST_CASE_P(Given_ProperTreeMatcherWithMidLevels_When_MatchCalled_Then_IPsMatched, TreeMatcherBasicMatchTestWithPresorting, testing::ValuesIn(ENV.SubsetTests));

struct TreeMatcherBasicMatchTestWithMidLevelsAndPresorting : testing::Test, testing::WithParamInterface<IPSubsetTest> {};
TEST_P(TreeMatcherBasicMatchTestWithMidLevelsAndPresorting, For)
{
	//given
	IPSubsetTest testCase = GetParam();
	IPSet set;
	set.Load(testCase.Setup, testCase.File.Path, testCase.MasksToLoad);

	TreeMatcher matcher(set.Size);
	matcher.UseMidLevels = true;
	matcher.UsePresorting = true;
	matcher.BuildModel(set);

	//when
	TreeResult result = matcher.Match(set);

	//then
	EXPECT_TRUE(result.MatchedIndexes != NULL);
	EXPECT_TRUE(result.IPsList != NULL);
	EXPECT_TRUE(result.SortedSubnetsBits != NULL);
	EXPECT_EQ(result.IPsListSize, set.Size);
	EXPECT_TRUE(result.ThreadTime != NULL);
	EXPECT_TRUE(result.ThreadTimeEnd != NULL);
	EXPECT_TRUE(result.ThreadTimeStart != NULL);
	EXPECT_TRUE(result.ThreadTimeRecorded);
	EXPECT_EQ(result.CountMatched(), set.Size);

	//result.PrintResult();

	//cleanup due to setting cudaLimitMallocHeapSize in this matcher
	matcher.Tree.Dispose();
	set.Dispose();
	GpuAssert(cudaDeviceReset(), "Reseting device in test failed");

	cout << "Model build time:" << matcher.ModelBuildTime << endl << "Matching time:" << result.MatchingTime << endl;
}
INSTANTIATE_TEST_CASE_P(Given_ProperTreeMatcherWithMidLevels_When_MatchCalled_Then_IPsMatched, TreeMatcherBasicMatchTestWithMidLevelsAndPresorting, testing::ValuesIn(ENV.SubsetTests));
