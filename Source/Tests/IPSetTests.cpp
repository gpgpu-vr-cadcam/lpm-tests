#include "../Core/IPSet.cuh"
#include "Tests.h"
#include <gtest/gtest.h>

struct LoadTest : testing::Test, testing::WithParamInterface<IPSetTest>{};
TEST_P(LoadTest, For)
{
	//given
	IPSetTest testCase = GetParam();

	//when
	IPSet set;
	set.Load(testCase.Setup, testCase.File.Path, testCase.MasksToLoad);

	//then
	EXPECT_EQ(set.Size, testCase.MasksToLoad);
	EXPECT_TRUE(set.d_IPData != NULL);
	EXPECT_EQ(set.Setup, testCase.Setup);

	//cout << set;
}
INSTANTIATE_TEST_CASE_P(Given_ProperIPSet_When_RandomSubsetCalled_Then_SubsetCreated, LoadTest, testing::ValuesIn(ENV.IPSetTests));

struct SubsetTest : testing::Test, testing::WithParamInterface<IPSubsetTest> {};
TEST_P(SubsetTest, For)
{
	//given
	IPSubsetTest testCase = GetParam();
	IPSet set;
	set.Load(testCase.Setup, testCase.File.Path, testCase.MasksToLoad);

	//when
	IPSet subset = set.RandomSubset(testCase.SubsetSize);

	//then
	EXPECT_EQ(subset.Size, testCase.SubsetSize);
	EXPECT_TRUE(subset.d_IPData != NULL);
	EXPECT_EQ(subset.Setup, testCase.Setup);

	//cout << subset;
}
INSTANTIATE_TEST_CASE_P(Given_ProperFileAndSetup_When_LoadCalled_Then_DataLoaded, SubsetTest, testing::ValuesIn(ENV.SubsetTests));