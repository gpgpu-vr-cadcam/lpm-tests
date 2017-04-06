#include "../Core/IPSet.cuh"
#include "TestEnv.h"
#include <gtest/gtest.h>

struct GenerateTest : testing::Test, testing::WithParamInterface<IPSetTest> {};
TEST_P(GenerateTest, For)
{
	//given
	IPSetTest testCase = GetParam();

	//when
	IPSet set;
	set.Generate(testCase.Setup, testCase.MasksToLoad);

	//then
	EXPECT_EQ(set.Size, testCase.MasksToLoad);
	EXPECT_TRUE(set.d_IPs != NULL);
	EXPECT_TRUE(set.d_Lenghts != NULL);
	EXPECT_EQ(set.Setup, testCase.Setup);

	//cout << set;
}
INSTANTIATE_TEST_CASE_P(Given_ProperIPSet_When_GenerateCalled_Then_DataLoaded, GenerateTest, testing::ValuesIn(ENV.IPSetTests));

struct LoadTest : testing::Test, testing::WithParamInterface<IPSetLoadTest>{};
TEST_P(LoadTest, For)
{
	//given
	IPSetLoadTest testCase = GetParam();

	//when
	IPSet set;
	set.Load(testCase.Setup, ENV.TestDataPath + testCase.File.FileName, testCase.MasksToLoad);

	//then
	EXPECT_EQ(set.Size, testCase.MasksToLoad);
	EXPECT_TRUE(set.d_IPs != NULL);
	EXPECT_TRUE(set.d_Lenghts != NULL);
	EXPECT_EQ(set.Setup, testCase.Setup);

	//cout << set;
}
INSTANTIATE_TEST_CASE_P(Given_ProperFileAndSetup_When_LoadCalled_Then_DataLoaded, LoadTest, testing::ValuesIn(ENV.IPSetLoadTests));

struct AddTest : testing::Test, testing::WithParamInterface<IPSetLoadTest> {};
TEST_P(AddTest, For)
{
	//given
	IPSetLoadTest testCase = GetParam();

	IPSet set1;
	set1.Load(testCase.Setup, ENV.TestDataPath + testCase.File.FileName, testCase.MasksToLoad);

	IPSet set2;
	set2.Generate(testCase.Setup, testCase.MasksToLoad);

	//when
	IPSet set = set1 + set2;

	//then
	EXPECT_EQ(set.Size, set1.Size + set2.Size);
	EXPECT_TRUE(set.d_IPs != NULL);
	EXPECT_TRUE(set.d_Lenghts != NULL);
	EXPECT_EQ(set.Setup, testCase.Setup);

	//cout << set;
}
INSTANTIATE_TEST_CASE_P(Given_ProperSets_When_AddCalled_Then_NewSetCreated, AddTest, testing::ValuesIn(ENV.IPSetLoadTests));


struct SubsetTest : testing::Test, testing::WithParamInterface<IPSubsetTest> {};
TEST_P(SubsetTest, For)
{
	//given
	IPSubsetTest testCase = GetParam();
	IPSet set;
	set.Load(testCase.Setup, ENV.TestDataPath + testCase.File.FileName, testCase.MasksToLoad);

	//when
	IPSet subset;
	subset.RandomSubset(testCase.SubsetSize, set);
	subset.Randomize();

	//then
	EXPECT_EQ(subset.Size, testCase.SubsetSize);
	EXPECT_TRUE(set.d_IPs != NULL);
	EXPECT_TRUE(set.d_Lenghts != NULL);
	EXPECT_EQ(subset.Setup, testCase.Setup);

	//cout << subset;
}
INSTANTIATE_TEST_CASE_P(Given_ProperIPSet_When_RandomSubsetCalled_Then_SubsetCreated, SubsetTest, testing::ValuesIn(ENV.SubsetTests));