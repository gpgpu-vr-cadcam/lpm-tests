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
	set.Load(testCase.Setup, testCase.File.Path);

	//then
	EXPECT_EQ(set.Size, testCase.File.Size);
	EXPECT_TRUE(set.d_IPData != NULL);
	EXPECT_EQ(set.Setup, testCase.Setup);
}

INSTANTIATE_TEST_CASE_P(Given_ProperFileAndSetup_When_LoadCalled_Then_DataLoaded, LoadTest, testing::ValuesIn(ENV.IPSetTests));