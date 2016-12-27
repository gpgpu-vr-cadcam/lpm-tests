
#include <gtest/gtest.h>
#include "Tests.h"

int main(int argc, char **argv) {

	::testing::InitGoogleTest(&argc, argv);
	auto ret = RUN_ALL_TESTS();
	ENV.ResultsFile.close();
	return ret;
}