
#include <gtest/gtest.h>
#include "TestEnv.h"

Environment ENV;

int main(int argc, char **argv) 
{
	if (argc > 1)
		ENV.TestDataPath = string(argv[1]);

	::testing::InitGoogleTest(&argc, argv);
	auto ret = RUN_ALL_TESTS();

	ENV.ArrayResultsFile.close();
	ENV.TreeResultsFile.close();
	ENV.RTreeResultsFile.close();

#ifndef NO_THREADS_TRACE
	ENV.ThreadsFile.close();
#endif

	return ret;
}