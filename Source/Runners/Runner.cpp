#include <cuda_runtime_api.h>
#include "../Utils/Utils.h"
#include "../Matchers/TreeMatcher.cuh"
#include <algorithm>
#include <iostream>
#include <fstream>

int main(int argc, char ** argv)
{
	if(argc < 11)
	{
		cerr << "Invalid parameters" << endl;
		return EXIT_FAILURE;
	}

	int blocks = atoi(argv[1]);
	int threads = atoi(argv[2]);
	int deviceID = atoi(argv[3]);
	int seed = atoi(argv[4]);
	int modelSubsetSize = atoi(argv[5]);
	int matchSubsetSize = atoi(argv[6]);
	int randomMasksSetSize = atoi(argv[7]);
	bool usePresorting = strcmp(argv[8], "true") == 0;
	bool useMidLevels = strcmp(argv[9], "true") == 0;

	srand(seed);
	GpuSetup setup(blocks, threads, deviceID);

	IPSet modelSet;
	modelSet.Load(setup, argv[10], modelSubsetSize);

	IPSet matchSet1 = modelSet.RandomSubset(matchSubsetSize);
	IPSet matchSet2;
	matchSet2.Generate(setup, randomMasksSetSize);
	IPSet matchSet = matchSet1 + matchSet2;

	TreeMatcher matcher(max(modelSet.Size, matchSet.Size));
	matcher.UsePresorting = usePresorting;
	matcher.UseMidLevels = useMidLevels;

	//when
	matcher.BuildModel(modelSet);
	TreeResult result = matcher.Match(matchSet);

	cout << deviceID << ";" << blocks << ";" << threads << ";"
		<< modelSubsetSize << ";" << matchSubsetSize << ";" << randomMasksSetSize << ";"
		<< argv[10] << ";" << usePresorting << ";" << useMidLevels << ";"
		<< matcher.ModelBuildTime << ";" << result.PresortingTime << ";" << result.MatchingTime << endl;

	//cleanup
	GpuAssert(cudaSetDevice(setup.DeviceID), "Cannot set cuda device.");

#ifndef NO_THREADS_TRACE
	ofstream threadsFile(argv[11]);
	for (int i = 0; i < matchSet.Size; ++i)
		threadsFile << i << ";" << result.ThreadTimeStart[i] << ";" << result.ThreadTimeEnd[i] << endl;
	threadsFile.close();
#endif

	modelSet.Dispose();
	matchSet1.Dispose();
	matchSet2.Dispose();
	matchSet.Dispose();
	matcher.Tree.Dispose();
	GpuAssert(cudaDeviceReset(), "Reseting device in test failed");
	GpuAssert(cudaSetDevice(0), "Cannot set cuda device.");

	return EXIT_SUCCESS;
}
