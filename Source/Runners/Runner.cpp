#include <cuda_runtime_api.h>
#include "../Utils/Utils.h"
#include "../Matchers/RTreeMatcher.cuh"
#include "../Tests/TestEnv.h"
#include <fstream>

int main(int argc, char ** argv)
{
//	if(argc < 11)
//	{
//		cerr << "Invalid parameters" << endl;
//		return EXIT_FAILURE;
//	}
//
//	int blocks = atoi(argv[1]);
//	int threads = atoi(argv[2]);
//	int deviceID = atoi(argv[3]);
//	int seed = atoi(argv[4]);
//	int modelSubsetSize = atoi(argv[5]);
//	int matchSubsetSize = atoi(argv[6]);
//	int randomMasksSetSize = atoi(argv[7]);
//	bool usePresorting = strcmp(argv[8], "true") == 0 || strcmp(argv[8], "True") == 0 || strcmp(argv[8], "TRUE") == 0;
//	bool useMidLevels = strcmp(argv[9], "true") == 0 || strcmp(argv[9], "True") == 0 || strcmp(argv[9], "TRUE") == 0;
//
//	srand(seed);
//	GpuSetup setup(blocks, threads, deviceID);
//
//	IPSet modelSet;
//	modelSet.Load(setup, argv[10], modelSubsetSize);
//
//	IPSet matchSet1 = modelSet.RandomSubset(matchSubsetSize);
//	IPSet matchSet2;
//	matchSet2.Generate(setup, randomMasksSetSize);
//	IPSet matchSet = matchSet1 + matchSet2;
//
//	TreeMatcher matcher(max(modelSet.Size, matchSet.Size));
//	matcher.UsePresorting = usePresorting;
//	matcher.UseMidLevels = useMidLevels;
//
//	//when
//	matcher.BuildModel(modelSet);
//	TreeResult result = matcher.Match(matchSet);
//
//	cout << deviceID << ";" << blocks << ";" << threads << ";"
//		<< modelSubsetSize << ";" << matchSubsetSize << ";" << randomMasksSetSize << ";"
//		<< argv[10] << ";" << (usePresorting ? "TRUE" : "FALSE") << ";" << (useMidLevels ? "TRUE" : "FALSE") << ";"
//		<< matcher.ModelBuildTime << ";" << result.PresortingTime << ";" << result.MatchingTime << endl;
//
//	//cleanup
//
//#ifndef NO_THREADS_TRACE
//	ofstream threadsFile(argv[11]);
//	for (int i = 0; i < matchSet.Size; ++i)
//		threadsFile << i << ";" << result.ThreadTimeStart[i] << ";" << result.ThreadTimeEnd[i] << endl;
//	threadsFile.close();
//#endif
//
//	modelSet.Dispose();
//	matchSet1.Dispose();
//	matchSet2.Dispose();
//	matchSet.Dispose();
//	matcher.Tree.Dispose();
//	GpuAssert(cudaDeviceReset(), "Reseting device in test failed");

	vector<int> MatchSetSize = { 250000, 500000, 750000, 1000000, 2500000, 5000000, 7500000, 10000000 };
	vector<bool> UsePresorting = { false, true };
	vector<float> RandomMasks = { 0, 0.25, 0.5, 0.75, 1 };

	int seed = atoi(argv[1]);
	int blocks = atoi(argv[2]);
	int threads = atoi(argv[3]);
	int deviceID = atoi(argv[4]);
	GpuSetup setup(blocks, threads, deviceID);

	string testDataPath(argv[5]);
	string fileName(argv[6]);
	int sourceSetSize = atoi(argv[7]);

	ofstream rTreeResultsFile;
	rTreeResultsFile.open(argv[8], ios_base::app);

	vector<int> r;
	int l = atoi(argv[9]);
	for (int i = 0; i < l; ++i)
		r.push_back(atoi(argv[9 + i]));

	size_t totalMemory, freeMemory1, freeMemory2;
	GpuAssert(cudaDeviceReset(), "Reseting device in test failed");
	srand(seed);

	IPSet modelSet;
	modelSet.Load(setup, testDataPath + fileName, sourceSetSize);

	RTreeMatcher matcher(r);

	GpuAssert(cudaMemGetInfo(&freeMemory1, &totalMemory), "Cannot check memory usage");
	matcher.BuildModel(modelSet);
	GpuAssert(cudaMemGetInfo(&freeMemory2, &totalMemory), "Cannot check memory usage");

	for (auto matchSetSize : MatchSetSize)
		for (auto randomSize : RandomMasks)
			for (auto usePresorting : UsePresorting)
			{

				int maskSubsetSize = (1 - randomSize) * matchSetSize;
				int rndSetSize = randomSize * matchSetSize;
				float presortingTime = 0;

				IPSet matchSet;
				if (maskSubsetSize == 0)
				{
					IPSet matchSet2;
					matchSet2.Generate(setup, rndSetSize);
					matchSet = matchSet2;
				}

				if (rndSetSize == 0)
				{
					IPSet matchSet1;
					matchSet1.RandomSubset(maskSubsetSize, modelSet);
					matchSet = matchSet1;
				}

				if (maskSubsetSize != 0 && rndSetSize != 0)
				{
					IPSet matchSet1;
					matchSet1.RandomSubset(maskSubsetSize, modelSet);

					IPSet matchSet2;
					matchSet2.Generate(setup, rndSetSize);
					matchSet = matchSet1 + matchSet2;
				}

				if (usePresorting)
				{
					Timer timer;
					timer.Start();
					matchSet.Sort();
					presortingTime = timer.Stop();
				}
				else
					matchSet.Randomize();

				Result result = matcher.Match(matchSet);

				rTreeResultsFile << seed << ";" << sourceSetSize << ";" << fileName << ";" << setup << maskSubsetSize << ";" << rndSetSize << ";" << matcher.ModelBuildTime << ";";
				rTreeResultsFile << usePresorting << ";" << presortingTime << ";" << result.MatchingTime << ";" << freeMemory1 - freeMemory2 << ";" << matcher.Model.L << ";";

				rTreeResultsFile << "{";
				for (int i = 0; i < r.size(); ++i)
					rTreeResultsFile << r[i] << ",";
				rTreeResultsFile << "}" << ";";

				for (int i = 0; i < matcher.Model.L; ++i)
					rTreeResultsFile << "{" << matcher.Model.GetMinListLenght(i) << "," << matcher.Model.GetMaxListLenght(i) << "," << ((matcher.Model.LevelsSizes[i] > 0) ? (matcher.Model.totalListItemsPerLevel[i] / matcher.Model.LevelsSizes[i]) : 0) << "," << matcher.Model.LevelsSizes[i] << "};";
				rTreeResultsFile << endl;
			}
		
	rTreeResultsFile.close();

	return EXIT_SUCCESS;
}
