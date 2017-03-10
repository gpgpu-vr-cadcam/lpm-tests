#include "RTreeMatcher.cuh"
#include <device_launch_parameters.h>

#include <thrust/scan.h>
#include <thrust/execution_policy.h>

__global__ void CopyMasks(int Count, int *R, int *rSums, int L, int** Masks, int *Lenghts, unsigned char *IPData)
{
	int mask = blockIdx.x * blockDim.x + threadIdx.x;

	while  (mask < Count)
	{
		int address = 0;
		Lenghts[mask] = IPData[mask * 5 + 4];

		int part;
		for (int i = 0; i < 4; ++i)
		{
			part = IPData[mask * 5 + i];
			address |= part << (8 * (3 - i));
		}

		for (int l = 0; l < L; ++l)
			Masks[l][mask] = (address >> (32 - rSums[l])) & ((2 << R[l] - 1) - 1);

		mask += blockDim.x * gridDim.x;
	}
}

__global__ void MarkNodesBorders(int Count, int l, int **nodesBorders, int **Masks)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
	while(i < Count)
	{
		if (Masks[l - 1][i - 1] != Masks[l - 1][i] || nodesBorders[l - 1][i] == 1)
			nodesBorders[l][i] = 1;

		i += blockDim.x * gridDim.x;
	}
}

__global__ void FillIndexes(int Count, int l, int **nodesIndexes, int **startIndexes, int **endIndexes)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
	while (i < Count)
	{
		if (nodesIndexes[l][i] > 0)
		{
			startIndexes[l][nodesIndexes[l][i] - 1] = i;
			endIndexes[l][nodesIndexes[l][i] - 2] = i;
		}

		i += blockDim.x * gridDim.x;
	}
}

__global__ void FillChildren(int l, int *LevelsSizes, int **startIndexes, int **endIndexes, int **Children, int *ChildrenCount, int **Masks, int **nodesIndexes)
{
	int node = blockIdx.x;
	while (node < LevelsSizes[l])
	{
		int i = startIndexes[l][node] + threadIdx.x;
		while (i < endIndexes[l][node])
		{
			if (nodesIndexes[l + 1][i] > 0)
				Children[l][node * ChildrenCount[l] + Masks[l][i]] = nodesIndexes[l + 1][i];

			i += blockDim.x;
		}
		node += gridDim.x;
	}
}

__global__ void FillListsLenghts(int l, int *R, int *rSums, int *rPreSums,  int *LevelsSizes, int **startIndexes, int **endIndexes, int *Lenghts, int **ListsLenghts)
{
	int node = blockIdx.x * blockDim.x + threadIdx.x;
	while(node < LevelsSizes[l])
	{
		int lenght = 0;
		for (int i = startIndexes[l][node]; i < endIndexes[l][node]; ++i)
			if (Lenghts[i] > rPreSums[l] && Lenghts[i] <= rSums[l])
				++lenght;

		ListsLenghts[l][node] = lenght;

		node += gridDim.x * blockDim.x;
	}

	//TODO: Ten kernel mo¿na zrobiæ lepiej. Bloki chodz¹ po wêz³ach na danym poziomie. W¹tki zliczaj¹ sumy czêœciowe. Potem redukcja w obrêbie bloku i wpisanie wartoœci.
}

__global__ void FillListItems(int l, int *R, int *rSums, int *rPreSums, int Count, int **startIndexes, int ** endIndexes, int **ListsStarts, int *LevelsSizes, int *Lenghts, int * ListItems)
{
	int node = blockIdx.x * blockDim.x + threadIdx.x;
	while(node < LevelsSizes[l])
	{
		int insertShift = 0;
		for (int maskLenght = rSums[l]; maskLenght > rPreSums[l]; --maskLenght)
		{
			for (int i = startIndexes[l][node]; i < endIndexes[l][node]; ++i)
				if (Lenghts[i] == maskLenght)
				{
					int index = (ListsStarts[l][node]) + insertShift;
					ListItems[index] = i;
					++insertShift;
				}
		}

		node += gridDim.x * blockDim.x;
	}

	//TODO: Ten kernel trzeba zrobiæ lepiej, bo dla jednego wêz³a dzia³a tylko jednen w¹tek. Prawdopodobnie trzeba bêdzie napisaæ osobne kernele dla "ma³ych" i "du¿ych" poziomów i u¿yæ _atomicAdd na insertShift.
}

void RTreeModel::Build(IPSet set, GpuSetup setup)
{
	Count = set.Size;
	L = h_R.size();

	//Allocating memory for Rs
	GpuAssert(cudaMalloc((void**)&R, L * sizeof(int)), "Cannot allocate memory for R");
	GpuAssert(cudaMalloc((void**)&rSums, L * sizeof(int)), "Cannot allocate memory for R");
	GpuAssert(cudaMalloc((void**)&rPreSums, L * sizeof(int)), "Cannot allocate memory for R");

	GpuAssert(cudaMemcpy(R, h_R.data(), L * sizeof(int), cudaMemcpyHostToDevice), "Cannot copy R memory");
	thrust::inclusive_scan(thrust::device, R, R + L, rSums);
	thrust::exclusive_scan(thrust::device, R, R + L, rPreSums);

	//TODO: Niepotrzebne jest budowanie wêz³ów, je¿eli ¿adna maska w zakresie nie jest dostatecznie d³uga.

	//Allocationg memory for masks
	GpuAssert(cudaMalloc(reinterpret_cast<void**>(&Masks), L * sizeof(int*)), "Cannot init ip masks device memory");
	GpuAssert(cudaMalloc(reinterpret_cast<void**>(&Lenghts), Count * sizeof(int)), "Cannot init Lenght mamory");

	int** h_Masks = new int*[L];
	for (int l = 0; l < L; ++l)
		GpuAssert(cudaMalloc(reinterpret_cast<void**>(&h_Masks[l]), Count * sizeof(int)), "Cannot init ip masks device memory");
	GpuAssert(cudaMemcpy(Masks, h_Masks, L * sizeof(int*), cudaMemcpyHostToDevice), "Cannot copy Masks pointers to GPU");

	delete[] h_Masks;

	//Copying masks from IPSet and partitioning them
	CopyMasks <<< setup.Blocks, setup.Threads >>> (Count,  R, rSums,  L, Masks, Lenghts, set.d_IPData);
	GpuAssert(cudaGetLastError(), "Error while launching CopyMasks kernel");
	GpuAssert(cudaDeviceSynchronize(), "Error while running CopyMasks kernel");

	//Allocating memory for nodesBorders
	int ** nodesBorders;
	int ** nodesIndexes;
	GpuAssert(cudaMalloc(reinterpret_cast<void**>(&nodesBorders), L * sizeof(int*)), "Cannot init nodes borders device memory");
	GpuAssert(cudaMalloc(reinterpret_cast<void**>(&nodesIndexes), L * sizeof(int*)), "Cannot init nodes indexes device memory");

	int **h_nodesBorders = new int*[L];
	int **h_nodesIndexes = new int*[L];
	for(int l = 0; l < L; ++l)
	{
		GpuAssert(cudaMalloc(reinterpret_cast<void**>(&h_nodesBorders[l]), Count * sizeof(int)), "Cannot init nodes borders device memory");
		GpuAssert(cudaMalloc(reinterpret_cast<void**>(&h_nodesIndexes[l]), Count * sizeof(int)), "Cannot init nodes indexes device memory");
	}
	GpuAssert(cudaMemcpy(nodesBorders, h_nodesBorders, L * sizeof(int*), cudaMemcpyHostToDevice), "Cannot copy nodes borders device memory");
	GpuAssert(cudaMemcpy(nodesIndexes, h_nodesIndexes, L * sizeof(int*), cudaMemcpyHostToDevice), "Cannot copy nodes indexes device memory");

	//Marking first nodes on each level, setting rest of the nodesBorders memory to 0
	int mark = 1;
	for (int l = 0; l < L; ++l)
	{
		GpuAssert(cudaMemset(h_nodesBorders[l], 0, Count * sizeof(int)), "Cannot clear nodesBorders memory");
		GpuAssert(cudaMemcpy(h_nodesBorders[l], &mark, sizeof(int), cudaMemcpyHostToDevice), "Cannot mark nodes start");
	}

	//Marking nodes borders
	for(int l = 1; l < L; ++l)
	{
		MarkNodesBorders <<<setup.Blocks, setup.Threads >>>(Count, l, nodesBorders, Masks);
		GpuAssert(cudaGetLastError(), "Error while launching MarkNodesBorders kernel");
		GpuAssert(cudaDeviceSynchronize(), "Error while running MarkNodesBorders kernel");
	}

	//Counting number of nodes and indexing them on each level. Indexing is done from 1 up, since 0 means empty value
	LevelsSizes = new int[L];

	for(int l = 0; l < L; ++l)
	{
		thrust::inclusive_scan(thrust::device, h_nodesBorders[l], h_nodesBorders[l] + Count, h_nodesIndexes[l]);
		GpuAssert(cudaMemcpy(LevelsSizes + l, h_nodesIndexes[l] + Count - 1, sizeof(int), cudaMemcpyDeviceToHost), "Cannot copy level size");
		thrust::transform(thrust::device, h_nodesBorders[l], h_nodesBorders[l] + Count, h_nodesIndexes[l], h_nodesIndexes[l], thrust::multiplies<int>());
	}

	int *d_LevelSizes;
	GpuAssert(cudaMalloc((void**)&d_LevelSizes, L * sizeof(int)), "Cannot init d_LevelSizes memory");
	GpuAssert(cudaMemcpy(d_LevelSizes, LevelsSizes, L * sizeof(int), cudaMemcpyHostToDevice), "Cannot copy LevelSizes memory");

	//Filling start and end indexes of tree nodes
	int ** startIndexes;
	int ** endIndexes;
	GpuAssert(cudaMalloc(reinterpret_cast<void**>(&startIndexes), L * sizeof(int*)), "Cannot init startIndexes device memory");
	GpuAssert(cudaMalloc(reinterpret_cast<void**>(&endIndexes), L * sizeof(int*)), "Cannot init endIndexes device memory");

	int **h_startIndexes = new int*[L];
	int **h_endIndexes = new int*[L];
	for (int l = 0; l < L; ++l)
	{
		GpuAssert(cudaMalloc(reinterpret_cast<void**>(&h_startIndexes[l]), LevelsSizes[l] * sizeof(int)), "Cannot init startIndexes device memory");
		GpuAssert(cudaMalloc(reinterpret_cast<void**>(&h_endIndexes[l]), LevelsSizes[l] * sizeof(int)), "Cannot init endIndexes device memory");

		GpuAssert(cudaMemset(h_startIndexes[l], 0, sizeof(int)), "Cannot mark first startIndex");
		GpuAssert(cudaMemcpy(h_endIndexes[l] + (LevelsSizes[l] - 1), &Count, sizeof(int), cudaMemcpyHostToDevice), "Cannot mark last endIndex");
	}
	GpuAssert(cudaMemcpy(startIndexes, h_startIndexes, L * sizeof(int*), cudaMemcpyHostToDevice), "Cannot copy startIndexes device memory");
	GpuAssert(cudaMemcpy(endIndexes, h_endIndexes, L * sizeof(int*), cudaMemcpyHostToDevice), "Cannot copy endIndexes device memory");

	for (int l = 1; l < L; ++l)
	{
		FillIndexes << <setup.Blocks, setup.Threads >> > (Count, l, nodesIndexes, startIndexes, endIndexes);
		GpuAssert(cudaGetLastError(), "Error while launching FillIndexes kernel");
		GpuAssert(cudaDeviceSynchronize(), "Error while running FillIndexes kernel");
	}

	//Filling children of tree nodes
	//TODO: Childrencount mog³oby byæ L-1

	int *h_ChildrenCount = new int[L];
	for(int l = 0; l < L; ++l)
		h_ChildrenCount[l] = 2 << (h_R[l] - 1);

	GpuAssert(cudaMalloc((void**)&ChildrenCount, L * sizeof(int)), "Cannot init Children memory");
	GpuAssert(cudaMalloc((void**)&Children, (L-1) * sizeof(int*)), "Cannot init Children memory");

	GpuAssert(cudaMemcpy(ChildrenCount, h_ChildrenCount, L * sizeof(int), cudaMemcpyHostToDevice), "Cannot copy Children memory");

	h_Children = new int*[L];
	for(int l = 0; l < L-1; ++l)
		GpuAssert(cudaMalloc((void**)&h_Children[l], LevelsSizes[l] * h_ChildrenCount[l] * sizeof(int)), "Cannot init children memory");

	GpuAssert(cudaMemcpy(Children, h_Children, (L - 1) * sizeof(int*), cudaMemcpyHostToDevice), "Cannot copy Children memory");

	delete[] h_ChildrenCount;

	for (int l = 0; l < L - 1; ++l)
	{
		FillChildren << <setup.Blocks, setup.Threads >> > (l, d_LevelSizes, startIndexes, endIndexes, Children, ChildrenCount, Masks, nodesIndexes);
		GpuAssert(cudaGetLastError(), "Error while launching FillChildren kernel");
		GpuAssert(cudaDeviceSynchronize(), "Error while running FillChildren kernel");
	}

	//Building lists of items for each node
	GpuAssert(cudaMalloc((void**)&ListItems, Count * sizeof(int)), "Cannot init ListItems memory");
	GpuAssert(cudaMalloc((void**)&ListsStarts, L * sizeof(int*)), "Cannot init ListsStarts memory");
	GpuAssert(cudaMalloc((void**)&ListsLenghts, L * sizeof(int*)), "Cannot init ListsLenghts memory");

	h_ListsStarts = new int*[L];
	h_ListsLenghts = new int*[L];

	for(int l = 0; l < L; ++l)
	{
		GpuAssert(cudaMalloc((void**)&h_ListsStarts[l], LevelsSizes[l] * sizeof(int)), "Cannot init ListsStarts memory");
		GpuAssert(cudaMalloc((void**)&h_ListsLenghts[l], LevelsSizes[l] * sizeof(int)), "Cannot init ListsLenghts memory");
	}
	
	GpuAssert(cudaMemcpy(ListsStarts, h_ListsStarts, L * sizeof(int*), cudaMemcpyHostToDevice), "Cannot copy ListsStarts memory");
	GpuAssert(cudaMemcpy(ListsLenghts, h_ListsLenghts, L * sizeof(int*), cudaMemcpyHostToDevice), "Cannot copy ListsLenghts memory");

	for(int l = 0; l < L; ++l)
	{
		FillListsLenghts << <setup.Blocks, setup.Threads >> > (l, R, rSums, rPreSums, d_LevelSizes, startIndexes, endIndexes, Lenghts, ListsLenghts);
		GpuAssert(cudaGetLastError(), "Error while launching FillListsLenghts kernel");
		GpuAssert(cudaDeviceSynchronize(), "Error while running FillListsLenghts kernel");
	}

	//Filling lists start indexes
	int *totalListItemsPerLevel = new int[L];
	for(int l = 0; l < L; ++l)
	{
		thrust::exclusive_scan(thrust::device, h_ListsLenghts[l], h_ListsLenghts[l] + LevelsSizes[l], h_ListsStarts[l]);
		totalListItemsPerLevel[l] = thrust::reduce(thrust::device, h_ListsLenghts[l], h_ListsLenghts[l] + LevelsSizes[l]);
	}

	//Shifting lists
	int shift = 0;
	for (int l = 1; l < L; ++l)
	{
		shift += totalListItemsPerLevel[l - 1];
		thrust::for_each_n(thrust::device, h_ListsStarts[l], LevelsSizes[l], thrust::placeholders::_1 += shift);

		//TODO: Zerowanie startów pustych list?
	}

	//Filling list items
	for(int l = 0; l < L; ++l)
	{
		//TODO: Ten kernel jest d³ugi, bo na poziomach gdzie jest ma³o wêz³ów dzia³a ma³o w¹tków
		FillListItems << <setup.Blocks, setup.Threads >> > (l, R, rSums, rPreSums, Count, startIndexes, endIndexes, ListsStarts, d_LevelSizes, Lenghts, ListItems);
		GpuAssert(cudaGetLastError(), "Error while launching FillListItems kernel");
		GpuAssert(cudaDeviceSynchronize(), "Error while running FillListItems kernel");
	}

	//Cleanup
	for(int i = 0; i < L; ++i)
	{
		GpuAssert(cudaFree(h_nodesBorders[i]), "Cannot free nodes borders device memory.");
		GpuAssert(cudaFree(h_nodesIndexes[i]), "Cannot free nodes indexes device memory.");
	}

	GpuAssert(cudaFree(nodesBorders), "Cannot free nodes borders device memory.");
	GpuAssert(cudaFree(nodesIndexes), "Cannot free nodes indexes device memory.");

	delete[] h_nodesBorders;
	delete[] h_nodesIndexes;

	GpuAssert(cudaFree(d_LevelSizes), "Cannot free d_LevelSizes memory");

	for (int i = 0; i < L; ++i)
	{
		GpuAssert(cudaFree(h_startIndexes[i]), "Cannot free startIndexes device memory.");
		GpuAssert(cudaFree(h_endIndexes[i]), "Cannot free endIndexes device memory.");
	}

	GpuAssert(cudaFree(startIndexes), "Cannot free startIndexes device memory.");
	GpuAssert(cudaFree(endIndexes), "Cannot free endIndexes device memory.");

	delete[] h_startIndexes;
	delete[] h_endIndexes;

	delete[] totalListItemsPerLevel;
}

void RTreeModel::Dispose()
{
	if(Masks != NULL)
	{
		int** h_Masks = new int*[L];
		GpuAssert(cudaMemcpy(h_Masks, Masks, L * sizeof(int*), cudaMemcpyDeviceToHost), "Cannot copy Masks pointers to CPU");
		for (int i = 0; i < L; ++i)
			GpuAssert(cudaFree(h_Masks[i]), "Cannot free Masks memory");
		delete[] h_Masks;
		GpuAssert(cudaFree(Masks), "Cannot free Masks memory");
		Masks = NULL;
	}

	if(R != NULL)
	{
		GpuAssert(cudaFree(R), "Cannot free R memory");
		GpuAssert(cudaFree(rSums), "Cannot free rSums memory");
		GpuAssert(cudaFree(rPreSums), "Cannot free rPreSums memory");

		R = rSums = rPreSums = NULL;
	}

	if(Lenghts != NULL)
	{
		GpuAssert(cudaFree(Lenghts), "Cannot free Lenghts memory.");
		Lenghts = NULL;
	}

	if(LevelsSizes != NULL)
	{
		delete[] LevelsSizes;
		LevelsSizes = NULL;
	}

	if(Children != NULL)
	{
		for (int l = 0; l < L - 1; ++l)
			GpuAssert(cudaFree(h_Children[l]), "Cannot free Children memory");

		GpuAssert(cudaFree(Children), "Cannot free children memory");
		GpuAssert(cudaFree(ChildrenCount), "Cannot free Children memory");
		delete[] h_Children;

		Children = h_Children = NULL;
		ChildrenCount = NULL;
	}

	if(ListItems != NULL)
	{
		GpuAssert(cudaFree(ListItems), "Cannot free ListItems memory");
		ListItems = NULL;
	}

	if(ListsStarts != NULL)
	{
		for (int l = 0; l < L; ++l)
			GpuAssert(cudaFree(h_ListsStarts[l]), "Cannot free ListsStarts memory");
		GpuAssert(cudaFree(ListsStarts), "Cannot free ListsStarts memory");

		delete[] h_ListsStarts;
		ListsStarts = h_ListsStarts = NULL;
	}

	if (ListsLenghts != NULL)
	{
		for (int l = 0; l < L; ++l)
			GpuAssert(cudaFree(h_ListsLenghts[l]), "Cannot free ListsLenghts memory");
		GpuAssert(cudaFree(ListsLenghts), "Cannot free ListsLenghts memory");

		delete[] h_ListsLenghts;
		ListsLenghts = h_ListsLenghts = NULL;
	}
}

void RTreeResult::PrintResult()
{
}

int RTreeResult::CountMatched()
{
	int result = 0;

	for (int i = 0; i < IpsToMatchCount; ++i)
		if (MatchedMaskIndex[i] != -1)
			++result;
		else
			printf("%d\n", i);

	return result;
}

void RTreeMatcher::BuildModel(IPSet set)
{
	Setup = set.Setup;
	GpuAssert(cudaSetDevice(Setup.DeviceID), "Cannot set cuda device in IPSet RandomSubset.");
	Timer timer;
	timer.Start();
	Model.Build(set, Setup);
	ModelBuildTime = timer.Stop();
	GpuAssert(cudaSetDevice(0), "Cannot set cuda device in IPSet RandomSubset.");
}

__global__ void MatchIPs(int ** Children, int *ChildrenCount, int **Masks, int *result, int **ListsStarts, int **ListsLenghts, int *Lenghts, int L, int *R, int *rPreSums, int *ListItems,
	int **ips, int Count)
{
	extern __shared__ int m[];
	int *nodesToCheck = m + threadIdx.x * L;

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	while( i < Count)
	{
		//Find nodes to be searched
		nodesToCheck[0] = 1;

		for (int l = 1; l < L; ++l)
		{
			nodesToCheck[l] = 0;
			if (nodesToCheck[l - 1] != 0)
				nodesToCheck[l] = Children[l - 1][(nodesToCheck[l - 1] - 1)*ChildrenCount[l - 1] + Masks[l - 1][i]];
			else
				break;
		}

		//Search lists
		for (int l = L - 1; l >= 0 && result[i] == -1; --l)
			if (nodesToCheck[l] != 0)
			{
				for (int s = ListsStarts[l][nodesToCheck[l] - 1];
					s < ListsStarts[l][nodesToCheck[l] - 1] + ListsLenghts[l][nodesToCheck[l] - 1] && result[i] == -1;
					++s)
				{
					int shitf = R[l] - (Lenghts[ListItems[s]] - rPreSums[l]);
					if (Masks[l][ListItems[s]] >> shitf == ips[l][i] >> shitf)
						result[i] = ListItems[s];
				}
			}

		i += gridDim.x * blockDim.x;
	}
}

RTreeResult RTreeMatcher::Match(IPSet set)
{
	RTreeResult result(set.Size);
	result.MatchedMaskIndex = new int[set.Size];

	Timer timer;
	timer.Start();

	int **d_IPs;
	int *d_IPsLenghts;

	GpuAssert(cudaMalloc(reinterpret_cast<void**>(&d_IPs), Model.L * sizeof(int*)), "Cannot init ip masks device memory");
	GpuAssert(cudaMalloc(reinterpret_cast<void**>(&d_IPsLenghts), set.Size * sizeof(int)), "Cannot init Lenght mamory");

	int** h_Masks = new int*[Model.L];
	for (int l = 0; l < Model.L; ++l)
		GpuAssert(cudaMalloc(reinterpret_cast<void**>(&h_Masks[l]), Model.Count * sizeof(int)), "Cannot init ip masks device memory");
	GpuAssert(cudaMemcpy(d_IPs, h_Masks, Model.L * sizeof(int*), cudaMemcpyHostToDevice), "Cannot copy Masks pointers to GPU");

	delete[] h_Masks;

	//Copying ips from IPSet and partitioning them
	//TODO: Tutaj budowanie d_IPsLenghts jest niepotrzebne
	CopyMasks << < Setup.Blocks, Setup.Threads >> > (set.Size, Model.R, Model.rSums, Model.L, d_IPs, d_IPsLenghts, set.d_IPData);
	GpuAssert(cudaGetLastError(), "Error while launching CopyMasks kernel");
	GpuAssert(cudaDeviceSynchronize(), "Error while running CopyMasks kernel");

	int *d_Result;
	GpuAssert(cudaMalloc((void**)&d_Result, result.IpsToMatchCount * sizeof(int)), "Cannot allocate memory for Result");
	thrust::fill_n(thrust::device, d_Result, result.IpsToMatchCount, -1);


	//Matching
	MatchIPs << <Setup.Blocks, Setup.Threads, Setup.Threads * Model.L * sizeof(int)>> > (Model.Children, Model.ChildrenCount, Model.Masks, d_Result, Model.ListsStarts, Model.ListsLenghts,
		Model.Lenghts, Model.L, Model.R, Model.rPreSums, Model.ListItems, d_IPs, set.Size);
	GpuAssert(cudaGetLastError(), "Error while launching MatchIPs kernel");
	GpuAssert(cudaDeviceSynchronize(), "Error while running MatchIPs kernel");

	GpuAssert(cudaMemcpy(result.MatchedMaskIndex, d_Result, result.IpsToMatchCount * sizeof(int), cudaMemcpyDeviceToHost), "Cannot copy Result data");

	GpuAssert(cudaFree(d_Result), "Cannot free Result memory");
	GpuAssert(cudaFree(d_IPs), "Cannot free d_IPs memory");
	GpuAssert(cudaFree(d_IPsLenghts), "Cannot free d_IPsLenghts memory");

	result.MatchingTime = timer.Stop();

	return result;
}
