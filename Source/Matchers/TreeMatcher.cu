#include "TreeMatcher.cuh"
#include <ctime>
#include <device_launch_parameters.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

template<typename T>
struct vector_less
{
	/*! \typedef first_argument_type
	*  \brief The type of the function object's first argument.
	*/
	typedef T first_argument_type;

	/*! \typedef second_argument_type
	*  \brief The type of the function object's second argument.
	*/
	typedef T second_argument_type;

	/*! \typedef result_type
	*  \brief The type of the function object's result;
	*/
	typedef bool result_type;

	/*! Function call operator. The return value is <tt>lhs >= rhs</tt>.
	*/
	__host__ __device__ bool operator()(const T &lhs, const T &rhs) const {
		if (lhs[4] == rhs[4])
			if (lhs[0] == rhs[0])
				if (lhs[1] == rhs[1])
					if (lhs[2] == rhs[2])
						return lhs[3] < rhs[3];
					else
						return lhs[2] < rhs[2];
				else
					return lhs[1] < rhs[1];
			else
				return lhs[0] < rhs[0];
		else
			return lhs[4] < rhs[4];
	}
}; // end vector less

struct my_sort_functor
{
	int cols;
	unsigned char *my_list;
	my_sort_functor(int _col, unsigned char * _lst) : cols(_col), my_list(_lst) {};

	__host__ __device__
		bool operator()(const int idx1, const int idx2) const
	{
		bool flip = false;
		if (my_list[(idx1*cols) + 4] == my_list[(idx2*cols) + 4])
		{
			for (int col_idx = 0; col_idx < cols - 1; col_idx++) {
				unsigned char d1 = my_list[(idx1*cols) + col_idx];
				unsigned char d2 = my_list[(idx2*cols) + col_idx];
				if (d1 > d2) break;
				if (d1 < d2) { flip = true; break; }
			}
			return flip;
		}
		else
			return my_list[(idx1*cols) + 4] < my_list[(idx2*cols) + 4];
	}
};

__global__ void initialize_tree_node_1(char * dev_temp_flag, unsigned int * dev_temp_Indx, int *** dev_tree, int level, int totalNodes)
{
	unsigned int thread = threadIdx.x + blockIdx.x * blockDim.x;
	if (thread >= totalNodes)
		return;
	if (thread == 0)
	{
		if (dev_temp_flag[0] > 0)
		{
			dev_tree[level][0] = new int[5];
			dev_tree[level][0][0] = thread / 2;
			dev_tree[level][0][1] = thread % 2;
			dev_tree[level][0][2] = -1;
			dev_tree[level][0][3] = -1;
			dev_tree[level][0][4] = dev_temp_Indx[thread];
			dev_tree[level - 1][thread / 2][2 + (thread % 2)] = 0;
		}
	}
	else if (dev_temp_flag[thread] > dev_temp_flag[thread - 1])
	{
		dev_tree[level][dev_temp_flag[thread] - 1] = new int[5];
		dev_tree[level][dev_temp_flag[thread] - 1][0] = thread / 2;
		dev_tree[level][dev_temp_flag[thread] - 1][1] = thread % 2;
		dev_tree[level][dev_temp_flag[thread] - 1][2] = -1;
		dev_tree[level][dev_temp_flag[thread] - 1][3] = -1;
		dev_tree[level][dev_temp_flag[thread] - 1][4] = dev_temp_Indx[thread];
		dev_tree[level - 1][thread / 2][2 + (thread % 2)] = dev_temp_flag[thread] - 1;

	}
}
__global__ void print_ranges(int * dev_prefixes_start, int * dev_prefixes_end)
{
	for (int i = 0; i < 32; i++)
	{
		printf("prefix : %d --> Start: %d   --- End : %d\n", i + 1, dev_prefixes_start[i], dev_prefixes_end[i]);
	}

}
__global__ void PrepareSubnetsIndxes(unsigned int * dev_subnets_indx, int size)
{
	int thread = threadIdx.x + blockIdx.x * blockDim.x;
	if (thread < size)
		dev_subnets_indx[thread] = thread;
}
__global__ void print2dlist(unsigned char * dev_list, unsigned int * dev_subnets_indx, int col, int count)
{
	for (int i = 0; i < count; i++)
	{
		printf("%d.%d.%d.%d/%d\n", dev_list[dev_subnets_indx[i] * col], dev_list[dev_subnets_indx[i] * col + 1], dev_list[dev_subnets_indx[i] * col + 2], dev_list[dev_subnets_indx[i] * col + 3], dev_list[dev_subnets_indx[i] * col + 4]);
	}
}

__global__ void PrepareLevels(int * dev_level8, int * dev_level16, int * dev_level24)
{
	int thread = threadIdx.x + blockIdx.x * blockDim.x;
	if (thread < 256 * 256 * 256)
		dev_level24[thread] = -1;
	if (thread < 256 * 256)
		dev_level16[thread] = -1;
	if (thread < 256)
		dev_level8[thread] = -1;

}

__global__ void print2dlist(unsigned char * dev_list, int col, int count)
{
	for (int i = 0; i < count; i++)
	{
		printf("%d.%d.%d.%d/%d\n", dev_list[i * col], dev_list[i * col + 1], dev_list[i * col + 2], dev_list[i * col + 3], dev_list[i * col + 4]);

	}
}
__global__ void print_subnets_bits(unsigned char * dev_subnets, unsigned int  * dev_subnets_indx, char* dev_sorted_subnets_bits, int size)
{
	for (int i = 0; i < size; i++)
	{
		printf("%d -> ", i);
		printf("%d.%d.%d.%d/%d  -> ", dev_subnets[dev_subnets_indx[i] * 5], dev_subnets[dev_subnets_indx[i] * 5 + 1], dev_subnets[dev_subnets_indx
			[i] * 5 + 2], dev_subnets[dev_subnets_indx[i] * 5 + 3], dev_subnets[dev_subnets_indx[i] * 5 + 4]);

		for (int j = 0; j < 33; j++)
		{
			printf("%d", dev_sorted_subnets_bits[i * 33 + j]);
		}
		printf("\n");

	}

}
__global__ void InitializeTreeNode(unsigned int * dev_temp_flag, int * dev_temp_Indx, int ** dev_tree, int * dev_real_values, int * dev_level, int level
	, int totalNodes)
{
	unsigned int thread = threadIdx.x + blockIdx.x * blockDim.x;
	if (thread < totalNodes)
	{
		if (thread == 0)
		{
			if (dev_temp_flag[0] > 0)
			{
				dev_tree[level][0] = thread / 2;
				dev_tree[level][1] = thread % 2;
				dev_tree[level][2] = -1;
				dev_tree[level][3] = -1;
				dev_tree[level][4] = dev_temp_Indx[thread];
				if (level == 8 || level == 16 || level == 24)
				{
					int loc = dev_real_values[0];
					if (loc != -1)
						dev_level[loc] = 0;
				}

				if (level > 0)
				{
					dev_tree[level - 1][2] = 0;
				}
			}
		}
		else if (dev_temp_flag[thread] > dev_temp_flag[thread - 1])
		{
			dev_tree[level][5 * (dev_temp_flag[thread] - 1)] = thread / 2;
			dev_tree[level][5 * (dev_temp_flag[thread] - 1) + 1] = thread % 2;
			dev_tree[level][5 * (dev_temp_flag[thread] - 1) + 2] = -1;
			dev_tree[level][5 * (dev_temp_flag[thread] - 1) + 3] = -1;
			dev_tree[level][5 * (dev_temp_flag[thread] - 1) + 4] = dev_temp_Indx[thread];
			if (level == 8 || level == 16 || level == 24)
			{
				int loc = dev_real_values[thread];
				if (loc != -1)
					dev_level[loc] = (dev_temp_flag[thread] - 1);

				//if (level == 24) printf("VAL: %d \n", dev_level[13551064]);
			}
			if (level > 0)
			{
				dev_tree[level - 1][(5 * (thread / 2)) + 2 + thread % 2] = dev_temp_flag[thread] - 1;
			}
		}
	}
}
__global__ void InitializeTreeLevel(int ** dev_tree, int level, int count, int * dev_level_nodes)
{
	dev_tree[level] = new  int[count * 5];
	dev_level_nodes[level] = count;
}
__global__ void initialize_tree_level_16(int ** dev_tree, int level, int count, int * dev_level_nodes)
{
	dev_tree[level] = new  int[count * 5];
	dev_level_nodes[level] = count;
	for (int i = 0; i < count * 5; i++)
		dev_tree[level][i] = -1;
}

__global__ void GetPrefixesRanges(char* dev_sorted_subnets_bits, int * dev_prefixes_start, int * dev_prefixes_end, int size)
{
	unsigned int thread = threadIdx.x + blockIdx.x * blockDim.x;
	if (thread >= size)
		return;
	if (thread == 0)
	{
		dev_prefixes_start[dev_sorted_subnets_bits[32] - 1] = 0;
		if (dev_sorted_subnets_bits[32] < dev_sorted_subnets_bits[65])
			dev_prefixes_end[dev_sorted_subnets_bits[32] - 1] = 0;
		return;
	}
	if (thread == size - 1)
	{
		dev_prefixes_end[dev_sorted_subnets_bits[thread * 33 + 32] - 1] = thread;
		if (dev_sorted_subnets_bits[thread * 33 + 32] > dev_sorted_subnets_bits[(thread - 1) * 33 + 32])
			dev_prefixes_start[dev_sorted_subnets_bits[thread * 33 + 32] - 1] = thread;

		return;
	}

	if (dev_sorted_subnets_bits[thread * 33 + 32] > dev_sorted_subnets_bits[(thread - 1) * 33 + 32])
		dev_prefixes_start[dev_sorted_subnets_bits[thread * 33 + 32] - 1] = thread;


	if (dev_sorted_subnets_bits[thread * 33 + 32] < dev_sorted_subnets_bits[(thread + 1) * 33 + 32])
		dev_prefixes_end[dev_sorted_subnets_bits[thread * 33 + 32] - 1] = thread;
}

__global__ void PrepareSortedSubnetsBitsList(unsigned char * dev_subnets, char * dev_sorted_subnets_bits, unsigned int * dev_subnets_indx, int size)
{
	int thread = threadIdx.x + blockIdx.x * blockDim.x;
	if (thread < size)
	{
		unsigned char maskValues[8] = { 128, 64, 32, 16, 8, 4, 2, 1 };
		unsigned char curr_byte;
		unsigned char res;

		for (int j = 0; j < 4; j++)
		{
			curr_byte = dev_subnets[dev_subnets_indx[thread] * 5 + j];
			for (int i = 0; i < 8; i++)
			{
				res = curr_byte & maskValues[i];
				if (res == maskValues[i])
					dev_sorted_subnets_bits[thread * 33 + j * 8 + i] = 1;
				else
					dev_sorted_subnets_bits[thread * 33 + j * 8 + i] = 0;
			}
		}
		dev_sorted_subnets_bits[thread * 33 + 32] = dev_subnets[dev_subnets_indx[thread] * 5 + 4];
	}
}

__global__ void InitializeTreeRoot(int ** dev_tree, int * dev_level_nodes)
{
	dev_tree[0] = new int[5];
	dev_tree[0][0] = -1;
	dev_tree[0][1] = -1;
	dev_tree[0][2] = -1;
	dev_tree[0][3] = -1;
	dev_tree[0][4] = -1;
	dev_level_nodes[0] = 1;
}

__device__ int isEqual(char* leftOper, char* rightOper, int bits_to_compare)
{
	/*
	return:
	0 if equal
	-1 if the left less than the right
	+1 if the left bigger than the right
	*/
	for (int i = 0; i < bits_to_compare; i++)
	{
		if (leftOper[i] < rightOper[i])
			return -1;
		if (leftOper[i] > rightOper[i])
			return 1;
	}
	return 0;
}

__device__ int find_node_prefix(char* node_prefix, char * dev_sorted_subnets_bits, unsigned int * dev_temp_flag, int * dev_temp_Indx, int pos, char bits_to_compare, int start, int end)
{
	int res;
	int curr_indx = 1;
	while (start <= end)
	{
		curr_indx = (start + end) / 2;
		res = isEqual(node_prefix, dev_sorted_subnets_bits + curr_indx * 33, bits_to_compare);
		if (res == 0)
		{
			dev_temp_flag[pos] = 1;
			if (bits_to_compare == dev_sorted_subnets_bits[curr_indx * 33 + 32])
				dev_temp_Indx[pos] = curr_indx;
			return curr_indx;
		}
		else if (res > 0)
		{
			start = curr_indx + 1;
		}
		else
		{
			end = curr_indx - 1;
		}
	}

	return -2;

}


__global__ void FlagBitLocation(unsigned int * dev_temp_flag, int * dev_temp_Indx, char * dev_sorted_subnets_bits, int ** dev_tree, int * dev_prefixes_start, int * dev_prefixes_end, char * currentNodePrefix, int * dev_real_values, int level_max, int level, int subnets_size, int totalelements)
{
	int thread = threadIdx.x + blockIdx.x * blockDim.x;
	if (thread < totalelements)
	{
		currentNodePrefix[thread * level + level - 1] = thread % 2;
		int thread_pos = thread;
		int node_trav = thread / 2;

		int level_trav = level - 1;

		int real_value = thread % 2;
		int bit;
		while (level_trav > 0)
		{
			bit = dev_tree[level_trav][5 * node_trav + 1];
			currentNodePrefix[thread * level + level_trav - 1] = bit;
			node_trav = dev_tree[level_trav][5 * node_trav];
			if (level_max != -1)
			{
				int yy = 1 << (level_max - level_trav);
				real_value += bit * yy;

			}


			level_trav--;
		}

		dev_temp_flag[thread_pos] = 0;
		dev_temp_Indx[thread_pos] = -1;
		int indx = -2;
		for (int interationCounter = level - 1; interationCounter < 32; interationCounter++)
		{
			if (dev_prefixes_start[interationCounter] != -1)
			{
				indx = find_node_prefix(currentNodePrefix + thread * level, dev_sorted_subnets_bits, dev_temp_flag, dev_temp_Indx, thread_pos, level, dev_prefixes_start[interationCounter], dev_prefixes_end[interationCounter]);
				if (indx != -2)
				{
					if (level_max != -1) {
						dev_real_values[thread_pos] = real_value;
					}

					break;
				}
			}
		}
		if (indx == -2 && level_max != -1)
			dev_real_values[thread_pos] = -1;

	}

}

__device__ char *  convert_to_bits(unsigned char b1, unsigned char b2, unsigned char b3, unsigned char b4)
{
	unsigned char maskValues[8] = { 128, 64, 32, 16, 8, 4, 2, 1 };
	char * dev_out_bits = new char[32];
	unsigned char res;

	for (int i = 0; i < 8; i++)
	{
		res = b1 & maskValues[i];
		if (res == maskValues[i])
			dev_out_bits[i] = 1;
		else
			dev_out_bits[i] = 0;
	}
	for (int i = 0; i < 8; i++)
	{
		res = b2 & maskValues[i];
		if (res == maskValues[i])
			dev_out_bits[8 + i] = 1;
		else
			dev_out_bits[8 + i] = 0;
	}
	for (int i = 0; i < 8; i++)
	{
		res = b3 & maskValues[i];
		if (res == maskValues[i])
			dev_out_bits[16 + i] = 1;
		else
			dev_out_bits[16 + i] = 0;
	}
	for (int i = 0; i < 8; i++)
	{
		res = b4 & maskValues[i];
		if (res == maskValues[i])
			dev_out_bits[24 + i] = 1;
		else
			dev_out_bits[24 + i] = 0;
	}
	return dev_out_bits;


}

__global__ void prepare_ips_bits_list(unsigned char * dev_ip_list, char * dev_ips_bits, int searchedIpsSize)
{
	int thread = threadIdx.x + blockIdx.x * blockDim.x;
	if (thread < searchedIpsSize)
	{
		unsigned char maskValues[8] = { 128, 64, 32, 16, 8, 4, 2, 1 };
		unsigned char curr_byte;
		unsigned char res;
		for (int j = 0; j < 4; j++)
		{
			curr_byte = dev_ip_list[thread * 4 + j];
			for (int i = 0; i < 8; i++)
			{
				res = curr_byte & maskValues[i];
				if (res == maskValues[i])
					dev_ips_bits[thread * 32 + j * 8 + i] = 1;
				else
					dev_ips_bits[thread * 32 + j * 8 + i] = 0;
			}
		}
	}
}

__device__ int search_between_two_levels(int low_level, int top_level, unsigned int ip, int trav, int ** dev_tree, int max, int * continue_search)
{
	if (top_level == 32) top_level++;
	int cur = dev_tree[low_level][trav * 5 + 4];
	if (cur > -1)
		max = cur;
	trav = dev_tree[low_level][trav * 5 + 2 + ((ip & (1 << (31 - low_level))) >> (31 - low_level))];
	low_level++;
	while (trav > -1 && low_level < top_level)
	{
		cur = dev_tree[low_level][trav * 5 + 4];
		if (cur > -1)
			max = cur;

		trav = dev_tree[low_level][trav * 5 + 2 + ((ip & (1 << (31 - low_level))) >> (31 - low_level))];
		low_level++;
	}

	if (low_level == top_level)
		*continue_search = 1;
	else
		*continue_search = 0;
	return max;


}
__global__ void AssignIPSubnetWithMidLevels(int ** dev_tree, unsigned int * dev_ips_list, int * dev_matched_ip_subnet, int * dev_level8, int * dev_level16,
	int * dev_level24, int searchedIpsSize, char min_prefx, unsigned int  * time_thread_start, unsigned int  * time_thread_end, unsigned int  * time_thread)
{
	__shared__ int * shared_dev_tree[33];
	unsigned int thread = threadIdx.x + blockIdx.x * blockDim.x;
	if (thread < searchedIpsSize)
	{
		if (threadIdx.x < 33)
		{
			shared_dev_tree[threadIdx.x] = dev_tree[threadIdx.x];
			//printf("thread: %d  -> clock: %f \n", thread, CLOCKS_PER_SECOND);
		}
		__syncthreads();

		clock_t start = clock();

		unsigned int ip = dev_ips_list[thread];
		int max = -1;
		int cur;
		char level;
		int trav;
		if (min_prefx <= 24)
		{
			trav = dev_level24[(ip & 4294967040) >> 8];
			if (trav != -1)
			{
				level = 24;
				cur = shared_dev_tree[level][trav * 5 + 4];
				if (cur > -1)
					max = cur;
				trav = shared_dev_tree[level][trav * 5 + 2 + ((ip & (1 << (31 - level))) >> (31 - level))];
				level++;
				while (trav > -1)
				{
					cur = shared_dev_tree[level][trav * 5 + 4];
					if (cur > -1)
						max = cur;

					trav = shared_dev_tree[level][trav * 5 + 2 + ((ip & (1 << (31 - level))) >> (31 - level))];
					level++;
				}
				if (cur != -1)
				{
					dev_matched_ip_subnet[thread] = max;

					clock_t stop = clock();
					time_thread[thread] = (unsigned int)(stop - start);
					time_thread_end[thread] = (unsigned int)stop;
					time_thread_start[thread] = (unsigned int)start;
					//printf("IPS Assigning time: %f ms\n", a_elapsed);
					return;
				}
			}
		}
		if (min_prefx <= 16)
		{
			trav = dev_level16[(ip & 4294901760) >> 16];
			if (trav != -1)
			{
				level = 16;
				cur = shared_dev_tree[level][trav * 5 + 4];
				if (cur > -1)
					max = cur;

				trav = shared_dev_tree[level][trav * 5 + 2 + ((ip & (1 << (31 - level))) >> (31 - level))];

				level++;
				while (trav > -1 && level < 24)
				{
					cur = shared_dev_tree[level][trav * 5 + 4];
					if (cur > -1)
						max = cur;

					trav = shared_dev_tree[level][trav * 5 + 2 + ((ip & (1 << (31 - level))) >> (31 - level))];
					level++;
				}
				if (cur != -1)
				{
					dev_matched_ip_subnet[thread] = max;
					clock_t stop = clock();
					time_thread[thread] = (unsigned int)(stop - start);
					time_thread_end[thread] = (unsigned int)stop;
					time_thread_start[thread] = (unsigned int)start;

					return;
				}
			}
		}
		if (min_prefx <= 8)
		{
			trav = dev_level8[(ip & 4278190080) >> 24];
			if (trav != -1)
			{
				level = 8;
				cur = shared_dev_tree[level][trav * 5 + 4];
				if (cur > -1)
					max = cur;
				trav = shared_dev_tree[level][trav * 5 + 2 + ((ip & (1 << (31 - level))) >> (31 - level))];
				level++;
				while (trav > -1 && level < 16)
				{
					cur = shared_dev_tree[level][trav * 5 + 4];
					if (cur > -1)
						max = cur;

					trav = shared_dev_tree[level][trav * 5 + 2 + ((ip & (1 << (31 - level))) >> (31 - level))];
					level++;
				}
				if (cur != -1)
				{
					dev_matched_ip_subnet[thread] = max;
					clock_t stop = clock();
					time_thread[thread] = (unsigned int)(stop - start);
					time_thread_end[thread] = (unsigned int)stop;
					time_thread_start[thread] = (unsigned int)start;

					return;
				}
			}

			if (min_prefx < 8)
			{
				int level = 0;
				int cur;
				trav = dev_tree[0][2 + ((ip & (1 << 31)) >> 31)];
				level++;
				while (trav > -1 && level < 8)
				{
					if (level >= min_prefx)
					{
						cur = dev_tree[level][trav * 5 + 4];
						if (cur > -1)
							max = cur;
					}
					trav = dev_tree[level][trav * 5 + 2 + ((ip & (1 << (31 - level))) >> (31 - level))];
					level++;
				}
			}
			dev_matched_ip_subnet[thread] = max;
			clock_t stop = clock();
			time_thread[thread] = (unsigned int)(stop - start);
			time_thread_end[thread] = (unsigned int)stop;
			time_thread_start[thread] = (unsigned int)start;

			return;
		}
		clock_t stop = clock();
		time_thread[thread] = (unsigned int)(stop - start);
		time_thread_end[thread] = (unsigned int)stop;
		time_thread_start[thread] = (unsigned int)start;

	}

}

__global__ void AssignIPSubnet(int ** dev_tree, unsigned int * dev_ips_list, int * dev_matched_ip_subnet, int searchedIpsSize, char min_prefx, unsigned int  * time_thread_start, unsigned int  * time_thread_end, unsigned int  * time_thread)
{
	__shared__ int * shared_dev_tree[33];
	unsigned int thread = threadIdx.x + blockIdx.x * blockDim.x;
	if (thread < searchedIpsSize)
	{
		if (threadIdx.x < 33)
		{
			shared_dev_tree[threadIdx.x] = dev_tree[threadIdx.x];
			//printf("thread: %d  -> clock: %f \n", thread, CLOCKS_PER_SECOND);
		}
		__syncthreads();

		clock_t start = clock();

		unsigned int ip = dev_ips_list[thread];
		int max = -1;
		int cur;
		char level = 0;
		int trav;
		trav = dev_tree[0][2 + ((ip & (1 << 31)) >> 31)];
		level++;
		while (trav > -1)
		{
			if (level >= min_prefx)
			{
				cur = dev_tree[level][trav * 5 + 4];
				if (cur > -1)
					max = cur;
			}
			trav = dev_tree[level][trav * 5 + 2 + ((ip & (1 << (31 - level))) >> (31 - level))];
			level++;
		}

		dev_matched_ip_subnet[thread] = max;
		clock_t stop = clock();
		time_thread[thread] = (unsigned int)(stop - start);
		time_thread_end[thread] = (unsigned int)stop;
		time_thread_start[thread] = (unsigned int)start;
	}
}




__global__ void print_tree(int ** dev_tree, int * levels_count, int end = 33, int start = 0)
{

	for (int i = start; i < end; i++)
	{
		printf("%d  ->  ", i);
		for (int j = 0; j < levels_count[i]; j++)
		{
			if (dev_tree[i][j * 5] != -1)
				printf("(%d : %d*%d*%d*%d*%d)  -  ", j, dev_tree[i][j * 5], dev_tree[i][j * 5 + 1], dev_tree[i][j * 5 + 2], dev_tree[i][j * 5 + 3], dev_tree[i][j * 5 + 4]);
		}
		printf("\n----------------------------------------------------------------------------\n");
	}
}

__global__ void print_level(int * dev_level, int size)
{
	for (int i = 0; i < size; i++)
		if (dev_level[i] != -1)
			printf("indx : %d exist %d\n", i, dev_level[i]);
}


void TreeResult::ConvertFromBits(char * inBits, unsigned char * outByte)
{
	unsigned char maskValues[8] = { 128, 64, 32, 16, 8, 4, 2, 1 };
	for (int j = 0; j < 4; j++)
	{
		outByte[j] = 0;
		for (int i = 0; i < 8; i++)
		{
			if (inBits[j * 8 + i] == 1)
				outByte[j] += maskValues[i];
		}
	}
}

int TreeResult::CountMatched()
{
	int unmatched = 0;
	for (int i = 0; i < IPsListSize; ++i)
		if (MatchedIndexes[i] < 0)
			unmatched++;

	return IPsListSize - unmatched;
}

void TreeResult::PrintResult()
{
	if(MatchedIndexes == NULL || SortedSubnetsBits == NULL || IPsList == NULL)
		return;

	for (int i = 0; i < IPsListSize; i++)
	{
		if (MatchedIndexes[i] > -1)
		{
			if (i % PrintJump != 0)
				continue;

			unsigned char * submit = new unsigned char[5];
			ConvertFromBits(SortedSubnetsBits + MatchedIndexes[i] * 33, submit);
			submit[4] = SortedSubnetsBits[MatchedIndexes[i] * 33 + 32];
			printf("ip: %d.%d.%d.%d --> subnet: %d.%d.%d.%d/%d\n", ((IPsList[i] & 4278190080) >> 24), ((IPsList[i] & 16711680) >> 16), ((IPsList[i] & 65280) >> 8), (IPsList[i] & 255), submit[0], submit[1], submit[2], submit[3], submit[4]);
		}
		else
			printf("IP_ID : %d -> ip: %d.%d.%d.%d ->  sIndx :  %d\n", i, ((IPsList[i] & 4278190080) >> 24), ((IPsList[i] & 16711680) >> 16), ((IPsList[i] & 65280) >> 8), (IPsList[i] & 255), MatchedIndexes[i]);
	}
}

void TreeMatcher::BuildModel(IPSet set)
{
	Setup = set.Setup;
	GpuAssert(cudaSetDevice(Setup.DeviceID), "Cannot set cuda device.");
	Timer timer;
	timer.Start();

	Tree.Setup = Setup;
	Tree.Size = set.Size;
	int chunks = set.Size / 50000 + 1;

	// Host Variables:
	char * sortedSubnetsBits = nullptr;
	int * matchedIndexes = nullptr;
	int * minus_one;
	int * prefixesStart;
	int * prefixesEnd;

	// Decvice Variables
	thrust::device_ptr<unsigned int> dev_subnets_indx_ptr;
	char * d_NodePrefix;
	unsigned int * d_TempFlag;
	int * d_TempIndx;
	int * d_RealValues;
	int * d_FullLevel;

	// Threads Variables
	int resetThreads;
	int resetBlocks;
	int threads;
	int blocks;

	GpuAssert(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 8000000 * chunks), "Cannot set cuda limit malloc heap size.");

	// Allocate GPU buffers for subnetmasks vector.
	GpuAssert(cudaMalloc((void**)&Tree.d_SortedSubnetBits, 33 * set.Size * sizeof(char)), "Cannot allocate memory for d_SortedSubnetBits");
	GpuAssert(cudaMalloc((void**)&Tree.d_SubnetsIndx, set.Size * sizeof(unsigned int)), "Cannot allocate memory for d_SubnetsIndx");
	GpuAssert(cudaMalloc((void**)&Tree.d_Tree, 33 * sizeof(int *)), "Cannot allocate memory for d_Tree");
	GpuAssert(cudaMalloc((void**)&Tree.d_PrefixesStart, 32 * sizeof(int)), "Cannot allocate memory for d_PrefixesStart");
	GpuAssert(cudaMalloc((void**)&Tree.d_PrefixesEnd, 32 * sizeof(int)), "Cannot allocate memory for d_PrefixesEnd");

	prefixesStart = new int[32];
	prefixesEnd = new int[32];

	minus_one = new int[33];
	for (int i = 0; i < 33; i++)
		minus_one[i] = -1;

	GpuAssert(cudaMemcpy(Tree.d_PrefixesStart, minus_one, 32 * sizeof(int), cudaMemcpyHostToDevice), "Cannot copy memory to d_PrefixesStart");
	GpuAssert(cudaMemcpy(Tree.d_PrefixesEnd, minus_one, 32 * sizeof(int), cudaMemcpyHostToDevice), "Cannot copy memory to d_PrefixesEnd");

	// prepare levels
	GpuAssert(cudaMalloc((void**)&Tree.d_Level16, 256 * 256 * sizeof(int)), "Cannot allocate memory for d_Level16");
	GpuAssert(cudaMalloc((void**)&Tree.d_Level24, 256 * 256 * 256 * sizeof(int)), "Cannot allocate memory for d_Level24");
	GpuAssert(cudaMalloc((void**)&Tree.d_Level8, 256 * sizeof(int)), "Cannot allocate memory for d_Level8");

	/// prepare Subnets Indexes
	resetThreads = Setup.Threads;
	resetBlocks = 16777216 / resetThreads;
	if (16777216 % resetThreads > 0)
		resetBlocks++;

	PrepareLevels << <resetBlocks, resetThreads >> >(Tree.d_Level8, Tree.d_Level16, Tree.d_Level24);
	GpuAssert(cudaGetLastError(), "Error while launching PrepareLevels kernel");
	GpuAssert(cudaDeviceSynchronize(), "Error while running PrepareLevels kernel");

	resetThreads = Setup.Threads;
	resetBlocks = 1;
	if (set.Size > Setup.Threads)
	{
		resetThreads = Setup.Threads;
		resetBlocks = set.Size / Setup.Threads;
		if (set.Size % Setup.Threads > 0)
			resetBlocks++;
	}

	PrepareSubnetsIndxes << <resetBlocks, resetThreads >> >(Tree.d_SubnetsIndx, set.Size);
	GpuAssert(cudaGetLastError(), "Error while launching PrepareSubnetsIndxes kernel");
	GpuAssert(cudaDeviceSynchronize(), "Error while running PrepareSubnetsIndxes kernel");


	// Sorting Subnets
	dev_subnets_indx_ptr = thrust::device_pointer_cast(Tree.d_SubnetsIndx);
	thrust::sort(dev_subnets_indx_ptr, dev_subnets_indx_ptr + set.Size, my_sort_functor(5, thrust::raw_pointer_cast(set.d_IPData)));

	PrepareSortedSubnetsBitsList << <resetBlocks, resetThreads >> >(set.d_IPData, Tree.d_SortedSubnetBits, Tree.d_SubnetsIndx, set.Size);
	GpuAssert(cudaGetLastError(), "Error while launching PrepareSortedSubnetsBitsList kernel");
	GpuAssert(cudaDeviceSynchronize(), "Error while running PrepareSortedSubnetsBitsList kernel");

	GetPrefixesRanges << <resetBlocks, resetThreads >> >(Tree.d_SortedSubnetBits, Tree.d_PrefixesStart, Tree.d_PrefixesEnd, set.Size);
	GpuAssert(cudaGetLastError(), "Error while launching GetPrefixesRanges kernel");
	GpuAssert(cudaDeviceSynchronize(), "Error while running GetPrefixesRanges kernel");

	GpuAssert(cudaMemcpy(prefixesStart, Tree.d_PrefixesStart, 32 * sizeof(int), cudaMemcpyDeviceToHost), "Cannot copy memory to prefixesStart");
	GpuAssert(cudaMemcpy(prefixesEnd, Tree.d_PrefixesEnd, 32 * sizeof(int), cudaMemcpyDeviceToHost), "Cannot copy memory to prefixesEnd");

	for (int i = 0; i < 32; i++)
		if (prefixesStart[i] != -1)
		{
			Tree.MinPrefix = i + 1;
			break;
		}

	int previousLevelChildren;
	previousLevelChildren = 2;

	GpuAssert(cudaMalloc((void**)&Tree.d_LevelNodes, 33 * sizeof(int)), "Cannot allocate memory for d_LevelNodes");
	GpuAssert(cudaMemcpy(Tree.d_LevelNodes, minus_one, 33 * sizeof(int), cudaMemcpyHostToDevice), "Cannot copy memory to d_LevelNodes");

	InitializeTreeRoot << <1, 1 >> >(Tree.d_Tree, Tree.d_LevelNodes);
	GpuAssert(cudaGetLastError(), "Error while launching InitializeTreeRoot kernel");
	GpuAssert(cudaDeviceSynchronize(), "Error while running InitializeTreeRoot kernel");

	for (int level = 1; level < 33; level++)
	{
		d_RealValues = NULL;
		GpuAssert(cudaMalloc((void**)&d_TempFlag, previousLevelChildren * sizeof(unsigned int)), "Cannot allocate memory for d_TemFlag");
		GpuAssert(cudaMalloc((void**)&d_TempIndx, previousLevelChildren * sizeof(int)), "Cannot allocate memory for d_TempIndx");
		GpuAssert(cudaMalloc((void**)&d_NodePrefix, level * previousLevelChildren * sizeof(char)), "Cannot allocate memory for d_NodePrefix");

		if (level == 8 || level == 16 || level == 24)
			GpuAssert(cudaMalloc((void**)&d_RealValues, previousLevelChildren * sizeof(int)), "Cannot allocate memory for d_RealValues");

		threads = Setup.Threads;
		blocks = 1;
		if (previousLevelChildren > Setup.Threads)
		{
			threads = Setup.Threads;
			blocks = previousLevelChildren / Setup.Threads;
			if (previousLevelChildren % Setup.Threads)
				blocks++;
		}

		if (level == 8)
			FlagBitLocation << <blocks, threads >> >(d_TempFlag, d_TempIndx, Tree.d_SortedSubnetBits, Tree.d_Tree, Tree.d_PrefixesStart, Tree.d_PrefixesEnd, d_NodePrefix, d_RealValues, 8, level, set.Size, previousLevelChildren);
		else if (level == 16)
			FlagBitLocation << <blocks, threads >> >(d_TempFlag, d_TempIndx, Tree.d_SortedSubnetBits, Tree.d_Tree, Tree.d_PrefixesStart, Tree.d_PrefixesEnd, d_NodePrefix, d_RealValues, 16, level, set.Size, previousLevelChildren);
		else if (level == 24)
			FlagBitLocation << <blocks, threads >> >(d_TempFlag, d_TempIndx, Tree.d_SortedSubnetBits, Tree.d_Tree, Tree.d_PrefixesStart, Tree.d_PrefixesEnd, d_NodePrefix, d_RealValues, 24, level, set.Size, previousLevelChildren);
		else
			FlagBitLocation << <blocks, threads >> >(d_TempFlag, d_TempIndx, Tree.d_SortedSubnetBits, Tree.d_Tree, Tree.d_PrefixesStart, Tree.d_PrefixesEnd, d_NodePrefix, NULL, -1, level, set.Size, previousLevelChildren);
		GpuAssert(cudaGetLastError(), "Error while launching FlagBitLocation kernel");
		GpuAssert(cudaDeviceSynchronize(), "Error while running FlagBitLocation kernel");

		thrust::device_ptr< unsigned int> d_TempFlagPtr(d_TempFlag);
		thrust::inclusive_scan(d_TempFlagPtr, d_TempFlagPtr + previousLevelChildren, d_TempFlagPtr);

		int total;
		GpuAssert(cudaMemcpy(&total, d_TempFlag + previousLevelChildren - 1, sizeof(int), cudaMemcpyDeviceToHost), "Cannot copy memory to total");
		if (total == 0)
			break;

		InitializeTreeLevel << <1, 1 >> >(Tree.d_Tree, level, total, Tree.d_LevelNodes);
		GpuAssert(cudaGetLastError(), "Error while launching InitializeTreeLevel kernel");
		GpuAssert(cudaDeviceSynchronize(), "Error while running InitializeTreeLevel kernel");

		if (level == 8)
			d_FullLevel = Tree.d_Level8;
		else if (level == 16)
			d_FullLevel = Tree.d_Level16;
		else
			d_FullLevel = Tree.d_Level24;

		InitializeTreeNode << <blocks, threads >> >(d_TempFlag, d_TempIndx, Tree.d_Tree, d_RealValues, d_FullLevel, level, previousLevelChildren);
		GpuAssert(cudaGetLastError(), "Error while launching InitializeTreeNode kernel");
		GpuAssert(cudaDeviceSynchronize(), "Error while running InitializeTreeNode kernel");

		previousLevelChildren = total * 2;
		GpuAssert(cudaFree(d_TempFlag), "Cannot free d_TempFlag");
		GpuAssert(cudaFree(d_TempIndx), "Cannot free d_TempIndx");
		GpuAssert(cudaFree(d_NodePrefix), "Cannot free d_NodePrefix");
		if (level == 8 || level == 16 || level == 24)
			GpuAssert(cudaFree(d_RealValues), "Cannot free d_RealValues");
	}

	free(sortedSubnetsBits);
	free(matchedIndexes);
	delete [] minus_one;
	delete [] prefixesStart;
	delete [] prefixesEnd;

	ModelBuildTime = timer.Stop();
	GpuAssert(cudaSetDevice(0), "Cannot reset device.");
}

__global__ void PrepareIPList(unsigned char * ipData, unsigned int * ipList, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned char b1, b2, b3, b4;

	while(i < size)
	{
		b1 = ipData[i * 5];
		b2 = ipData[i * 5 + 1];
		b3 = ipData[i * 5 + 2];
		b4 = ipData[i * 5 + 3];
		ipList[i] = (b1 << 24) + (b2 << 16) + (b3 << 8) + b4;

		i += blockDim.x * gridDim.x;
	}
}

TreeResult TreeMatcher::Match(IPSet set)
{
	GpuAssert(cudaSetDevice(Setup.DeviceID), "Cannot set cuda device.");
	Timer timer;

	TreeResult result;

	int threads = Setup.Threads;
	int blocks = 1;
	if (set.Size > Setup.Threads)
	{
		threads = Setup.Threads;
		blocks = set.Size / Setup.Threads;
		if (set.Size % Setup.Threads)
			blocks++;
	}

	unsigned int * d_IPList;
	GpuAssert(cudaMalloc((void**)&d_IPList, set.Size * sizeof(unsigned int)), "Cannot allocate memory for d_IPList");

	PrepareIPList << <Setup.Blocks, Setup.Threads >> > (set.d_IPData, d_IPList, set.Size);
	GpuAssert(cudaGetLastError(), "Error while launching PrepareIPList");
	GpuAssert(cudaDeviceSynchronize(), "Error while running PrepareIPList");

	int * d_MatchedIndexes;
	GpuAssert(cudaMalloc((void**)&d_MatchedIndexes, set.Size * sizeof(int)), "Cannot allocate memory for d_MatchedIndexes");

	unsigned int * d_ThreadTimeStart;
	GpuAssert(cudaMalloc((void**)&d_ThreadTimeStart, set.Size * sizeof(unsigned int)), "Cannot allocate memory for d_ThreadTimeStart");

	unsigned int * d_ThreadTimeEnd;
	GpuAssert(cudaMalloc((void**)&d_ThreadTimeEnd, set.Size * sizeof(unsigned int)), "Cannot allocate memory for d_ThreadTimeEnd");

	unsigned int * d_ThreadTime;
	GpuAssert(cudaMalloc((void**)&d_ThreadTime, set.Size * sizeof(unsigned int)), "Cannot allocate memory for d_ThreadTime");

	timer.Start();

	if(UsePresorting)
		thrust::sort(thrust::device, d_IPList, d_IPList + set.Size);

	if (!UseMidLevels)
	{
		AssignIPSubnet << <blocks, threads >> > (Tree.d_Tree, d_IPList, d_MatchedIndexes, set.Size, Tree.MinPrefix, d_ThreadTimeStart, d_ThreadTimeEnd, d_ThreadTime);
		GpuAssert(cudaGetLastError(), "Error while launching AssignIPSubnet kernel");
		GpuAssert(cudaDeviceSynchronize(), "Error while running AssignIPSubnet kernel");
	}
	else
	{
		AssignIPSubnetWithMidLevels << < blocks, threads >> > (Tree.d_Tree, d_IPList, d_MatchedIndexes, Tree.d_Level16, Tree.d_Level24, Tree.d_Level8, set.Size, Tree.MinPrefix, d_ThreadTimeStart, d_ThreadTimeEnd, d_ThreadTime);
		GpuAssert(cudaGetLastError(), "Error while launching AssignIPSubnetWithMidLevels kernel");
		GpuAssert(cudaDeviceSynchronize(), "Error while running AssignIPSubnetWithMidLevels kernel");
	}

	result.MatchedIndexes = new int[set.Size];
	result.SortedSubnetsBits = new char[33 * Tree.Size];
	result.IPsList = new unsigned int[set.Size];
	result.IPsListSize = set.Size;

	result.ThreadTimeRecorded = true;
	result.ThreadTimeStart = new unsigned int[set.Size];
	result.ThreadTimeEnd = new unsigned int[set.Size];
	result.ThreadTime = new unsigned int[set.Size];

	GpuAssert(cudaMemcpy(result.MatchedIndexes, d_MatchedIndexes, set.Size * sizeof(int), cudaMemcpyDeviceToHost), "Cannot copy memory to MatchedIndexes");
	GpuAssert(cudaMemcpy(result.SortedSubnetsBits, Tree.d_SortedSubnetBits, 33 * Tree.Size * sizeof(char), cudaMemcpyDeviceToHost), "Cannot copy memory to SortedSubnetsBits");
	GpuAssert(cudaMemcpy(result.IPsList, d_IPList, set.Size * sizeof(unsigned int), cudaMemcpyDeviceToHost), "Cannot copy memory to IPsList");

	GpuAssert(cudaMemcpy(result.ThreadTimeStart, d_ThreadTimeStart, set.Size * sizeof(unsigned int), cudaMemcpyDeviceToHost), "Cannot copy memory to ThreadTimeStart");
	GpuAssert(cudaMemcpy(result.ThreadTimeEnd, d_ThreadTimeEnd, set.Size * sizeof(unsigned int), cudaMemcpyDeviceToHost), "Cannot copy memory to ThreadTimeEnd");
	GpuAssert(cudaMemcpy(result.ThreadTime, d_ThreadTime, set.Size * sizeof(unsigned int), cudaMemcpyDeviceToHost), "Cannot copy memory to ThreadTime");

	GpuAssert(cudaFree(d_MatchedIndexes), "Cannot free d_MatchedIndexes");
	GpuAssert(cudaFree(d_ThreadTime), "Cannot free dev_thread_time");
	GpuAssert(cudaFree(d_ThreadTimeStart), "Cannot free dev_thread_time_start");
	GpuAssert(cudaFree(d_ThreadTimeEnd), "Cannot free dev_thread_time_end");
	GpuAssert(cudaFree(d_IPList), "Cannot free d_IPList");

	result.MatchingTime = timer.Stop();
	GpuAssert(cudaSetDevice(0), "Cannot reset device.");

	return result;
}

void TreeModel::Dispose()
{
	GpuAssert(cudaSetDevice(Setup.DeviceID), "Cannot set cuda device.");
	GpuAssert(cudaFree(d_LevelNodes), "Cannot free d_LevelNodes");
	GpuAssert(cudaFree(d_PrefixesEnd), "Cannot free d_PrefixesEnd");
	GpuAssert(cudaFree(d_PrefixesStart), "CAnnot free d_PrefixesStart");
	GpuAssert(cudaFree(d_SubnetsIndx), "Cannot free d_SubnetsIndx");
	GpuAssert(cudaFree(d_Tree), "Cannot free d_Tree");
	GpuAssert(cudaFree(d_SortedSubnetBits), "Cannot free d_SortedSubnetBits");
	GpuAssert(cudaFree(d_Level16), "Cannot free d_Level16");
	GpuAssert(cudaFree(d_Level24), "Cannot free d_Level24");
	GpuAssert(cudaFree(d_Level8), "Cannot free d_Level8");
	GpuAssert(cudaSetDevice(0), "Cannot reset device.");
	disposed = true;
}
