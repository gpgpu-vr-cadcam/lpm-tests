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
__global__ void prepare_submits_indxes(unsigned int * dev_subnets_indx, int size)
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

__global__ void prepare_levels(int * dev_level8, int * dev_level16, int * dev_level24)
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
__global__ void initialize_tree_node(unsigned int * dev_temp_flag, int * dev_temp_Indx, int ** dev_tree, int * dev_real_values, int * dev_level, int level
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
__global__ void initialize_tree_level(int ** dev_tree, int level, int count, int * dev_level_nodes)
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

__global__ void get_prefixes_ranges(char* dev_sorted_subnets_bits, int * dev_prefixes_start, int * dev_prefixes_end, int size)
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

__global__ void prepare_sorted_subnets_bits_list(unsigned char * dev_subnets, char * dev_sorted_subnets_bits, unsigned int * dev_subnets_indx, int size)
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

__global__ void initialize_tree_root(int ** dev_tree, int * dev_level_nodes)
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


__global__ void flag_bit_location(unsigned int * dev_temp_flag, int * dev_temp_Indx, char * dev_sorted_subnets_bits, int ** dev_tree, int * dev_prefixes_start, int * dev_prefixes_end, char * currentNodePrefix, int * dev_real_values, int level_max, int level, int subnets_size, int totalelements)
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
__global__ void assign_ip_subnet_old(int ** dev_tree, unsigned int * dev_ips_list, int * dev_matched_ip_subnet, int * dev_level8, int * dev_level16,
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

__global__ void assign_ip_subnet(int ** dev_tree, unsigned int * dev_ips_list, int * dev_matched_ip_subnet, int searchedIpsSize, char min_prefx, unsigned int  * time_thread_start, unsigned int  * time_thread_end, unsigned int  * time_thread)
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


void convert_from_bits(char * dev_in_bits, unsigned char * out_byte)
{
	unsigned char maskValues[8] = { 128, 64, 32, 16, 8, 4, 2, 1 };
	for (int j = 0; j < 4; j++)
	{
		out_byte[j] = 0;
		for (int i = 0; i < 8; i++)
		{
			if (dev_in_bits[j * 8 + i] == 1)
				out_byte[j] += maskValues[i];
		}
	}



}

void printresult(int * dev_matched_indx, int searched_ips, char* dev_sorted_subnets_bits, unsigned int * dev_ip_list, int print_type)
{
	//if (print_type == 0) return;
	for (int i = 0; i < searched_ips; i++)
	{

		if (dev_matched_indx[i] > -1)
		{
			if (print_type == 0) continue;
			if (i % print_type != 0)
				continue;
			unsigned char * submit = new unsigned char[5];
			convert_from_bits(dev_sorted_subnets_bits + dev_matched_indx[i] * 33, submit);
			submit[4] = dev_sorted_subnets_bits[dev_matched_indx[i] * 33 + 32];
			printf("IP_ID: %d -> sIndx: %d - ip: %d.%d.%d.%d --> subnet: %d.%d.%d.%d/%d\n", i, dev_matched_indx[i], ((dev_ip_list[i] & 4278190080) >> 24), ((dev_ip_list[i] & 16711680) >> 16), ((dev_ip_list[i] & 65280) >> 8), (dev_ip_list[i] & 255), submit[0], submit[1], submit[2], submit[3], submit[4]);
		}
		else
			printf("IP_ID : %d -> ip: %d.%d.%d.%d ->  sIndx :  %d\n", i, ((dev_ip_list[i] & 4278190080) >> 24), ((dev_ip_list[i] & 16711680) >> 16), ((dev_ip_list[i] & 65280) >> 8), (dev_ip_list[i] & 255), dev_matched_indx[i]);

	}


}

void TreeMatcher::BuildModel(IPSet set)
{
	Setup = set.Setup;

	int chunks = set.Size / 50000 + 1;

	// Host Variables:
	char * sorted_subnets_bits = nullptr;
	int * matchedIndexes = nullptr;
	int * minus_one;
	int * prefixes_start;
	int * prefixes_end;


	// Decvice Variables

	thrust::device_ptr<unsigned int> dev_subnets_indx_ptr;
	unsigned int * dev_subnets_indx;
	int * dev_prefixes_start;
	int * dev_prefixes_end;
	char * dev_node_prefix;
	int ** dev_tree;
	char * dev_sorted_subnets_bits;
	unsigned int * dev_temp_flag;
	int * dev_temp_Indx;
	int * dev_real_values;
	int * dev_full_level;
	int * dev_level8;
	int * dev_level16;
	int * dev_level24;
	int * dev_full_level_0;
	int * dev_full_level_1;
	int * dev_full_level_2;
	int * dev_level_nodes;


	// Threads Variables
	int resetThreads;
	int resetBlocks;
	int threads;
	int blocks;


	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(Setup.DeviceID);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}
	cudaStatus = cudaDeviceSetLimit(cudaLimitMallocHeapSize, 8000000 * chunks);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "set device heap limit failed!");
	}
	// Allocate GPU buffers for subnetmasks vector.
	cudaStatus = cudaMalloc((void**)&dev_sorted_subnets_bits, 33 * set.Size * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&dev_subnets_indx, set.Size * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&dev_tree, 33 * sizeof(int *));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}


	cudaStatus = cudaMalloc((void**)&dev_prefixes_start, 32 * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&dev_prefixes_end, 32 * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	prefixes_start = new int[32];
	prefixes_end = new int[32];

	minus_one = new int[33];
	for (int i = 0; i < 33; i++)
		minus_one[i] = -1;

	cudaStatus = cudaMemcpy(dev_prefixes_start, minus_one, 32 * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}


	cudaStatus = cudaMemcpy(dev_prefixes_end, minus_one, 32 * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	// prepare levels
	cudaStatus = cudaMalloc((void**)&dev_level16, 256 * 256 * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	cudaStatus = cudaMalloc((void**)&dev_level24, 256 * 256 * 256 * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	cudaStatus = cudaMalloc((void**)&dev_level8, 256 * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	/// prepare Subnets Indexes
	resetThreads = Setup.Threads;
	resetBlocks = 16777216 / resetThreads;
	if (16777216 % resetThreads > 0)
		resetBlocks++;

	prepare_levels << <resetBlocks, resetThreads >> >(dev_level8, dev_level16, dev_level24);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching prepare_levels!\n", cudaStatus);
	}

	resetThreads = Setup.Threads;
	resetBlocks = 1;
	if (set.Size > Setup.Threads)
	{
		resetThreads = Setup.Threads;
		resetBlocks = set.Size / Setup.Threads;
		if (set.Size % Setup.Threads > 0)
			resetBlocks++;
	}

	prepare_submits_indxes << <resetBlocks, resetThreads >> >(dev_subnets_indx, set.Size);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching prepare_submits_indxes!\n", cudaStatus);
	}
	// Sorting Subnets
	dev_subnets_indx_ptr = thrust::device_pointer_cast(dev_subnets_indx);
	thrust::sort(dev_subnets_indx_ptr, dev_subnets_indx_ptr + set.Size, my_sort_functor(5, thrust::raw_pointer_cast(set.d_IPData)));
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching stable_sort!\n", cudaStatus);
	}

	prepare_sorted_subnets_bits_list << <resetBlocks, resetThreads >> >(set.d_IPData, dev_sorted_subnets_bits, dev_subnets_indx, set.Size);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching prepare_sorted_subnets_bits_list!\n", cudaStatus);
	}
	get_prefixes_ranges << <resetBlocks, resetThreads >> >(dev_sorted_subnets_bits, dev_prefixes_start, dev_prefixes_end, set.Size);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching get_prefixes_ranges!\n", cudaStatus);
	}

	cudaStatus = cudaMemcpy(prefixes_start, dev_prefixes_start, 32 * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}
	cudaStatus = cudaMemcpy(prefixes_end, dev_prefixes_end, 32 * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	int min_prefix;
	for (int i = 0; i < 32; i++)
		if (prefixes_start[i] != -1)
		{
			min_prefix = i + 1;
			break;
		}
	printf("Min Prefix : %d \n", min_prefix);

	int previousLevelChildren;
	previousLevelChildren = 2;



	cudaStatus = cudaMalloc((void**)&dev_level_nodes, 33 * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMemcpy(dev_level_nodes, minus_one, 33 * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	initialize_tree_root << <1, 1 >> >(dev_tree, dev_level_nodes);
	for (int level = 1; level < 33; level++)
	{
		//printf("level:  %d  -> %d\n", level, previousLevelChildren);
		dev_real_values = NULL;
		cudaStatus = cudaMalloc((void**)&dev_temp_flag, previousLevelChildren * sizeof(unsigned int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed! dev_temp_flag");
		}
		cudaStatus = cudaMalloc((void**)&dev_temp_Indx, previousLevelChildren * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed! dev_temp_Indx");
		}
		cudaStatus = cudaMalloc((void**)&dev_node_prefix, level * previousLevelChildren * sizeof(char));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed! dev_temp_Indx");
		}

		if (level == 8 || level == 16 || level == 24)
		{
			cudaStatus = cudaMalloc((void**)&dev_real_values, previousLevelChildren * sizeof(int));
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMalloc failed! dev_temp_Indx");
			}
		}

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
			flag_bit_location << <blocks, threads >> >(dev_temp_flag, dev_temp_Indx, dev_sorted_subnets_bits, dev_tree, dev_prefixes_start, dev_prefixes_end, dev_node_prefix, dev_real_values, 8, level, set.Size, previousLevelChildren);
		else if (level == 16)
			flag_bit_location << <blocks, threads >> >(dev_temp_flag, dev_temp_Indx, dev_sorted_subnets_bits, dev_tree, dev_prefixes_start, dev_prefixes_end, dev_node_prefix, dev_real_values, 16, level, set.Size, previousLevelChildren);
		else if (level == 24)
			flag_bit_location << <blocks, threads >> >(dev_temp_flag, dev_temp_Indx, dev_sorted_subnets_bits, dev_tree, dev_prefixes_start, dev_prefixes_end, dev_node_prefix, dev_real_values, 24, level, set.Size, previousLevelChildren);
		else
			flag_bit_location << <blocks, threads >> >(dev_temp_flag, dev_temp_Indx, dev_sorted_subnets_bits, dev_tree, dev_prefixes_start, dev_prefixes_end, dev_node_prefix, NULL, -1, level, set.Size, previousLevelChildren);




		thrust::device_ptr< unsigned int> dev_temp_flag_ptr(dev_temp_flag);

		thrust::inclusive_scan(dev_temp_flag_ptr, dev_temp_flag_ptr + previousLevelChildren, dev_temp_flag_ptr);


		int total;

		cudaStatus = cudaMemcpy(&total, dev_temp_flag + previousLevelChildren - 1, sizeof(int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed! total");
		}
		//printf("level : %d  total: %d\n", level, total);
		if (total == 0)
			break;

		initialize_tree_level << <1, 1 >> >(dev_tree, level, total, dev_level_nodes);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching initialize_tree_level!\n", cudaStatus);
		}
		if (level == 8)
			dev_full_level = dev_level8;
		else if (level == 16)
			dev_full_level = dev_level16;
		else
			dev_full_level = dev_level24;

		initialize_tree_node << <blocks, threads >> >(dev_temp_flag, dev_temp_Indx, dev_tree, dev_real_values, dev_full_level, level, previousLevelChildren);
		GpuAssert(cudaPeekAtLastError(), "Error while launching initialize_tree_node kernel");
		GpuAssert(cudaDeviceSynchronize(), "Error while running initialize_tree_node kernel");


		previousLevelChildren = total * 2;
		cudaFree(dev_temp_flag);
		cudaFree(dev_temp_Indx);
		cudaFree(dev_node_prefix);
		if (level == 8 || level == 16 || level == 24)
			cudaFree(dev_real_values);
		//print_tree << <1, 1 >> >(dev_tree, level_nodes, level + 1, level);
	}

	free(sorted_subnets_bits);
	free(matchedIndexes);
	free(minus_one);
	free(prefixes_start);
	free(prefixes_end);

	cudaFree(dev_level_nodes);
	cudaFree(dev_prefixes_end);
	cudaFree(dev_prefixes_start);
	cudaFree(dev_subnets_indx);
	cudaFree(dev_tree);
	cudaFree(dev_sorted_subnets_bits);
	cudaFree(dev_subnets_indx);
	cudaFree(dev_full_level);
	cudaFree(dev_full_level_0);
	cudaFree(dev_full_level_1);
	cudaFree(dev_full_level_2);
	cudaFree(dev_level16);
	cudaFree(dev_level24);
	cudaFree(dev_level8);
	cudaFree(dev_node_prefix);
	cudaFree(dev_real_values);
	cudaFree(dev_temp_flag);
	cudaFree(dev_temp_Indx);

	GpuAssert(cudaSetDevice(0), "Cannot reset device.");
}

void TreeMatcher::Match(IPSet set)
{
}
