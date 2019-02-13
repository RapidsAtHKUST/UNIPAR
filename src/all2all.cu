/*
 * all2all.cu
 *
 *  Created on: 2018-9-24
 *      Author: qiushuang
 */

#include "../include/dbgraph.h"
#include "../include/graph.h"
#include "../include/comm.h"
#include "../include/distribute.h"

#define DEVICE_SHIFT 0

extern voff_t * tmp_counts[NUM_OF_PROCS];
int lock_flag[NUM_OF_PROCS];

float all2all_time_async = 0;

extern "C"
{
int get_device_config (void)
{
	int num_of_devices;
	cudaGetDeviceCount(&num_of_devices);
	int device;
	for (device = 0; device < num_of_devices; ++device) {
		   cudaDeviceProp deviceProp;
		   cudaGetDeviceProperties(&deviceProp, device);
		   printf("Device %d has compute capability %d.%d.\n",
		           device, deviceProp.major, deviceProp.minor);
	}
	return num_of_devices;
}

size_t get_device_memory_size (int did)
{
	cudaSetDevice(did);
	size_t total_size;
	size_t free_size;
	CUDA_CHECK_RETURN (cudaMemGetInfo(&free_size, &total_size));
	printf ("DEVICE %d: free memory: %lu, total memory %lu\n", did, free_size, total_size);
	return free_size;
}


long long master_all2all (master_t * mst)
{
	int num_of_devices = mst->num_of_devices;
	int num_of_cpus = mst->num_of_cpus;
	int num_of_procs = num_of_devices + num_of_cpus;
	int total_num_partitions_intra = mst->num_partitions[num_of_procs];
	int total_num_of_partitions = mst->total_num_partitions;
	uint elem_size = mst->mssg_size;
	voff_t total_inter_mssgs = 0;
	voff_t total_intra_mssgs = 0;
	int i;
	int j;
	for (i = 0; i < num_of_procs; i++)
	{
		memset (mst->soff[i], 0, sizeof(voff_t) * (total_num_of_partitions + 1));
		int nump_proc = mst->num_partitions[i + 1] - mst->num_partitions[i];
		memset (tmp_counts[i], 0, sizeof(voff_t) * nump_proc);
	}

	int world_size = mst->world_size;
	int world_rank = mst->world_rank;

	init_temp_offset_counts (total_num_of_partitions, mst);

	voff_t * gathered_offsets = (voff_t *) mpi_allgather_offsets (total_num_of_partitions, world_size, world_rank);
	// use this to init_count_displs_receive_all2all, needed for single and multiple nodes
	void * all2all_receive = init_count_displs_receive_all2all (total_num_of_partitions, mst, gathered_offsets);
	// only used to receive messages from other computer node, get NULL when only using single node
	voff_t * all2all_receive_offsets = (voff_t *) mpi_all2allv (NULL, all2all_receive, total_num_of_partitions, elem_size, world_size, world_rank);
	// use it to for receiving messages from processors of all computer node(s); needed for single and multiple nodes
	for (i=0; i<(mst->num_partitions[num_of_procs] * world_size); i++)
	{
		all2all_receive_offsets[i + 1] += all2all_receive_offsets[i];
	}


	// for each processor, calculate the number of messages to be communicated
	for (i = 0; i < num_of_procs; i++)
	{
		int nump_proc = mst->num_partitions[i + 1] - mst->num_partitions[i]; // number of partitions for a processor
		uint * roff = mst->roff[i] + nump_proc; // the first nump_proc partitions are intra-processor partitions
		int num_of_partitions = total_num_partitions_intra - (mst->num_partitions[i + 1] - mst->num_partitions[i]);
		//total_num_partitions_intra - (num_partitions_procs[i + 1] - num_partitions_procs[i]);
		int * plist = mst->not_reside[i];
		int pid; // partition id
		int prid; // processor id
		int index;
		for (j = 0; j < num_of_partitions; j++)
		{
			pid = plist[j]; // partition id
			prid = mst->r2s[pid]; // processor id
			index = mst->id2index[prid][pid];
			mst->soff[prid][index + 1] += roff[j + 1] - roff[j];
			tmp_counts[prid][index] += roff[j + 1] - roff[j]; // record total number of messages for each partition intra-compute-node
		}
	}

	// Get the number of messages transfered from other compute node for each partition in this compute node
	int num_of_partitions_intra = mst->num_partitions[num_of_procs];
	int * plist = mst->partition_list;
	int np_per_node;
	int np_node;
	get_np_node(&np_per_node, &np_node, total_num_of_partitions, world_size, world_rank);
	int start_partition_id = np_per_node * world_rank;
	for (i=0; i<num_of_procs; i++)
	{
		for (j=mst->num_partitions[i]; j<mst->num_partitions[i+1]; j++)
		{
			int pid = plist[j];
			int index = mst->id2index[i][pid];
			int w;
			for (w=0; w<world_size; w++)
			{
				if (w==world_rank)
					continue;
				voff_t start = all2all_receive_offsets[w*num_of_partitions_intra + pid-start_partition_id];
				voff_t end = all2all_receive_offsets[w*num_of_partitions_intra + pid-start_partition_id+1];
				mst->soff[i][index+1] += end - start;
			}
		}
	}

	// prefix-sum of the amount of messages
	for (i = 0; i < num_of_procs; i++)
	{
		// careful: number of partitions sent to each processor
		int num_of_partitions = mst->num_partitions[i + 1] - mst->num_partitions[i];
		for (j = 0; j < num_of_partitions; j++)
		{
			mst->soff[i][j + 1] += mst->soff[i][j];
		}
		total_inter_mssgs += mst->soff[i][num_of_partitions];
	}

	if(total_inter_mssgs == 0)
	{
		for (i=0; i<num_of_procs; i++)
		{
			int num_of_partitions = mst->num_partitions[i+1] - mst->num_partitions[i];
			for (j=0; j<num_of_partitions; j++)
			{
				total_intra_mssgs += mst->roff[i][num_of_partitions];
			}
		}
		long long local_sum = 0;
		long long global_sum = 0;
		if (mst->world_size > 1)
			mpi_allsum (&local_sum, &global_sum);
		if (global_sum == 0)
			return total_intra_mssgs;
		return global_sum;
	}

	// ******** for each processor, copy the message heading to the partitions not in this processor to the corresponding processors *****
	for (i=0; i < num_of_devices; i++)
	{
#ifdef SINGLE_NODE
		cudaSetDevice(world_rank * num_of_devices + i);
#else
		cudaSetDevice(i + DEVICE_SHIFT);
#endif
		CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	}
#pragma omp parallel for num_threads(num_of_procs) private(j)
	for (i = 0; i < num_of_procs; i++)
	{
		int num_of_partitions = mst->num_partitions[i + 1] - mst->num_partitions[i]; // number of partitions in a processor
		int * pfrom = mst->pfrom[i];
		void * send_ptr = mst->send[i];
		for (j = 0; j < num_of_partitions; j++)
		{
			int ij;
			int pr=0;
			voff_t offset = mst->soff[i][j];
			voff_t poff = 0;
			for (ij=0; ij<num_of_procs; ij++)
			{
				if (ij==i) continue;
				int nump_proc = mst->num_partitions[ij+1] - mst->num_partitions[ij];
				int index = pfrom[j*(num_of_procs-1) + pr++];
				voff_t * roff = mst->roff[ij] + nump_proc;
				voff_t init_offset = mst->roff[ij][nump_proc];
				uint num_of_elems = roff[index+1] - roff[index];
				void * rbuf = mst->receive[ij];
				memcpy((char*)send_ptr + ((ull)(offset + poff) * elem_size), (char*)rbuf + ((ull)(roff[index] - init_offset) * elem_size), (ull)elem_size*num_of_elems);
				poff += num_of_elems;
			}
		}
	}

	// ************ for each processor, copy the message transfered from other compute nodes ************
	plist = mst->partition_list;
	for (i=0; i<num_of_procs; i++)
	{
		for (j=mst->num_partitions[i]; j<mst->num_partitions[i+1]; j++)
		{
			int pid = plist[j];
			int index = mst->id2index[i][pid]; ////// CAREFUL HERE!!!!!!!!!!!!!!!!!!!!!!!!
			voff_t cnt = 0;
			int w;
			for (w=0; w<world_size; w++)
			{
				if (w==world_rank)
					continue;
				voff_t start = all2all_receive_offsets[w*num_of_partitions_intra + pid-start_partition_id];
				voff_t end = all2all_receive_offsets[w*num_of_partitions_intra + pid-start_partition_id+1];
//				printf ("WORLD RANK %d: w=%d, j=%d, start=%u, end=%u\n", world_rank, w, j, start, end);
				void * send_buf = (char*)all2all_receive + (ull)elem_size * start; // different from send_ptr in above
				void * receive_buf = (char*)mst->send[i] + (ull)(mst->soff[i][index] + tmp_counts[i][index] + cnt) * elem_size;
				memcpy((char*)receive_buf, send_buf, (ull)elem_size * (end-start));
				cnt += end-start;
			}
		}
	}

/*	for (i=0; i<num_of_procs; i++)
	{
		printf ("WORLD RANK %d::::: CCCCCCCCCCheck send offsets::::::::::::", world_rank);
		print_offsets (mst->soff[i], (total_num_of_partitions+1));
	}*/

	long long local_sum = total_inter_mssgs;
	long long global_sum = 0;
	if (mst->world_size > 1)
	{
		mpi_allsum (&local_sum, &global_sum);
		return global_sum;
	}
	return local_sum;

}

void * master_all2all_async (void * arg)
{
	comm_arg * cmm_arg = (comm_arg *) arg;
	master_t * mst = cmm_arg->mst;
	int num_of_devices = mst->num_of_devices;
	int num_of_cpus = mst->num_of_cpus;
	int num_of_procs = num_of_devices + num_of_cpus;
	int total_num_partitions_intra = mst->num_partitions[num_of_procs];
	int total_num_of_partitions = mst->total_num_partitions;
	uint elem_size = mst->mssg_size;
	cmm_arg->num_mssgs = 0;
	voff_t total_inter_mssgs = 0;
	voff_t total_intra_mssgs = 0;
	evaltime_t start, end;
	int i;
	int j;
	for (i=0; i<num_of_procs; i++)
	{
//		while(mst->flag[i] != 1) {}
		while(lock_flag[i] != 1) {}
	}

	gettimeofday(&start, NULL);
	for (i = 0; i < num_of_procs; i++)
	{
		memset (mst->soff[i], 0, sizeof(voff_t) * (total_num_of_partitions + 1));
		int nump_proc = mst->num_partitions[i + 1] - mst->num_partitions[i];
		memset (tmp_counts[i], 0, sizeof(voff_t) * nump_proc);
	}

	int world_size = mst->world_size;
	int world_rank = mst->world_rank;

	init_temp_offset_counts (total_num_of_partitions, mst);

	voff_t * gathered_offsets = (voff_t *) mpi_allgather_offsets (total_num_of_partitions, world_size, world_rank); // use this to init_count_displs_receive_all2all

	void * all2all_receive = init_count_displs_receive_all2all (total_num_of_partitions, mst, gathered_offsets);
	voff_t * all2all_receive_offsets = (voff_t *) mpi_all2allv (NULL, all2all_receive, total_num_of_partitions, elem_size, world_size, world_rank);
	for (i=0; i<(mst->num_partitions[num_of_procs] * world_size); i++)
	{
		all2all_receive_offsets[i + 1] += all2all_receive_offsets[i];
	}

	// for each processor, calculate the amount of messages to be communicated within this compute node
	for (i = 0; i < num_of_procs; i++)
	{
		int nump_proc = mst->num_partitions[i + 1] - mst->num_partitions[i];
		uint * roff = mst->roff[i] + nump_proc; // the first nump_proc offsets are intra-processor partitions
		int num_of_partitions = total_num_partitions_intra - (mst->num_partitions[i + 1] - mst->num_partitions[i]);
		//total_num_partitions_intra - (num_partitions_procs[i + 1] - num_partitions_procs[i]);
		int * plist = mst->not_reside[i];
		int pid; // partition id
		int prid; // processor id
		int index;
		for (j = 0; j < num_of_partitions; j++)
		{
			pid = plist[j]; // partition id
			prid = mst->r2s[pid]; // processor id
			index = mst->id2index[prid][pid];
			mst->soff[prid][index + 1] += roff[j + 1] - roff[j];
			tmp_counts[prid][index] += roff[j + 1] - roff[j]; // record total number of messages for each partition intra-compute-node
		}
	}

	// Get the number of messages transfered from other compute node for each partition in this compute node
	int num_of_partitions_intra = mst->num_partitions[num_of_procs];
	int * plist = mst->partition_list;
	int np_per_node = (total_num_of_partitions + world_size - 1)/world_size;
	int start_partition_id = np_per_node * world_rank;
	for (i=0; i<num_of_procs; i++)
	{
		for (j=mst->num_partitions[i]; j<mst->num_partitions[i+1]; j++)
		{
			int pid = plist[j];
			int index = mst->id2index[i][pid];
			int w;
			for (w=0; w<world_size; w++)
			{
				if (w==world_rank)
					continue;
				voff_t start = all2all_receive_offsets[w*num_of_partitions_intra + pid-start_partition_id];
				voff_t end = all2all_receive_offsets[w*num_of_partitions_intra + pid-start_partition_id+1];
				mst->soff[i][index+1] += end - start;
//				printf ("WORLD RANK %d: w=%d, j=%d, start=%u, end=%u, index=%d\n", world_rank, w, j, start, end, index);
			}
		}
	}

	// prefix-sum of the amount of messages
	for (i = 0; i < num_of_procs; i++)
	{
		// careful: number of partitions sent to each processor
		int num_of_partitions = mst->num_partitions[i + 1] - mst->num_partitions[i];
		for (j = 0; j < num_of_partitions; j++)
		{
			mst->soff[i][j + 1] += mst->soff[i][j];
		}
		total_inter_mssgs += mst->soff[i][num_of_partitions];
	}

	if(total_inter_mssgs == 0)
	{
		for (i=0; i<num_of_procs; i++)
		{
			int num_of_partitions = mst->num_partitions[i+1] - mst->num_partitions[i];
			for (j=0; j<num_of_partitions; j++)
			{
				total_intra_mssgs += mst->roff[i][num_of_partitions];
			}
		}
		cmm_arg->num_mssgs = total_intra_mssgs;
	}


	// ******** for each processor, copy the message heading to the partitions not in this processor to the corresponding processors *****
	for (i=0; i < num_of_devices; i++)
	{
#ifdef SINGLE_NODE
		cudaSetDevice(world_rank * num_of_devices + i);
#else
		cudaSetDevice(i + DEVICE_SHIFT);
#endif
		CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	}
#pragma omp parallel for num_threads(num_of_procs) private(j)
	for (i = 0; i < num_of_procs; i++)
	{
		int num_of_partitions = mst->num_partitions[i + 1] - mst->num_partitions[i]; // number of partitions in a processor
		int * pfrom = mst->pfrom[i];
		void * send_ptr = mst->send[i];
		for (j = 0; j < num_of_partitions; j++)
		{
			int ij;
			int pr=0;
			voff_t offset = mst->soff[i][j];
			voff_t poff = 0;
			for (ij=0; ij<num_of_procs; ij++)
			{
				if (ij==i) continue;
				int nump_proc = mst->num_partitions[ij+1] - mst->num_partitions[ij];
				int index = pfrom[j*(num_of_procs-1) + pr++];
				voff_t * roff = mst->roff[ij] + nump_proc;
				voff_t init_offset = mst->roff[ij][nump_proc];
				uint num_of_elems = roff[index+1] - roff[index];
				void * rbuf = mst->receive[ij];
				memcpy((char*)send_ptr + ((ull)(offset + poff) * elem_size), (char*)rbuf + ((ull)(roff[index] - init_offset) * elem_size), (ull)elem_size*num_of_elems);
				poff += num_of_elems;
			}
		}
	}

	// ************ for each processor, copy the message transfered from other compute nodes ************
	for (i=0; i<num_of_procs; i++)
	{
//#pragma omp parallel for schedule(dynamic)
		for (j=mst->num_partitions[i]; j<mst->num_partitions[i+1]; j++)
		{
			int pid = plist[j];
			int index = mst->id2index[i][pid]; ////// CAREFUL HERE!!!!!!!!!!!!!!!!!!!!!!!!
			voff_t cnt = 0;
			int w;
			for (w=0; w<world_size; w++)
			{
				if (w==world_rank)
					continue;
				voff_t start = all2all_receive_offsets[w*num_of_partitions_intra + pid-start_partition_id];
				voff_t end = all2all_receive_offsets[w*num_of_partitions_intra + pid-start_partition_id+1];
				void * send_buf = (char*)all2all_receive + (ull)elem_size * start; // different from send_ptr in above
				void * receive_buf = (char*)mst->send[i] + (ull)(mst->soff[i][index] + tmp_counts[i][index] + cnt) * elem_size;
				memcpy((char*)receive_buf, send_buf, (ull)elem_size * (end-start));
				cnt += end-start;
			}
		}
	}

	cmm_arg->num_mssgs += total_inter_mssgs;
	for (i=0; i<num_of_procs; i++)
	{
#ifndef SYNC_ALL2ALL_
		atomic_set_value(&lock_flag[i], 0, 1);
#endif
//		printf ("WORLD RANK %d::::: CCCCCCCCCCheck send offsets::::::::::::", world_rank);
//		print_offsets (mst->soff[i], (total_num_of_partitions+1));
	}
	gettimeofday (&end, NULL);
	all2all_time_async += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	return ((void *) 0);
}
}
