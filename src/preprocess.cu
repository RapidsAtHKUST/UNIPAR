/*
 * pre_process.cu
 *
 *  Created on: 2018-3-28
 *      Author: qiushuang
 *
 *  This file preprocesses De Bruijn graph: removing one-directed edge, indexing vertices with their location index,
 *  using new index to replace the neighbors of vertices, splitting and gathering vertices to junctions and linear vertices
 */

//#include <cub/cub.cuh>
#include <pthread.h>
#include "../include/dbgraph.h"
#include "../include/comm.h"
#include "../include/distribute.h"
#include "../include/share.h"
#include "malloc.cuh"
#include "preprocess.cuh"
#include "../include/scan.cu"

static uint * size_prime_index_ptr;
static uint * size_prime_index_host;

extern float elem_factor;
voff_t max_ss = 0;
extern int cutoff;

float push_offset_time[NUM_OF_PROCS] = {0,0,0,0,0};
float push_time[NUM_OF_PROCS] = {0,0,0,0,0};
float pull_intra_time[NUM_OF_PROCS] = {0,0,0,0,0};
float pull_inter_time[NUM_OF_PROCS] = {0,0,0,0,0};
float memcpydh_time[NUM_OF_PROCS] = {0,0,0,0,0};
float memcpyhd_time[NUM_OF_PROCS] = {0,0,0,0,0};
float over_time[NUM_OF_PROCS] = {0,0,0,0,0};

extern float all2all_time_async;
extern int lock_flag[NUM_OF_PROCS];
extern double mssg_factor;
double junction_factor = 0;
extern float inmemory_time;

extern uint gmax_lsize;
extern uint gmax_jsize;

extern "C"
{
void init_hashtab_data_gpu (int did, master_t * mst, dbmeta_t * dbm, dbtable_t * tbs)
{
	int * num_partitions = mst->num_partitions;
	int * partition_list = mst->partition_list;
	int num_of_partitions = num_partitions[did+1]-num_partitions[did];// number of partitions in this processor
	int total_num_partitions = mst->total_num_partitions; // total number of partitions in this compute node
	voff_t * index_offset = mst->index_offset[did];

	int world_size = mst->world_size;
	int world_rank = mst->world_rank;
	int np_per_node = (total_num_partitions + world_size - 1)/world_size;
	int start_partition_id = np_per_node*world_rank;

#ifdef SINGLE_NODE
	int num_of_devices = mst->num_of_devices;
	CUDA_CHECK_RETURN(cudaSetDevice (world_rank * num_of_devices + did));
#else
	CUDA_CHECK_RETURN(cudaSetDevice (did + DEVICE_SHIFT));
#endif

	CUDA_CHECK_RETURN (cudaMalloc (&size_prime_index_ptr, sizeof(uint) * num_of_partitions));
	CUDA_CHECK_RETURN (cudaMemcpyToSymbol (size_prime_index, &size_prime_index_ptr, sizeof(uint*)));
	size_prime_index_host = (uint *) malloc (sizeof(uint) * num_of_partitions);
	CHECK_PTR_RETURN (size_prime_index_host, "malloc size_prime_index_host error!\n");

	int i;
	voff_t offset = 0;
	for (i=0; i<num_of_partitions; i++)
	{
		int poffset = num_partitions[did];
		int pid = partition_list[poffset+i] - start_partition_id;//!!!be careful here, this pid is not the global partition id
		int pindex = mst->id2index[did][pid + start_partition_id];
		if (pindex != i)
		{
			printf ("ERROR IN DISTRIBUTING PARTITIONS!!!!!!!!\n");
//			exit(0);
		}
		voff_t size = tbs[pid].size;
		CUDA_CHECK_RETURN (cudaMemcpy(dbm->comm.send, tbs[pid].buf, sizeof(entry_t) * size, cudaMemcpyHostToDevice));
		int num_of_blocks = (size + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		init_hashtab <<<block_size, THREADS_PER_BLOCK_NODES>>> (size, offset);
//		init_hashtab_gpu <<<block_size, THREADS_PER_BLOCK_NODES>>> (size, offset);
		index_offset[i] = offset;
		offset += size;

		uint num_of_elems = tbs[pid].num_elems;
		size_prime_index_host[i] = higher_prime_index (num_of_elems * elem_factor);
		free (tbs[pid].buf);
	}
	index_offset[i] = offset;
	CUDA_CHECK_RETURN (cudaMemcpy(size_prime_index_ptr, size_prime_index_host, sizeof(uint) * num_of_partitions, cudaMemcpyHostToDevice));
//	printf ("index offset on GPU %d: \n", did);
//	print_offsets(index_offset[i], num_of_partitions);
}

void finalize_hashtab_data_gpu (void)
{
	cudaFree (size_prime_index_ptr);
	free (size_prime_index_host);
}

void d2h_mem (ull * dkmers, vid_t * dvids, ull * hkmers, vid_t * hvids, pair_t * pairs, uint size)
{
	CUDA_CHECK_RETURN (cudaMemcpy(hkmers, dkmers, sizeof(ull) * size, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN (cudaMemcpy(hvids, dvids, sizeof(vid_t) * size, cudaMemcpyDeviceToHost));
	uint i;
	for (i=0; i<size; i++)
	{
		pairs[i].kmer = hkmers[i];
		pairs[i].vid = hvids[i];
	}
}

void d2h_mem2 (kmer_t * dkmers, vid_t * dvids, kmer_t * hkmers, vid_t * hvids, kmer_vid_t * pairs, uint size)
{
	CUDA_CHECK_RETURN (cudaMemcpy(hkmers, dkmers, sizeof(kmer_t) * size, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN (cudaMemcpy(hvids, dvids, sizeof(vid_t) * size, cudaMemcpyDeviceToHost));
	uint i;
	for (i=0; i<size; i++)
	{
		pairs[i].kmer = hkmers[i];
		pairs[i].vid = hvids[i];
	}
}

void h2d_mem (ull * dkmers, vid_t * dvids, ull * hkmers, vid_t * hvids, pair_t * pairs, uint size)
{
	uint i;
	for (i=0; i<size; i++)
	{
		hkmers[i] = pairs[i].kmer;
		hvids[i] = pairs[i].vid;
	}
	CUDA_CHECK_RETURN (cudaMemcpy(dkmers, hkmers, sizeof(ull) * size, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN (cudaMemcpy(dvids, hvids, sizeof(vid_t) * size, cudaMemcpyHostToDevice));
}

void h2d_mem2 (kmer_t * dkmers, vid_t * dvids, kmer_t * hkmers, vid_t * hvids, kmer_vid_t * pairs, uint size)
{
	uint i;
	for (i=0; i<size; i++)
	{
		hkmers[i] = pairs[i].kmer;
		hvids[i] = pairs[i].vid;
	}
	CUDA_CHECK_RETURN (cudaMemcpy(dkmers, hkmers, sizeof(kmer_t) * size, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN (cudaMemcpy(dvids, hvids, sizeof(vid_t) * size, cudaMemcpyHostToDevice));
}

void init_binary_data_gpu (int did, master_t * mst, dbmeta_t * dbm, dbtable_t * tbs)
{
	int * num_partitions = mst->num_partitions;
	int * partition_list = mst->partition_list;
	int num_of_partitions = num_partitions[did+1]-num_partitions[did];// number of partitions in this processor
	int total_num_partitions = mst->total_num_partitions; // total number of partitions in this compute node
	voff_t * index_offset = mst->index_offset[did];

	int world_size = mst->world_size;
	int world_rank = mst->world_rank;
	int np_per_node = (total_num_partitions + world_size - 1)/world_size;
	int start_partition_id = np_per_node*world_rank;

#ifdef SINGLE_NODE
	int num_of_devices = mst->num_of_devices;
	CUDA_CHECK_RETURN(cudaSetDevice (world_rank * num_of_devices + did));
#else
	CUDA_CHECK_RETURN(cudaSetDevice (did + DEVICE_SHIFT));
#endif

	int i;
	uint offset = 0;
	kmer_vid_t * pairs = (kmer_vid_t *) malloc (sizeof(kmer_vid_t) * max_ss);
	kmer_t * hkmers = (kmer_t *) malloc (sizeof(kmer_t) * max_ss);
	vid_t * hvids = (vid_t *) malloc (sizeof(vid_t) * max_ss);
#ifdef USE_CUB_
	void * dtmp;
	size_t temp_size = 0;
	cub::DeviceRadixSort::SortPairs (dtmp, temp_size, dbm->before_sort, dbm->sorted_kmers, dbm->before_vids, dbm->sorted_vids, max_ss, 0, sizeof(ull) * 8);
	printf ("max subsize: %u, cub device temp size for sort:%lu\n", max_ss, temp_size);
	CUDA_CHECK_RETURN (cudaMalloc(&dtmp, temp_size));
#endif

	for (i=0; i<num_of_partitions; i++)
	{
		int poffset = num_partitions[did];
		int pid = partition_list[poffset+i] - start_partition_id;//!!!be careful here, this pid is not the global partition id
		int pindex = mst->id2index[did][pid + start_partition_id];
		if (pindex != i)
		{
			printf ("ERROR IN DISTRIBUTING PARTITIONS!!!!!!!!\n");
//			exit(0);
		}
		voff_t size = tbs[pid].size;
		CUDA_CHECK_RETURN (cudaMemset (dbm->lvld, 0, sizeof(voff_t) * (size+1)));
		CUDA_CHECK_RETURN (cudaMemcpy(dbm->comm.send, tbs[pid].buf, sizeof(entry_t) * size, cudaMemcpyHostToDevice));
		int num_of_blocks = (size + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		init_kmers <<<block_size, THREADS_PER_BLOCK_NODES>>> (size, offset);

		inclusive_scan<voff_t> (dbm->lvld + 1, size, NULL);
		CUDA_CHECK_RETURN (cudaMemcpy (&offset, &dbm->lvld[size], sizeof(voff_t), cudaMemcpyDeviceToHost));

		gather_kmers <<<block_size, THREADS_PER_BLOCK_NODES>>> (size, offset);
#ifdef USE_CUB_
		cub::DeviceRadixSort::SortPairs (dtmp, temp_size, dbm->before_sort, dbm->sorted_kmers+index_offset[i], dbm->before_vids, dbm->sorted_vids+index_offset[i], offset, 0, sizeof(ull) * 8);
#endif
		if (offset > max_ss)
		{
			printf ("error!!!!!!\n");
//			exit(0);
		}

#ifndef USE_CUB_
		d2h_mem2 (dbm->before_sort, dbm->before_vids, hkmers, hvids, pairs, offset);
		tbb_kmer_vid_sort (pairs, offset); // sort the kmers with the vertex ids
		h2d_mem2 (dbm->sorted_kmers + index_offset[i], dbm->sorted_vids + index_offset[i], hkmers, hvids, pairs, offset);
#endif
		num_of_blocks = (offset + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		gather_edges <<<block_size, THREADS_PER_BLOCK_NODES>>> (offset, index_offset[i]); // gather edges with sorted vertices
		free (tbs[pid].buf);
		index_offset[i+1] = index_offset[i] + offset;
	}
#ifdef USE_CUB_
	cudaFree(dtmp);
#endif
	free (pairs);
	free (hkmers);
	free (hvids);
}

void init_binary_data_gpu_sorted (int did, master_t * mst, dbmeta_t * dbm, dbtable_t * tbs)
{
	int * num_partitions = mst->num_partitions;
	int * partition_list = mst->partition_list;
	int num_of_partitions = num_partitions[did+1]-num_partitions[did];// number of partitions in this processor
	int total_num_partitions = mst->total_num_partitions; // total number of partitions in this compute node
	voff_t * index_offset = mst->index_offset[did];

	int world_size = mst->world_size;
	int world_rank = mst->world_rank;
	int np_per_node = (total_num_partitions + world_size - 1)/world_size;
	int start_partition_id = np_per_node*world_rank;

#ifdef SINGLE_NODE
	int num_of_devices = mst->num_of_devices;
	CUDA_CHECK_RETURN(cudaSetDevice (world_rank * num_of_devices + did));
#else
	CUDA_CHECK_RETURN(cudaSetDevice (did + DEVICE_SHIFT));
#endif

	int i;
	for (i=0; i<num_of_partitions; i++)
	{
		int poffset = num_partitions[did];
		int pid = partition_list[poffset+i] - start_partition_id;//!!!be careful here, this pid is not the global partition id
		int pindex = mst->id2index[did][pid + start_partition_id];
		if (pindex != i)
		{
			printf ("ERROR IN DISTRIBUTING PARTITIONS!!!!!!!!\n");
		}
		voff_t size = tbs[pid].size;
		CUDA_CHECK_RETURN (cudaMemcpy(dbm->comm.send, tbs[pid].buf, sizeof(entry_t) * size, cudaMemcpyHostToDevice));
		int num_of_blocks = (size + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		gather_vs <<<block_size, THREADS_PER_BLOCK_NODES>>> (size, index_offset[i]); // gather edges with sorted vertices
		free (tbs[pid].buf);
		index_offset[i+1] = index_offset[i] + size;
	}
}

void * neighbor_push_intra_pull_gpu (void * arg)
{
	pre_arg * garg = (pre_arg *) arg;
	int did = garg->did;
	comm_t * cm = &garg->dbm->comm;
	master_t * mst = garg->mst;
	int k = garg->k;
	int p = garg->p;
	int total_num_partitions = mst->total_num_partitions;
	int num_of_partitions = mst->num_partitions[did + 1] - mst->num_partitions[did];
	voff_t * index_offset = mst->index_offset[did];
	if (mst->world_rank == 0)
		printf ("WORLD RANK %d: Neigbhors push intra pull gpu %d:\n", mst->world_rank, did);

#ifdef SINGLE_NODE
	int world_rank = mst->world_rank;
	int num_of_devices = mst->num_of_devices;
	CUDA_CHECK_RETURN(cudaSetDevice (world_rank * num_of_devices + did));
#else
	CUDA_CHECK_RETURN(cudaSetDevice (did + DEVICE_SHIFT));
#endif
	CUDA_CHECK_RETURN (cudaMemset(cm->send_offsets, 0, sizeof(voff_t) * (total_num_partitions + 1)));

#ifdef MEASURE_TIME_
	evaltime_t start, end;
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&start, NULL);
#endif
	int i;
	for (i=0; i<num_of_partitions; i++)
	{
		voff_t size = index_offset[i+1] - index_offset[i];
		int num_of_blocks = (size + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;

//		push_mssg_offset_assign_id <<<block_size, THREADS_PER_BLOCK_NODES>>> (size, total_num_partitions, index_offset[i], k, p);
//		push_mssg_offset_assign_id_gpu <<<block_size, THREADS_PER_BLOCK_NODES>>> (size, total_num_partitions, index_offset[i], k, p);
		push_mssg_offset_assign_id_binary <<<block_size, THREADS_PER_BLOCK_NODES>>> (size, total_num_partitions, index_offset[i], k, p, cutoff);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	push_offset_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PUSH MSSG OFFSET FOR GPU *ASSIGNING IDS* INTRA PROCESSOR TIME: ");
#endif

	inclusive_scan<voff_t> (cm->send_offsets, total_num_partitions + 1, NULL);

	CUDA_CHECK_RETURN (cudaMemset(cm->tmp_send_offsets, 0, sizeof(voff_t) * (total_num_partitions + 1)));

#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&start, NULL);
#endif

	for (i=0; i< num_of_partitions; i++)
	{
		voff_t size = index_offset[i+1] - index_offset[i];
		int num_of_blocks = (size + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		push_mssg_assign_id_binary <<<block_size, THREADS_PER_BLOCK_NODES>>> (size, total_num_partitions, index_offset[i], k, p, cutoff);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	push_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PUSH MSSG FOR GPU *ASSIGNING IDS* INTRA PROCESSOR TIME: ");
#endif

	CUDA_CHECK_RETURN (cudaMemcpy(mst->roff[did], cm->send_offsets, sizeof(voff_t) * (total_num_partitions + 1), cudaMemcpyDeviceToHost));
	voff_t inter_start = mst->roff[did][num_of_partitions];
	voff_t inter_end = mst->roff[did][total_num_partitions];

	if (atomic_set_value(&lock_flag[did], 1, 0) == false)
		printf ("!!!!!!!!!!! CAREFUL: ATOMIC SET VALUE ERROR IN GPU %d\n", did);
//	mst->flag[did] = 1;
#ifdef MEASURE_MEMCPY_
	gettimeofday(&start, NULL);
	CUDA_CHECK_RETURN (cudaMemcpy(mst->receive[did], (assid_t *)cm->send + inter_start, sizeof(assid_t) * (inter_end-inter_start), cudaMemcpyDeviceToHost));
	gettimeofday(&end, NULL);
	memcpydh_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
#else
//	CUDA_CHECK_RETURN (cudaMemcpyAsync(mst->receive[did], (assid_t *)cm->send + inter_start, sizeof(assid_t) * (inter_end-inter_start), cudaMemcpyDeviceToHost, streams[did]));
	CUDA_CHECK_RETURN (cudaMemcpy(mst->receive[did], (assid_t *)cm->send + inter_start, sizeof(assid_t) * (inter_end-inter_start), cudaMemcpyDeviceToHost));
#endif

	if (INTER_BUF_FACTOR == 1)
	{
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&start, NULL);
#endif

	for (i=0; i<num_of_partitions; i++)
	{
		int poffset = mst->num_partitions[did];
		int pid = mst->partition_list[poffset+i];
		voff_t num_mssgs = mst->roff[did][i+1] - mst->roff[did][i];
		voff_t size = index_offset[i+1] - index_offset[i];
		int num_of_blocks = (num_mssgs + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		pull_mssg_assign_id_binary <<<block_size, THREADS_PER_BLOCK_NODES>>> (num_mssgs, pid, size, index_offset[i], total_num_partitions, 0, 1, did);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	pull_intra_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PULL MSSG FOR GPU %d LISTRANKING INTRA PROCESSOR TIME: ", did);
#endif
	}
	return ((void *) 0);
}

void * neighbor_inter_pull_gpu (void * arg)
{
	pre_arg * garg = (pre_arg *) arg;
	int did = garg->did;
	comm_t * cm = &garg->dbm->comm;
	master_t * mst = garg->mst;
	if (mst->world_rank == 0)
		printf ("WORLD RANK %d: neighbor inter pull gpu %d:\n", mst->world_rank, did);

	int total_num_partitions = mst->total_num_partitions;
	int num_of_partitions = mst->num_partitions[did + 1] - mst->num_partitions[did];
	voff_t * index_offset = mst->index_offset[did];

#ifdef SINGLE_NODE
	int world_rank = mst->world_rank;
	int num_of_devices = mst->num_of_devices;
	CUDA_CHECK_RETURN(cudaSetDevice (world_rank * num_of_devices + did));
#else
	CUDA_CHECK_RETURN(cudaSetDevice (did + DEVICE_SHIFT));
#endif

	voff_t receive_start = mst->roff[did][num_of_partitions];
#ifdef MEASURE_MEMCPY_
	evaltime_t start, end;
	gettimeofday(&start, NULL);
#endif
	CUDA_CHECK_RETURN (cudaMemcpy(cm->receive_offsets, mst->soff[did], sizeof(voff_t) * (num_of_partitions + 1), cudaMemcpyHostToDevice));
	voff_t inter_size = mst->soff[did][num_of_partitions];
	if (inter_size == 0)
		return ((void *) 0);

//	tbb_assid_sort ((assid_t *)(mst->send[did]), inter_size);

	CUDA_CHECK_RETURN (cudaMemcpy((assid_t*)cm->send + receive_start, mst->send[did], sizeof(assid_t) * inter_size, cudaMemcpyHostToDevice));
#ifdef MEASURE_MEMCPY_
	gettimeofday(&end, NULL);
	memcpyhd_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
#endif

#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&start, NULL);
#endif

	int i;
	for (i=0; i<num_of_partitions; i++)
	{
		int poffset = mst->num_partitions[did];
		int pid = mst->partition_list[poffset+i];
		voff_t size = index_offset[i+1] - index_offset[i];
		voff_t num_mssgs = mst->soff[did][i+1] - mst->soff[did][i];
		int num_of_blocks = (num_mssgs + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		pull_mssg_assign_id_binary <<<block_size, THREADS_PER_BLOCK_NODES>>> (num_mssgs, pid, size, index_offset[i], total_num_partitions, receive_start, 0, did);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	pull_inter_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PULL MSSG FOR GPU %d *ASSIGNING IDS* INTER PROCESSORS TIME: ", did);
#endif

	return ((void *) 0);
}

void * identify_vertices_gpu (void * arg)
{
	pre_arg * garg = (pre_arg *) arg;
	master_t * mst = garg->mst;
	dbmeta_t * dbm = garg->dbm;
	int did = garg->did;
	if (mst->world_rank == 0)
		printf ("WORLD RANK %d: CPU identifying vertices DID=%d:\n", mst->world_rank, did);

#ifdef SINGLE_NODE
	int world_rank = mst->world_rank;
	int num_of_devices = mst->num_of_devices;
	CUDA_CHECK_RETURN(cudaSetDevice (world_rank * num_of_devices + did));
#else
	CUDA_CHECK_RETURN(cudaSetDevice (did + DEVICE_SHIFT));
#endif

	int num_of_partitions = mst->num_partitions[did + 1] - mst->num_partitions[did];
	voff_t * index_offset = mst->index_offset[did];
	cudaMemset (dbm->lvld, 0, sizeof(voff_t) * (max_ss+1));
	int i;
#ifdef MEASURE_TIME_
	evaltime_t start, end;
	gettimeofday (&start, NULL);
#endif
	for (i=0; i<num_of_partitions; i++)
	{
		int poffset = mst->num_partitions[did];
		int pid = mst->partition_list[poffset+i];
		voff_t size = index_offset[i+1] - index_offset[i];
		int num_of_blocks = (size + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		CUDA_CHECK_RETURN (cudaMemset(dbm->jvld, 0, sizeof(uint) * size));
		CUDA_CHECK_RETURN (cudaMemset(dbm->lvld, 0, sizeof(uint) * size));
//		label_vertex_with_flags <<<block_size, THREADS_PER_BLOCK_NODES>>> (size, index_offset[i]);
		label_vertex_with_flags_binary <<<block_size, THREADS_PER_BLOCK_NODES>>> (size, index_offset[i], cutoff);
//		inclusive_scan<uint> (dbm->jvld + index_offset[i], size, NULL);
//		inclusive_scan<uint> (dbm->lvld + index_offset[i], size, NULL);
		inclusive_scan<uint> (dbm->jvld, size, NULL);
		inclusive_scan<uint> (dbm->lvld, size, NULL);
		voff_t jsize, lsize;
//		CUDA_CHECK_RETURN (cudaMemcpy(&jsize, &(dbm->jvld + index_offset[i])[size-1], sizeof(voff_t), cudaMemcpyDeviceToHost));
//		CUDA_CHECK_RETURN (cudaMemcpy(&lsize, &(dbm->lvld + index_offset[i])[size-1], sizeof(voff_t), cudaMemcpyDeviceToHost));
		CUDA_CHECK_RETURN (cudaMemcpy(&jsize, &(dbm->jvld)[size-1], sizeof(voff_t), cudaMemcpyDeviceToHost));
		CUDA_CHECK_RETURN (cudaMemcpy(&lsize, &(dbm->lvld)[size-1], sizeof(voff_t), cudaMemcpyDeviceToHost));
		mst->jid_offset[pid] = jsize;
		mst->id_offsets[pid+1] = jsize + lsize;
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& IDENTIFYING IDS OF VERTICES TIME: ");
#endif
	return ((void *)0);
}

void * assign_vertex_ids_gpu (void * arg)
{
	pre_arg * garg = (pre_arg *) arg;
	master_t * mst = garg->mst;
	int did = garg->did;
	dbmeta_t * dbm = garg->dbm;
	if (mst->world_rank == 0)
		printf ("WORLD RANK %d: CPU assigning vertex ids DID = %d:\n", mst->world_rank, did);

	int num_of_partitions = mst->num_partitions[did + 1] - mst->num_partitions[did];
	voff_t * index_offset = mst->index_offset[did];

#ifdef SINGLE_NODE
	int world_rank = mst->world_rank;
	int num_of_devices = mst->num_of_devices;
	CUDA_CHECK_RETURN(cudaSetDevice (world_rank * num_of_devices + did));
#else
	CUDA_CHECK_RETURN(cudaSetDevice (did + DEVICE_SHIFT));
#endif

	int i;
#ifdef MEASURE_TIME_
	evaltime_t start, end;
	gettimeofday (&start, NULL);
#endif
	for (i=0; i<num_of_partitions; i++)
	{
		int poffset = mst->num_partitions[did];
		int pid = mst->partition_list[poffset+i];
		voff_t size = index_offset[i+1] - index_offset[i];
		int num_of_blocks = (size + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		CUDA_CHECK_RETURN (cudaMemset(dbm->jvld, 0, sizeof(uint) * size));
		CUDA_CHECK_RETURN (cudaMemset(dbm->lvld, 0, sizeof(uint) * size));
		label_vertex_with_flags_binary <<<block_size, THREADS_PER_BLOCK_NODES>>> (size, index_offset[i], cutoff);
		inclusive_scan<uint> (dbm->jvld, size, NULL);
		inclusive_scan<uint> (dbm->lvld, size, NULL);
		assid_vertex_with_flags <<<block_size, THREADS_PER_BLOCK_NODES>>> (size, pid, index_offset[i]);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& ASSIGNING IDS OF VERTICES TIME: ");
#endif

	return ((void *)0);
}

void * gather_vertices_gpu (void * arg)
{
	pre_arg * garg = (pre_arg *) arg;
	master_t * mst = garg->mst;
	dbmeta_t * dbm = garg->dbm;
	int k = garg->k;
	int p = garg->p;
	d_jvs_t * js = garg->js;
	d_lvs_t * ls = garg->ls;
	ull * js_spids = garg->dbm->djs.spids;
	ull * js_spidsr = garg->dbm->djs.spidsr;
	uint * ls_spids = garg->dbm->dls.spids;
	subgraph_t * subgraph = garg->subgraph;
	int did = garg->did;
	printf ("identifying vertices gpu %d:\n", did);

	int total_num_partitions = mst->total_num_partitions;
	int num_of_partitions = mst->num_partitions[did + 1] - mst->num_partitions[did];
	voff_t * index_offset = mst->index_offset[did];

#ifdef SINGLE_NODE
	int world_rank = mst->world_rank;
	int num_of_devices = mst->num_of_devices;
	CUDA_CHECK_RETURN(cudaSetDevice (world_rank * num_of_devices + did));
#else
	CUDA_CHECK_RETURN(cudaSetDevice (did + DEVICE_SHIFT));
#endif

	int i;
#ifdef MEASURE_TIME_
	evaltime_t start, end;
	gettimeofday (&start, NULL);
#endif
	for (i=0; i<num_of_partitions; i++)
	{
		int poffset = mst->num_partitions[did];
		int pid = mst->partition_list[poffset+i];
		voff_t size = index_offset[i+1] - index_offset[i];
		int num_of_blocks = (size + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
//		gather_vertex_binary <<<block_size, THREADS_PER_BLOCK_NODES>>> (size, pid, index_offset[i], cutoff);
		CUDA_CHECK_RETURN (cudaMemset (js_spids, 0, sizeof(ull) * gmax_jsize));
		CUDA_CHECK_RETURN (cudaMemset (js_spidsr, 0, sizeof(ull) * gmax_jsize));
		CUDA_CHECK_RETURN (cudaMemset (ls_spids, 0, sizeof(uint) * gmax_lsize));
		gather_vertex_partitioned <<<block_size, THREADS_PER_BLOCK_NODES>>> (size, pid, index_offset[i], cutoff, k, p, total_num_partitions);
		uint jsize = mst->jid_offset[pid];
		uint lsize = mst->id_offsets[pid+1] - mst->id_offsets[pid] - jsize;
//		write_junctions_gpu (dbm, mst, jsize, lsize, pid, total_num_partitions, did);
//		write_linear_vertices_gpu (dbm, mst, jsize, lsize, pid, total_num_partitions, did);
		output_vertices_gpu (dbm, mst, jsize, lsize, pid, total_num_partitions, did, js, ls, subgraph);
		write_kmers_edges_gpu (dbm, mst, jsize, lsize, pid, total_num_partitions, did);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	print_exec_time (start, end, "DEVICE %d: &&&&&&&&&&&&&&&&&&&& GATHERING VERTICES TIME: ", did);
#endif

//	printf ("DEVICE %d: NUMBER OF VERTICES PROCESSED: %u\n", did, index_offset[num_of_partitions]);
//	write_ids_gpu (dbm, mst, num_of_partitions, did);

	return ((void *)0);
}

void * shakehands_push_respond_intra_push_gpu (void * arg)
{
	evaltime_t overs, overe;
	pre_arg * garg = (pre_arg *) arg;
	int did = garg->did;
	comm_t * cm = &garg->dbm->comm;
	master_t * mst = garg->mst;
	int k = garg->k;
	int p = garg->p;
	if (mst->world_rank == 0)
		printf ("WORLD RANK %d: Shakehands push respond intra push gpu %d:\n", mst->world_rank, did);

#ifdef SINGLE_NODE
	int num_of_devices = mst->num_of_devices;
	int world_rank = mst->world_rank;
	CUDA_CHECK_RETURN(cudaSetDevice (world_rank * num_of_devices + did));
#else
	CUDA_CHECK_RETURN(cudaSetDevice (did + DEVICE_SHIFT));
#endif
	gettimeofday (&overs, NULL);

	int total_num_partitions = mst->total_num_partitions;
	int num_of_partitions = mst->num_partitions[did + 1] - mst->num_partitions[did];
	int poffset = mst->num_partitions[did];
	voff_t * index_offset = mst->index_offset[did];

	CUDA_CHECK_RETURN (cudaMemset(cm->send_offsets, 0, sizeof(voff_t) * (total_num_partitions + 1)));

#ifdef MEASURE_TIME_
	evaltime_t start, end;
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&start, NULL);
#endif
	int i;
	for (i=0; i<num_of_partitions; i++)
	{
		voff_t size = index_offset[i+1] - index_offset[i];
		int num_of_blocks = (size + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		push_mssg_offset_shakehands <<<block_size, THREADS_PER_BLOCK_NODES, sizeof(vid_t)*(total_num_partitions+1)>>> (size, total_num_partitions, index_offset[i], k, p, cutoff);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	push_offset_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PUSH MSSG OFFSET FOR GPU *SHAKEHANDS* INTRA PROCESSOR TIME: ");
#endif

	inclusive_scan<voff_t> (cm->send_offsets, total_num_partitions+1, NULL);
	CUDA_CHECK_RETURN (cudaMemset(cm->tmp_send_offsets, 0, sizeof(voff_t) * (total_num_partitions + 1)));

#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&start, NULL);
#endif
	for (i=0; i< num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset+i];
		voff_t size = index_offset[i+1] - index_offset[i];
		int num_of_blocks = (size + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		push_mssg_shakehands <<<block_size, THREADS_PER_BLOCK_NODES, sizeof(vid_t)*(total_num_partitions+1)>>> (size, total_num_partitions, index_offset[i], k, p, pid, cutoff);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	push_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PUSH MSSG FOR GPU *SHAKEHANDS* INTRA PROCESSOR TIME: ");
#endif

#ifdef MEASURE_MEMCPY_
	gettimeofday(&start, NULL);
#endif
	CUDA_CHECK_RETURN (cudaMemcpy(mst->roff[did], cm->send_offsets, sizeof(voff_t) * (total_num_partitions + 1), cudaMemcpyDeviceToHost));
#ifdef MEASURE_MEMCPY_
	gettimeofday(&end, NULL);
	memcpydh_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
#endif
	voff_t inter_start = mst->roff[did][num_of_partitions];
	voff_t inter_end = mst->roff[did][total_num_partitions];

#ifdef MEASURE_MEMCPY_
	gettimeofday(&start, NULL);
	CUDA_CHECK_RETURN (cudaMemcpy(mst->receive[did], (shakehands_t *)cm->send + inter_start, sizeof(shakehands_t) * (inter_end-inter_start), cudaMemcpyDeviceToHost));
	gettimeofday(&end, NULL);
	memcpydh_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
#else
//	CUDA_CHECK_RETURN (cudaMemcpyAsync(mst->receive[did], (shakehands_t *)cm->send + inter_start, sizeof(shakehands_t) * (inter_end-inter_start), cudaMemcpyDeviceToHost, streams[did]));
	CUDA_CHECK_RETURN (cudaMemcpy(mst->receive[did], (shakehands_t *)cm->send + inter_start, sizeof(shakehands_t) * (inter_end-inter_start), cudaMemcpyDeviceToHost));
#endif
	CUDA_CHECK_RETURN (cudaMemset (cm->extra_send_offsets, 0, sizeof(voff_t) * (total_num_partitions+1)));

#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&start, NULL);
#endif
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset + i];
		voff_t num_mssgs = mst->roff[did][i+1] - mst->roff[did][i];
		int num_of_blocks = (num_mssgs + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		voff_t size = index_offset[i+1] - index_offset[i];
		push_mssg_offset_respond <<<block_size, THREADS_PER_BLOCK_NODES, sizeof(vid_t)*(total_num_partitions+1)>>> (num_mssgs, pid, size, index_offset[i], total_num_partitions, 0, 1, k, p, cutoff);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	pull_intra_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PUSH *RESPOND* OFFSET GPU INTRA PROCESSOR TIME: ");
#endif
	gettimeofday (&overe, NULL);
	over_time[did] += (float)((overe.tv_sec * 1000000 + overe.tv_usec) - (overs.tv_sec * 1000000 + overs.tv_usec)) / 1000;

	return ((void *) 0);
}

void * shakehands_pull_respond_inter_push_intra_pull_gpu (void * arg)
{
	evaltime_t overs, overe;
	pre_arg * carg = (pre_arg *) arg;
	int did = carg->did;
	int k = carg->k;
	int p = carg->p;
	comm_t * cm = &carg->dbm->comm;
	master_t * mst = carg->mst;
	int num_of_devices = mst->num_of_devices;
	int world_rank = mst->world_rank;
	if (world_rank == 0)
		printf ("WORLD RANK %d: shakehands pull respond inter push intra pull gpu %d\n", world_rank, did);

#ifdef SINGLE_NODE
	CUDA_CHECK_RETURN(cudaSetDevice (world_rank * num_of_devices + did));
#else
	CUDA_CHECK_RETURN(cudaSetDevice (did + DEVICE_SHIFT));
#endif
	gettimeofday (&overs, NULL);
	int total_num_partitions = mst->total_num_partitions;
	int num_of_partitions = mst->num_partitions[did + 1] - mst->num_partitions[did];
	int poffset = mst->num_partitions[did];
	voff_t * index_offset = mst->index_offset[did];

	voff_t receive_start = mst->roff[did][num_of_partitions];
	voff_t inter_size = mst->soff[did][num_of_partitions];

#ifdef MEASURE_MEMCPY_
	evaltime_t start, end;
	gettimeofday(&start, NULL);
#endif
	CUDA_CHECK_RETURN (cudaMemcpy(cm->receive_offsets, mst->soff[did], sizeof(voff_t) * (num_of_partitions + 1), cudaMemcpyHostToDevice));
#ifdef MEASURE_MEMCPY_
	gettimeofday(&end, NULL);
	memcpyhd_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	gettimeofday(&start, NULL);
#endif
	CUDA_CHECK_RETURN (cudaMemcpy((shakehands_t*)(cm->send) + receive_start, mst->send[did], sizeof(shakehands_t) * inter_size, cudaMemcpyHostToDevice));
#ifdef MEASURE_MEMCPY_
	gettimeofday(&end, NULL);
	memcpyhd_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
#endif

#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&start, NULL);
#endif
	int i;
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset + i];
		voff_t num_mssgs = mst->soff[did][i+1] - mst->soff[did][i];
		voff_t size = index_offset[i+1] - index_offset[i];
		int num_of_blocks = (num_mssgs + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		push_mssg_offset_respond <<<block_size, THREADS_PER_BLOCK_NODES, sizeof(vid_t)*(total_num_partitions+1)>>> (num_mssgs, pid, size, index_offset[i], total_num_partitions, receive_start, 0, k, p, cutoff);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	pull_inter_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PUSH *RESPOND* OFFSET GPU INTER PROCESSORS TIME: ");
#endif

	inclusive_scan<voff_t> (cm->extra_send_offsets, total_num_partitions+1, NULL);
	// *************** malloc (send and) receive buffer for pull and push mode
	voff_t rcv_size;
	CUDA_CHECK_RETURN (cudaMemcpy (&rcv_size, cm->extra_send_offsets + num_of_partitions, sizeof(voff_t), cudaMemcpyDeviceToHost));
	if (rcv_size == 0)
	{
		printf ("CCCCCCCCCcccareful:::::::::: receive size from intra junction update push is 0!!!!!!!!\n");
		rcv_size = 1000;
	}
	cm->temp_size = malloc_pull_push_receive_device (&cm->receive, sizeof(shakehands_t), did, rcv_size, 2*(total_num_partitions+num_of_partitions-1)/num_of_partitions, world_rank, num_of_devices);
	set_receive_buffer_gpu (&cm->receive, did, world_rank, num_of_devices);

	CUDA_CHECK_RETURN (cudaMemset(cm->tmp_send_offsets, 0, sizeof(voff_t) * (total_num_partitions + 1)));

#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&start, NULL);
#endif
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset+i];
		voff_t num_mssgs = mst->roff[did][i+1] - mst->roff[did][i];
		voff_t size = index_offset[i+1] - index_offset[i];
		int num_of_blocks = (num_mssgs + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		push_mssg_respond <<<block_size, THREADS_PER_BLOCK_NODES, sizeof(vid_t)*(total_num_partitions+1)>>> (num_mssgs, pid, size, index_offset[i], total_num_partitions, 0, 1, k, p, cutoff);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	push_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PUSH *RESPOND* GPU INTRA PROCESSOR TIME: ");
#endif

#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&start, NULL);
#endif
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset+i];
		voff_t num_mssgs = mst->soff[did][i+1] - mst->soff[did][i];
		voff_t size = index_offset[i+1] - index_offset[i];
		int num_of_blocks = (num_mssgs + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		push_mssg_respond <<<block_size, THREADS_PER_BLOCK_NODES, sizeof(vid_t)*(total_num_partitions+1)>>> (num_mssgs, pid, size, index_offset[i], total_num_partitions, receive_start, 0, k, p, cutoff);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	push_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PUSH *RESPOND* GPU INTER PROCESSORS TIME: ");
#endif
#ifdef MEASURE_MEMCPY_
	gettimeofday(&start, NULL);
#endif

	CUDA_CHECK_RETURN (cudaMemcpy(mst->roff[did], cm->extra_send_offsets, sizeof(voff_t)*(total_num_partitions + 1), cudaMemcpyDeviceToHost));

#ifdef MEASURE_MEMCPY_
	gettimeofday(&end, NULL);
	memcpydh_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
#endif
	voff_t inter_start = mst->roff[did][num_of_partitions];
	voff_t inter_end = mst->roff[did][total_num_partitions];

	printf ("WORLD RANK %d: @@@@@@@@@@@@@@@@@@@@@ total number of shakehands pushed in device %d: %lu\n", mst->world_rank, did, inter_end);
	printf ("WORLD RANK %d: ############### number of intra mssgs pulled for inter shakehands of device %d: %lu\n", mst->world_rank, did, inter_start);

#ifdef MEASURE_TIME_
	gettimeofday(&start, NULL);
	CUDA_CHECK_RETURN (cudaMemcpy(mst->receive[did], (shakehands_t *)cm->receive+inter_start, (inter_end-inter_start)*sizeof(shakehands_t), cudaMemcpyDeviceToHost));
	gettimeofday(&end, NULL);
	memcpydh_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
#else
//	CUDA_CHECK_RETURN (cudaMemcpyAsync(mst->receive[did], (shakehands_t *)cm->receive+inter_start, (inter_end-inter_start)*sizeof(shakehands_t), cudaMemcpyDeviceToHost, streams[did]));
	CUDA_CHECK_RETURN (cudaMemcpy(mst->receive[did], (shakehands_t *)cm->receive+inter_start, (inter_end-inter_start)*sizeof(shakehands_t), cudaMemcpyDeviceToHost));
#endif

#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&start, NULL);
#endif
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset + i];
		voff_t num_mssgs = mst->roff[did][i+1] - mst->roff[did][i];
		voff_t size = index_offset[i+1] - index_offset[i];
		int num_of_blocks = (num_mssgs + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		pull_mssg_respond <<<block_size, THREADS_PER_BLOCK_NODES>>> (num_mssgs, pid, size, index_offset[i], cm->receive, 1);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	pull_intra_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PULL *RESPOND* GPU %d INTRA PROCESSOR TIME: ", did);
#endif
	gettimeofday (&overe, NULL);
	over_time[did] += (float)((overe.tv_sec * 1000000 + overe.tv_usec) - (overs.tv_sec * 1000000 + overs.tv_usec)) / 1000;

	return ((void *) 0);
}

void * respond_inter_pull_gpu (void * arg)
{
	evaltime_t overs, overe;
	pre_arg * carg = (pre_arg *) arg;
	comm_t * cm = &carg->dbm->comm;
	master_t * mst = carg->mst;
	int did = carg->did;
	int num_of_partitions = mst->num_partitions[did + 1] - mst->num_partitions[did];
	int poffset = mst->num_partitions[did];
	voff_t * index_offset = mst->index_offset[did];
	int num_of_devices = mst->num_of_devices;
	int world_rank = mst->world_rank;

	if (world_rank == 0)
		printf ("WORLD RANK %d: respond inter pull gpu %d:\n", world_rank, did);

#ifdef SINGLE_NODE
	CUDA_CHECK_RETURN(cudaSetDevice (world_rank * num_of_devices + did));
#else
	CUDA_CHECK_RETURN(cudaSetDevice (did + DEVICE_SHIFT));
#endif

	gettimeofday (&overs, NULL);
	voff_t receive_start = mst->roff[did][num_of_partitions];
	voff_t inter_size = mst->soff[did][num_of_partitions];
	printf ("WORLD RANK %d: ############### number of inter mssgs pulled for inter shakehands of device %d: %lu\n", mst->world_rank, did, inter_size);
	if (cm->temp_size <= (inter_size+receive_start)*sizeof(shakehands_t))
	{
		printf("WORLD RANK %d: Error:::::::: malloced receive buffer size smaller than actual receive buffer size!\n", mst->world_rank);
		exit(0);
	}

#ifdef MEASURE_MEMCPY_
	evaltime_t start, end;
	gettimeofday(&start, NULL);
#endif
	CUDA_CHECK_RETURN (cudaMemcpy(cm->receive_offsets, mst->soff[did], sizeof(voff_t) * (num_of_partitions+1), cudaMemcpyHostToDevice));
#ifdef MEASURE_MEMCPY_
	gettimeofday(&end, NULL);
	memcpyhd_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	gettimeofday(&start, NULL);
#endif
	CUDA_CHECK_RETURN (cudaMemcpy((shakehands_t *)(cm->receive) + receive_start, mst->send[did], sizeof(shakehands_t) * inter_size, cudaMemcpyHostToDevice));
#ifdef MEASURE_MEMCPY_
	gettimeofday(&end, NULL);
	memcpyhd_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
#endif

#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&start, NULL);
#endif
	int i;
	for (i=0; i<num_of_partitions; i++)
	{
		if (inter_size == 0)
			break;
		int pid = mst->partition_list[poffset + i];
		voff_t num_mssgs = mst->soff[did][i+1] - mst->soff[did][i];
		voff_t size = index_offset[i+1] - index_offset[i];
		int num_of_blocks = (num_mssgs + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		pull_mssg_respond <<<block_size, THREADS_PER_BLOCK_NODES>>> (num_mssgs, pid, size, index_offset[i], (char*)cm->receive + sizeof(shakehands_t) * receive_start, 0);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	pull_inter_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PULL *RESPOND* GPU INTER PROCESSORS TIME: ");
#endif

	gettimeofday (&overe, NULL);
	over_time[did] += (float)((overe.tv_sec * 1000000 + overe.tv_usec) - (overs.tv_sec * 1000000 + overs.tv_usec)) / 1000;

	// *************** free (send and) receive buffer for pull and push mode
	free_pull_push_receive_device (did, cm, world_rank, num_of_devices);

	return ((void *) 0);
}


void pre_process_dbgraph (int num_of_partitions, int k, int p, dbtable_t * tbs, master_t * mst, subgraph_t * subgraph, d_jvs_t * js, d_lvs_t * ls, int world_size, int world_rank)
{
	float all2all_time = 0;
	evaltime_t start, end;
	evaltime_t overs, overe;
	evaltime_t tmps, tmpe;
	evaltime_t inms, inme;
//	gettimeofday (&start, NULL);
	mst->total_num_partitions = num_of_partitions;	//total_num_of_partitions=input num_of_partitions
	mst->world_size = world_size;
	mst->world_rank = world_rank;

	int np_per_node;
	int np_node;
	get_np_node (&np_per_node, &np_node, num_of_partitions, world_size, world_rank);

	if (mst->world_rank == 0)
		printf ("WORLD RANK %d IIIIIIIIIII initialize distributing partitions: \n", mst->world_rank);

	gettimeofday(&start, NULL);
	init_distribute_partitions (num_of_partitions, mst, world_size);
	get_subgraph_sizes (subgraph, np_node);
	if (mssg_factor == 0)
		mssg_factor = MSSG_FACTOR;
	double unit_vsize = sizeof(assid_t)*(mssg_factor+0.1)+sizeof(kmer_t)+sizeof(vid_t)*EDGE_DIC_SIZE+sizeof(ull)+sizeof(voff_t)*2+sizeof(vid_t);
	distribute_partitions (num_of_partitions, mst, subgraph, uneven, world_size, world_rank, subgraph->total_graph_size, unit_vsize);
	gettimeofday(&end, NULL);
	if (mst->world_rank == 0)
		print_exec_time (start, end, "WORLD RANK %d #################### distributing partitions time: ", mst->world_rank);

	int num_of_cpus = mst->num_of_cpus;
	int num_of_devices = mst->num_of_devices;
	dbmeta_t * dbm = (dbmeta_t *) malloc (sizeof(dbmeta_t) * (num_of_devices + num_of_cpus));

	pthread_t cpu_threads[NUM_OF_CPUS];
	pthread_t gpu_threads[NUM_OF_DEVICES];
	pre_arg arg[NUM_OF_DEVICES + NUM_OF_CPUS];

#ifndef SYNC_ALL2ALL_
	pthread_t comm_thread;
	comm_arg cm_arg;
	cm_arg.mst=mst;
#endif

	uint intra_mssgs[NUM_OF_PROCS*MAX_NUM_ITERATION];
	uint inter_mssgs[NUM_OF_PROCS*MAX_NUM_ITERATION];
	init_mssg_count (intra_mssgs, inter_mssgs);
	int i;
	for (i = 0; i < num_of_devices + num_of_cpus; i++)
	{
		arg[i].did = i;
		arg[i].dbm = &dbm[i];
		arg[i].mst = mst;
		arg[i].k = k;
		arg[i].p = p;
		arg[i].js = js;
		arg[i].ls = ls;
		arg[i].subgraph = subgraph;
#ifndef SYNC_ALL2ALL_
		cm_arg.cm[i] = &dbm[i].comm;
#endif
	}

	//***************** PRE-PROCESSING BEGINS: *****************
	gettimeofday(&overs, NULL);

	uint max_subgraph_size;
	uint max_lsize;
	uint max_jsize;
	int intra_num_of_partitions = mst->num_partitions[num_of_devices + num_of_cpus];
	max_ss = get_max (subgraph->subgraphs, NULL, NULL, &max_subgraph_size, &max_jsize, &max_lsize, intra_num_of_partitions, num_of_partitions);

	gettimeofday (&start, NULL);
	mst->mssg_size = sizeof(assid_t); // IMPORTANT:::::::::: initiate MAXIMUM message size for message buffer
	init_device_filter1 (dbm, mst, max_subgraph_size);
	set_globals_filter1_gpu (dbm, mst);

	init_host_filter2 (dbm, mst, max_subgraph_size);
	gettimeofday (&end, NULL);
	if (mst->world_rank == 0)
		print_exec_time (start, end, "WORLD RANK %d: IIIIIIIIIIIIIIInit filter memory time device1 and host: \n", mst->world_rank);

	init_device_preprocessing (dbm, mst);
	set_globals_preprocessing_gpu (dbm, mst);

	init_host_preprocessing (dbm, mst);
	for (i=num_of_devices; i<num_of_devices+num_of_cpus; i++)
	{
		init_preprocessing_data_cpu (&dbm[i], num_of_partitions);
	}

	// **************** malloc writing offset buffer for pull and push mode
	for (i=0; i<num_of_devices; i++)
	{
		malloc_pull_push_offset_gpu (&dbm[i].comm.extra_send_offsets, mst, i);
		set_extra_send_offsets_gpu (&dbm[i].comm.extra_send_offsets, mst, i);
	}
	for (i=num_of_devices; i<num_of_devices+num_of_cpus; i++)
	{
		malloc_pull_push_offset_cpu(&dbm[i].comm.extra_send_offsets, mst);
	}

	gettimeofday (&start, NULL);
	for (i=0; i<num_of_devices; i++)
	{
//		init_hashtab_data_gpu (i, mst, &dbm[i], tbs);
#ifdef USE_DISK_IO
		init_binary_data_gpu (i, mst, &dbm[i], tbs);
#else
		init_binary_data_gpu_sorted (i, mst, &dbm[i], tbs);
#endif
	}
	for (i=num_of_devices; i<num_of_devices+num_of_cpus; i++)
	{
//		init_hashtab_data_cpu (i, mst, &dbm[i], tbs);
#ifdef USE_DISK_IO
		init_binary_data_cpu (i, mst, &dbm[i], tbs);
#else
		init_binary_data_cpu_sorted (i, mst, &dbm[i], tbs);
#endif
	}
	gettimeofday (&end, NULL);
	if (mst->world_rank == 0)
		print_exec_time (start, end, "WORLD RANK %d: ++++++++++++++++Init hash table data input time: ++++++++++++++++\n", mst->world_rank);
	free_dbgraph_hashtab (num_of_partitions, tbs);

	//************** ADD AN EXTRA STEP:::::::::::: modify edges here: ***************
	gettimeofday (&inms, NULL); // in-memory processing begins
	mst->mssg_size = sizeof(shakehands_t); // IMPORTANT:::::::::: RESET message size for message buffer
	for (i = 0; i < num_of_devices; i++)
	{
		if (pthread_create (&gpu_threads[i], NULL, shakehands_push_respond_intra_push_gpu, &arg[i]) != 0)
		{
			printf ("create thread for shakehands push respond intra push on gpu %d error!\n", i);
		}
	}
	for (i = 0; i < num_of_cpus; i++)
	{
		if (pthread_create (&cpu_threads[i], NULL, shakehands_push_respond_intra_push_cpu, &arg[num_of_devices + i]) != 0)
		{
			printf ("create thread for shakehands push respond intra push on cpu %d error!\n", i);
		}
	}
	for (i = 0; i < num_of_devices; i++)
	{
		if (pthread_join (gpu_threads[i], NULL) != 0)
		{
			printf ("join thread on shakehands push respond intra push on gpu %d failure!\n", i);
		}
	}
	for (i = 0; i < num_of_cpus; i++)
	{
		if (pthread_join (cpu_threads[i], NULL) != 0)
		{
			printf ("Join thread on shakehands push respond intra push on cpu %d failure!\n", i);
		}
	}
	gettimeofday (&start, NULL);
	master_all2all(mst);
	gettimeofday (&end, NULL);
	all2all_time += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	if (mst->world_rank == 0)
		print_exec_time (start, end, "WORLD RANK %d: TICKTOCK TICKTOCK TICKTOCK:: master all to all time after listrank push: ", mst->world_rank);

//	while(debug) {}
	for (i = 0; i < num_of_devices; i++)
	{
		if (pthread_create (&gpu_threads[i], NULL, shakehands_pull_respond_inter_push_intra_pull_gpu, &arg[i]) != 0)
		{
			printf ("create thread for shakehands pull respond inter upsh intra pull on gpu %d error!\n", i);
		}
	}
	for (i = 0; i < num_of_cpus; i++)
	{
		if (pthread_create (&cpu_threads[i], NULL, shakehands_pull_respond_inter_push_intra_pull_cpu, &arg[num_of_devices + i]) != 0)
		{
			printf ("create thread for shakehands pull respond inter push intra pull on cpu %d error!\n", i);
		}
	}
	for (i = 0; i < num_of_devices; i++)
	{
		if (pthread_join (gpu_threads[i], NULL) != 0)
		{
			printf ("join thread on shakehands pull respond inter push intra pull on gpu %d failure!\n", i);
		}
	}
	for (i = 0; i < num_of_cpus; i++)
	{
		if (pthread_join (cpu_threads[i], NULL) != 0)
		{
			printf ("Join thread on shakehands pull respond inter push intra pull on cpu %d failure!\n", i);
		}
	}

	gettimeofday (&start, NULL);
	master_all2all(mst);
	gettimeofday (&end, NULL);
	all2all_time += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;

	if (mst->world_rank == 0)
		print_exec_time (start, end, "WORLD RANK %d: TICKTOCK TICKTOCK TICKTOCK:: master all to all time after listrank pull modifygraph push: ", mst->world_rank);

	for (i = 0; i < num_of_devices; i++)
	{
		if (pthread_create (&gpu_threads[i], NULL, respond_inter_pull_gpu, &arg[i]) != 0)
		{
			printf ("create thread on respond inter pull on gpu %d error!\n", i);
		}
	}
	for (i = 0; i < num_of_cpus; i++)
	{
		if (pthread_create (&cpu_threads[i], NULL, respond_inter_pull_cpu, &arg[num_of_devices + i]) != 0)
		{
			printf ("create thread for respond inter pull on cpu %d error!\n", i);
		}
	}
	for (i = 0; i < num_of_devices; i++)
	{
		if (pthread_join (gpu_threads[i], NULL) != 0)
		{
			printf ("Join thread on respond inter pull on gpu %d error!\n", i);
		}
	}
	for (i = 0; i < num_of_cpus; i++)
	{
		if (pthread_join (cpu_threads[i], NULL) != 0)
		{
			printf ("Join thread on respond inter pull on cpu %d failure!\n", i);
		}
	}
	gettimeofday (&inme, NULL); // in-memory processing end
	inmemory_time += (float)((inme.tv_sec * 1000000 + inme.tv_usec) - (inms.tv_sec * 1000000 + inms.tv_usec)) / 1000;

	 // ************ free writing offset buffer for pull and push mode
	for (i=0; i<num_of_devices; i++)
		free_pull_push_offset_gpu(dbm[i].comm.extra_send_offsets);
	for (i=num_of_devices; i<num_of_devices+num_of_cpus; i++)
		free_pull_push_offset_cpu(dbm[i].comm.extra_send_offsets);

	//************** Reset memory here ************
	gettimeofday (&start, NULL);
	init_device_filter2 (dbm, mst, max_subgraph_size);
	set_globals_filter2_gpu (dbm, mst);
	gettimeofday (&end, NULL);
	if (mst->world_rank == 0)
		print_exec_time (start, end, "WORLD RANK %d: IIIIIIIIIIIIIIInit filter memory time device2: \n", mst->world_rank);

	//*********** FIRST: ASSIGN EACH VERTEX A GLOBAL ID **************
	gettimeofday (&inms, NULL); // in-memory processing begins
	gettimeofday (&start, NULL);
	if (mst->world_rank == 0)
		printf("\n++++++++++++++++ Identifying vertices: WORLD RANK %d ++++++++++++++++++\n", world_rank);
	for (i = 0; i < num_of_devices; i++)
	{
		if (pthread_create (&gpu_threads[i], NULL, identify_vertices_gpu, &arg[i]) != 0)
		{
			printf ("create thread for hashtab filtering on gpu %d error!\n", i);
		}
	}
	for (i = 0; i < num_of_cpus; i++)
	{
		if (pthread_create (&cpu_threads[i], NULL, identify_vertices_cpu, &arg[num_of_devices + i]) != 0)
		{
			printf ("create thread for hashtab filtering on cpu %d error!\n", i);
		}
	}
	for (i = 0; i < num_of_devices; i++)
	{
		if (pthread_join (gpu_threads[i], NULL) != 0)
		{
			printf ("Join thread on hashtab filtering on gpu %d failure!\n", i);
		}
	}
	for (i = 0; i < num_of_cpus; i++)
	{
		if (pthread_join (cpu_threads[i], NULL) != 0)
		{
			printf ("Join thread on hashtab filtering on cpu %d failure!\n", i);
		}
	}

	// ******** allgather id offsets here: ************
	gettimeofday (&tmps, NULL);
	printf ("goffset size: %d\n", sizeof(goffset_t));
	mpi_allgatherv_inplace (mst->id_offsets+1 + world_rank*np_per_node, mst->id_offsets + 1, num_of_partitions, world_size, world_rank, sizeof(goffset_t));
	mpi_allgatherv_inplace (mst->jid_offset + world_rank*np_per_node, mst->jid_offset, num_of_partitions, world_size, world_rank, sizeof(goffset_t));
	gettimeofday (&tmpe, NULL);
	all2all_time += (float)((tmpe.tv_sec * 1000000 + tmpe.tv_usec) - (tmps.tv_sec * 1000000 + tmps.tv_usec)) / 1000;

	inclusive_prefix_sum_long (mst->id_offsets, num_of_partitions + 1);
	ull total_num_junctions = 0;
	for (i=0; i<num_of_partitions; i++)
		total_num_junctions += mst->jid_offset[i];
	junction_factor = (double)total_num_junctions/(mst->id_offsets[num_of_partitions]-total_num_junctions) + 0.01;
	printf ("WORLD RANK %d: TOTAL NUMBER OF JUNCITIONS: %u\nJUNCTION FACTOR SET TO BE::::::::: %f\n", mst->world_rank, total_num_junctions, junction_factor);

	set_id_offsets_cpu (dbm, mst);
	set_id_offsets_gpu (dbm, mst);

	for (i = 0; i < num_of_devices; i++)
	{
		if (pthread_create (&gpu_threads[i], NULL, assign_vertex_ids_gpu, &arg[i]) != 0)
		{
			printf ("create thread for assigning vertex ids on gpu %d error!\n", i);
		}
	}
	for (i = 0; i < num_of_cpus; i++)
	{
		if (pthread_create (&cpu_threads[i], NULL, assign_vertex_ids_cpu, &arg[num_of_devices + i]) != 0)
		{
			printf ("create thread for assigning vertex ids on cpu %d error!\n", i);
		}
	}
	for (i = 0; i < num_of_devices; i++)
	{
		if (pthread_join (gpu_threads[i], NULL) != 0)
		{
			printf ("Join thread on assigning vertex ids on gpu %d failure!\n", i);
		}
	}
	for (i = 0; i < num_of_cpus; i++)
	{
		if (pthread_join (cpu_threads[i], NULL) != 0)
		{
			printf ("Join thread on assigning vertex ids on cpu %d failure!\n", i);
		}
	}
	gettimeofday (&end, NULL);
	gettimeofday (&inme, NULL); // inmemory processing end
	inmemory_time += (float)((inme.tv_sec * 1000000 + inme.tv_usec) - (inms.tv_sec * 1000000 + inms.tv_usec)) / 1000;
	if (mst->world_rank == 0)
		print_exec_time (start, end, "WORLD RANK %d: ++++++++++++++++ Identifying vertex time: ++++++++++++++++++\n", mst->world_rank);

	gettimeofday (&start, NULL);
	finalize_device_filter2 (dbm, mst);
	set_globals_filter2_gpu (dbm, mst);

	finalize_host_filter2 (dbm, mst);
	gettimeofday (&end, NULL);
	if (mst->world_rank == 0)
		print_exec_time (start, end, "WORLD RANK %d: FFFFFFFFFFFfinalize device and host filter time:\n", mst->world_rank);


	//*********** SECOND: NEIGHBORING WITH VERTEX IDS ****************
	gettimeofday(&start, NULL);
	mst->mssg_size = sizeof(assid_t);  // IMPORTANT:::::::::: RESET message size for message buffer

	gettimeofday(&end, NULL);
	if (mst->world_rank == 0)
		print_exec_time (start, end, "WORLD RANK %d: IIIIIIIIIIIIIIInitializing pre-processing time: \n", mst->world_rank);

	// **************** STEP 1
	gettimeofday (&inms, NULL); // in-memory processing begins
	gettimeofday (&start, NULL);
	if (mst->world_rank == 0)
		printf("\n++++++++++++++++ Identifying neighbors: WORLD RANK %d ++++++++++++++++++\n", world_rank);
	for (i = 0; i < num_of_devices; i++)
	{
		if (pthread_create (&gpu_threads[i], NULL, neighbor_push_intra_pull_gpu, &arg[i]) != 0)
		{
			printf ("create thread for NEIGHBORING push on gpu %d error!\n", i);
		}
	}
	for (i = 0; i < num_of_cpus; i++)
	{
		if (pthread_create (&cpu_threads[i], NULL, neighbor_push_intra_pull_cpu, &arg[num_of_devices + i]) != 0)
		{
			printf ("create thread for NEIGHBORING push on cpu %d error!\n", i);
		}
	}
	for (i = 0; i < num_of_devices; i++)
	{
		if (pthread_join (gpu_threads[i], NULL) != 0)
		{
			printf ("Join thread on NEIGHBORING push on gpu %d failure!\n", i);
		}
	}
	for (i = 0; i < num_of_cpus; i++)
	{
		if (pthread_join (cpu_threads[i], NULL) != 0)
		{
			printf ("Join thread on NEIGHBORING push on cpu %d failure!\n", i);
		}
	}

	gettimeofday (&tmps, NULL);
	master_all2all(mst);
	gettimeofday (&tmpe, NULL);
	all2all_time += (float)((tmpe.tv_sec * 1000000 + tmpe.tv_usec) - (tmps.tv_sec * 1000000 + tmps.tv_usec)) / 1000;
//	get_mssg_count(mst, intra_mssgs, inter_mssgs, 0);

	for (i = 0; i < num_of_devices; i++)
	{
		if (pthread_create (&gpu_threads[i], NULL, neighbor_inter_pull_gpu, &arg[i]) != 0)
		{
			printf ("create thread for NEIGHBORING pull on gpu %d error!\n", i);
		}
	}
	for (i = 0; i < num_of_cpus; i++)
	{
		if (pthread_create (&cpu_threads[i], NULL, neighbor_inter_pull_cpu, &arg[num_of_devices + i]) != 0)
		{
			printf ("create thread for NEIGHBORING pull on cpu %d error!\n", i);
		}
	}
	for (i = 0; i < num_of_devices; i++)
	{
		if (pthread_join (gpu_threads[i], NULL) != 0)
		{
			printf ("Join thread on NEIGHBORING pull on gpu %d failure!\n", i);
		}
	}
	for (i = 0; i < num_of_cpus; i++)
	{
		if (pthread_join (cpu_threads[i], NULL) != 0)
		{
			printf ("Join thread on NEIGHBORING pull on cpu %d failure!\n", i);
		}
	}
	gettimeofday (&end, NULL);
	gettimeofday (&inme, NULL); // in-memory processing end
	inmemory_time += (float)((inme.tv_sec * 1000000 + inme.tv_usec) - (inms.tv_sec * 1000000 + inms.tv_usec)) / 1000;

	if (mst->world_rank == 0)
		print_exec_time (start, end, "WORLD RANK %d: ++++++++++++++++Neighboring time: ++++++++++++++++++\n", mst->world_rank);


	gettimeofday(&start, NULL);
	finalize_device_preprocessing (dbm, mst);
	finalize_host_preprocessing (dbm, mst);
	finalize_preprocessing_data_cpu ();
	finalize_receive_all2all(world_size);
	gettimeofday(&end, NULL);
	if (mst->world_rank == 0)
		print_exec_time (start, end, "WORLD RANK %d: FFFFFFFFFFFFFFFFFFFinalizing pre-processing time: \n", mst->world_rank);

	//*********** THIRD: GARTHER VERTICES INTO VERTEX ARRAYS *************

	gettimeofday (&start, NULL);
	get_max (subgraph->subgraphs, mst->jid_offset, mst->id_offsets, &max_subgraph_size, &max_jsize, &max_lsize, intra_num_of_partitions, num_of_partitions);

	if (mst->world_rank == 0)
		printf ("WORLD RANK %d: max subgraph size: %u, max junction size: %u, max linear vertex size: %u\n", world_rank, max_subgraph_size, max_jsize, max_lsize);
	gmax_jsize = max_jsize;
	gmax_lsize = max_lsize;
	init_write_buffer (num_of_devices);
	init_device_gather (dbm, mst, max_subgraph_size, max_jsize, max_lsize);
	set_globals_gather_gpu (dbm, mst);

	init_host_gather (dbm, mst, max_subgraph_size, max_jsize, max_lsize);
	for (i=num_of_devices; i<num_of_devices+num_of_cpus; i++)
	{
		reset_globals_gather_cpu (&dbm[i]);
	}
	gettimeofday (&end, NULL);
	if (mst->world_rank == 0)
		print_exec_time (start, end, "WORLD RANK %d: IIIIIIIIIIIIIIInitializing gathering time: \n", world_rank);

	gettimeofday (&inms, NULL); // in-memory processing begins
	gettimeofday (&start, NULL);
	if (mst->world_rank == 0)
		printf("\n++++++++++++++++ Gather vertices: WORLD RANK %d ++++++++++++++++++\n", world_rank);
	for (i = 0; i < num_of_devices; i++)
	{
		if (pthread_create (&gpu_threads[i], NULL, gather_vertices_gpu, &arg[i]) != 0)
		{
			printf ("create thread for hashtab filtering on gpu %d error!\n", i);
		}
	}
	for (i = 0; i < num_of_cpus; i++)
	{
		if (pthread_create (&cpu_threads[i], NULL, gather_vertices_cpu, &arg[num_of_devices + i]) != 0)
		{
			printf ("create thread for hashtab filtering on cpu %d error!\n", i);
		}
	}
	for (i = 0; i < num_of_devices; i++)
	{
		if (pthread_join (gpu_threads[i], NULL) != 0)
		{
			printf ("Join thread on hashtab filtering on gpu %d failure!\n", i);
		}
	}
	for (i = 0; i < num_of_cpus; i++)
	{
		if (pthread_join (cpu_threads[i], NULL) != 0)
		{
			printf ("Join thread on hashtab filtering on cpu %d failure!\n", i);
		}
	}
	gettimeofday (&end, NULL);
	gettimeofday (&inme, NULL); // in-memory processing end
	inmemory_time += (float)((inme.tv_sec * 1000000 + inme.tv_usec) - (inms.tv_sec * 1000000 + inms.tv_usec)) / 1000;

	if (mst->world_rank == 0)
		print_exec_time (start, end, "WORLD RANK %d: ++++++++++++++++Gathering vertex array time: ++++++++++++++++++\n", mst->world_rank);

	gettimeofday (&start, NULL);
	finalize_device_gather2 (dbm, mst);
	finalize_host_gather2 (dbm, mst);

	finalize_distribute_partitions (mst);
//	print_mssg_count(mst->num_of_cpus+mst->num_of_devices, intra_mssgs, inter_mssgs, 0);
	gettimeofday (&end, NULL);
	if (mst->world_rank == 0)
		print_exec_time (start, end, "WORLD RANK %d: FFFFFFFFFFFFFFFFFFFinalizing gathering time: \n", mst->world_rank);

	gettimeofday(&overe, NULL);
	print_exec_time(overs, overe, "WORLD RANK %d: ***********************Overall PRE-PROCESSING time: \n", mst->world_rank);

	finalize_write_buffer (num_of_devices);
	// print graph statistics:
	uint total_num_nodes = mst->id_offsets[num_of_partitions];
	if (mst->world_rank==0)
	{
	printf ("WORLD RANK %d: &&&&&&&&&&&&&&& Total number of valid nodes in de bruijn graph: %lu\n", mst->world_rank, total_num_nodes);
	printf("WORLD RANK %d:TTTTTTTTTTTTTTTTTTTTIMING: within that:\n", mst->world_rank);
	printf ("WORLD RANK %d:~~~~~~~~~~~~~~~~ ALLTOALL TIME MEASURED: %f\n", mst->world_rank, all2all_time);
	}

	free (dbm);
}

}
