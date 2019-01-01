/*
 * comm.cu
 *
 *  Created on: 2017-7-14
 *      Author: qiushuang
 */

#define USE_COMM_

//#define SINGLE_NODE
//#include <cub/cub.cuh>
#include <pthread.h>
#include <math.h>
#include <omp.h>
#include "include/dbgraph.h"
#include "include/comm.h"
#include "include/io.h"
#include "include/share.h"
#include "malloc.cuh"
#include "comm.cuh"
#include "include/distribute.h"
#include "include/scan.cu"

//#include "sort.cu"
//#define USE_CUB_SORT_

#define CONSTRUCT_BINARY_THREADS 1024
#define NUM_TRAVERSAL_THREADS 24
#define THRESHOLD 100
#define MEAN_CONTIG 300

extern double junction_factor;

#ifdef MEASURE_TIME_
extern float push_offset_time[NUM_OF_PROCS];
extern float push_time[NUM_OF_PROCS];
extern float pull_intra_time[NUM_OF_PROCS];
extern float pull_inter_time[NUM_OF_PROCS];
extern float memcpydh_time[NUM_OF_PROCS];
extern float memcpyhd_time[NUM_OF_PROCS];
#endif

extern float all2all_time_async;
extern float over_time[NUM_OF_PROCS];
extern float comm_time;
extern float inmemory_time;

extern int debug;
extern uint dst_not_found;
extern uint selfloop_made;
extern int lock_flag[NUM_OF_PROCS];

static uint maxx = 0;

extern "C"
{
void * listrank_push_gpu (void * arg)
{
	lr_arg * garg = (lr_arg *) arg;
	int did = garg->did;
	comm_t * cm = garg->cm;
	master_t * mst = garg->mst;
	if (mst->world_rank == 0)
		printf ("WORLD RANK %d: listrank push gpu %d:\n", mst->world_rank, did);

#ifdef SINGLE_NODE
	int num_of_devices = mst->num_of_devices;
	int world_rank = mst->world_rank;
	CUDA_CHECK_RETURN(cudaSetDevice (world_rank * num_of_devices + did));
#else
	CUDA_CHECK_RETURN(cudaSetDevice (did + DEVICE_SHIFT));
#endif
	int total_num_partitions = mst->total_num_partitions;
	int num_of_partitions = mst->num_partitions[did + 1] - mst->num_partitions[did];
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
		int poffset = mst->num_partitions[did];
		int pid = mst->partition_list[poffset+i];
		voff_t size = mst->id_offsets[pid+1] - mst->id_offsets[pid] - mst->jid_offset[pid];
		int num_of_blocks = (size + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		push_mssg_offset_lr <<<block_size, THREADS_PER_BLOCK_NODES, sizeof(vid_t)*(total_num_partitions+1)>>> (size, total_num_partitions, pid, index_offset[i]);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PUSH MSSG OFFSET FOR GPU LISTRANKING INTRA PROCESSOR TIME: ");
#endif

	inclusive_scan ((int *)(cm->send_offsets), total_num_partitions + 1, NULL);

	CUDA_CHECK_RETURN (cudaMemset(cm->tmp_send_offsets, 0, sizeof(voff_t) * (total_num_partitions + 1)));

#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&start, NULL);
#endif

	for (i=0; i< num_of_partitions; i++)
	{
		int poffset = mst->num_partitions[did];
		int pid = mst->partition_list[poffset+i];
		voff_t size = mst->id_offsets[pid+1] - mst->id_offsets[pid] - mst->jid_offset[pid];
		int num_of_blocks = (size + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		push_mssg_lr <<<block_size, THREADS_PER_BLOCK_NODES, sizeof(vid_t)*(total_num_partitions+1)>>> (size, total_num_partitions, pid, index_offset[i]);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PUSH MSSG FOR GPU LISTRANKING INTRA PROCESSOR TIME: ");
#endif

	CUDA_CHECK_RETURN (cudaMemcpy(mst->roff[did], cm->send_offsets, sizeof(voff_t) * (total_num_partitions + 1), cudaMemcpyDeviceToHost));
	voff_t inter_start = mst->roff[did][num_of_partitions];
	voff_t inter_end = mst->roff[did][total_num_partitions];
	CUDA_CHECK_RETURN (cudaMemcpy(mst->receive[did], (path_t *)cm->send + inter_start, sizeof(path_t) * (inter_end-inter_start), cudaMemcpyDeviceToHost));

	return ((void *) 0);
}

void * listrank_pull_gpu (void * arg)
{
	lr_arg * garg = (lr_arg *) arg;
	int did = garg->did;
	comm_t * cm = garg->cm;
	master_t * mst = garg->mst;
	if (mst->world_rank == 0)
		printf ("WORLD_RANK %d: listrank pull gpu %d:\n", mst->world_rank, did);

#ifdef SINGLE_NODE
	int world_rank = mst->world_rank;
	int num_of_devices = mst->num_of_devices;
	CUDA_CHECK_RETURN(cudaSetDevice (num_of_devices * world_rank + did));
#else
	CUDA_CHECK_RETURN(cudaSetDevice(did + DEVICE_SHIFT));
#endif
	int num_of_partitions = mst->num_partitions[did + 1] - mst->num_partitions[did];
	voff_t * index_offset = mst->index_offset[did];

#ifdef MEASURE_TIME_
	evaltime_t start, end;
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&start, NULL);
#endif

	int i;
	for (i=0; i<num_of_partitions; i++)
	{
		int poffset = mst->num_partitions[did];
		int pid = mst->partition_list[poffset+i];
		voff_t num_mssgs = mst->roff[did][i+1] - mst->roff[did][i];
		int num_of_blocks = (num_mssgs + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		pull_mssg_lr <<<block_size, THREADS_PER_BLOCK_NODES>>> (num_mssgs, pid, index_offset[i], 0, 1);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PULL MSSG FOR GPU LISTRANKING INTRA PROCESSOR TIME: ");
#endif

	voff_t receive_start = mst->roff[did][num_of_partitions];
	CUDA_CHECK_RETURN (cudaMemcpy(cm->receive_offsets, mst->soff[did], sizeof(voff_t) * (num_of_partitions + 1), cudaMemcpyHostToDevice));
	voff_t inter_size = mst->soff[did][num_of_partitions];
	CUDA_CHECK_RETURN (cudaMemcpy((path_t*)cm->send + receive_start, mst->send[did], sizeof(path_t) * inter_size, cudaMemcpyHostToDevice));
	printf ("&&&&&&&&&& listrank pull gpu %d: receive_offsets\n", did);
//	print_offsets (mst->soff[did], num_of_partitions+1);

#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&start, NULL);
#endif

	for (i=0; i<num_of_partitions; i++)
	{
		int poffset = mst->num_partitions[did];
		int pid = mst->partition_list[poffset+i];
		voff_t num_mssgs = mst->soff[did][i+1] - mst->soff[did][i];
		int num_of_blocks = (num_mssgs + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		pull_mssg_lr <<<block_size, THREADS_PER_BLOCK_NODES>>> (num_mssgs, pid, index_offset[i], receive_start, 0);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PULL MSSG FOR GPU LISTRANKING INTER PROCESSORS TIME: ");
#endif

	return ((void *) 0);
}

void * listrank_push_intra_pull_gpu (void * arg)
{
	evaltime_t overs, overe;
	lr_arg * garg = (lr_arg *) arg;
	int did = garg->did;
	comm_t * cm = garg->cm;
	master_t * mst = garg->mst;
	if (mst->world_rank == 0)
		printf ("WORLD_RANK %d: listrank push gpu %d:\n", mst->world_rank, did);

#ifdef SINGLE_NODE
	int world_rank = mst->world_rank;
	int num_of_devices = mst->num_of_devices;
	CUDA_CHECK_RETURN(cudaSetDevice (world_rank * num_of_devices + did));
#else
	CUDA_CHECK_RETURN(cudaSetDevice (did + DEVICE_SHIFT));
#endif
	gettimeofday (&overs, NULL);
	int total_num_partitions = mst->total_num_partitions;
	int num_of_partitions = mst->num_partitions[did + 1] - mst->num_partitions[did];
	voff_t * index_offset = mst->index_offset[did];
	int i;

	CUDA_CHECK_RETURN (cudaMemset(cm->send_offsets, 0, sizeof(voff_t) * (total_num_partitions + 1)));

#ifdef MEASURE_TIME_
	evaltime_t start, end;
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&start, NULL);
#endif
	for (i=0; i<num_of_partitions; i++)
	{
		int poffset = mst->num_partitions[did];
		int pid = mst->partition_list[poffset+i];
		voff_t size = mst->id_offsets[pid+1] - mst->id_offsets[pid] - mst->jid_offset[pid];
		int num_of_blocks = (size + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		push_mssg_offset_lr_async <<<block_size, THREADS_PER_BLOCK_NODES, sizeof(vid_t)*(total_num_partitions+1)>>> (size, total_num_partitions, pid, index_offset[i]);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	push_offset_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PUSH MSSG OFFSET FOR GPU LISTRANKING INTRA PROCESSOR TIME: ");
#endif

	inclusive_scan ((int *)(cm->send_offsets), total_num_partitions + 1, NULL);

	CUDA_CHECK_RETURN (cudaMemset(cm->tmp_send_offsets, 0, sizeof(voff_t) * (total_num_partitions + 1)));

#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&start, NULL);
#endif

	for (i=0; i< num_of_partitions; i++)
	{
		int poffset = mst->num_partitions[did];
		int pid = mst->partition_list[poffset+i];
		voff_t size = mst->id_offsets[pid+1] - mst->id_offsets[pid] - mst->jid_offset[pid];
		int num_of_blocks = (size + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		push_mssg_lr_async <<<block_size, THREADS_PER_BLOCK_NODES, sizeof(vid_t)*(total_num_partitions+1)>>> (size, total_num_partitions, pid, index_offset[i]);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	push_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PUSH MSSG FOR GPU LISTRANKING INTRA PROCESSOR TIME: ");
#endif

	CUDA_CHECK_RETURN (cudaMemcpy(mst->roff[did], cm->send_offsets, sizeof(voff_t) * (total_num_partitions + 1), cudaMemcpyDeviceToHost));

	voff_t inter_start = mst->roff[did][num_of_partitions];
	voff_t inter_end = mst->roff[did][total_num_partitions];

#ifndef SYNC_ALL2ALL_
	if (atomic_set_value(&lock_flag[did], 1, 0) == false)
		printf ("!!!!!!!!!!! CAREFUL: ATOMIC SET VALUE ERROR IN GPU %d\n", did);
#endif
#ifdef MEASURE_MEMCPY_
	gettimeofday(&start, NULL);
	CUDA_CHECK_RETURN (cudaMemcpy(mst->receive[did], (path_t *)cm->send + inter_start, sizeof(path_t) * (inter_end-inter_start), cudaMemcpyDeviceToHost));
	gettimeofday(&end, NULL);
	memcpydh_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
#else
//	CUDA_CHECK_RETURN (cudaMemcpyAsync(mst->receive[did], (path_t *)cm->send + inter_start, sizeof(path_t) * (inter_end-inter_start), cudaMemcpyDeviceToHost, streams[did]));
	CUDA_CHECK_RETURN (cudaMemcpy(mst->receive[did], (path_t *)cm->send + inter_start, sizeof(path_t) * (inter_end-inter_start), cudaMemcpyDeviceToHost));
#endif
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&start, NULL);
#endif

/*	printf ("WORLD RANK %d::::::: RRRRRRRRRRRECHECK INDEX_OFFSETS::::::::", did);
	print_offsets (index_offset, num_of_partitions+1);

	printf ("WORLD RANK %d::::: TTTTTTTTTTTTTSET RECEIVE OFFSETS::::::::::::::", mst->world_rank);
	print_offsets (mst->roff[did], total_num_partitions+1);*/

	for (i=0; i<num_of_partitions; i++)
	{
		int poffset = mst->num_partitions[did];
		int pid = mst->partition_list[poffset+i];
		voff_t num_mssgs = mst->roff[did][i+1] - mst->roff[did][i];
		int num_of_blocks = (num_mssgs + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		pull_mssg_lr_async <<<block_size, THREADS_PER_BLOCK_NODES>>> (num_mssgs, pid, index_offset[i], 0, 1);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	pull_intra_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PULL MSSG FOR GPU LISTRANKING INTRA PROCESSOR TIME: ");
#endif
	gettimeofday (&overe, NULL);
	over_time[did] += (float)((overe.tv_sec * 1000000 + overe.tv_usec) - (overs.tv_sec * 1000000 + overs.tv_usec)) / 1000;

	return ((void *) 0);
}

void * listrank_inter_pull_gpu (void * arg)
{
	evaltime_t overs, overe;
	lr_arg * garg = (lr_arg *) arg;
	int did = garg->did;
	comm_t * cm = garg->cm;
	master_t * mst = garg->mst;
	if (mst->world_rank == 0)
		printf ("WORLD RANK %d: listrank pull gpu %d:\n", mst->world_rank, did);

#ifdef SINGLE_NODE
	int num_of_devices = mst->num_of_devices;
	int world_rank = mst->world_rank;
	CUDA_CHECK_RETURN(cudaSetDevice (world_rank * num_of_devices + did));
#else
	CUDA_CHECK_RETURN(cudaSetDevice (did + DEVICE_SHIFT));
#endif
	gettimeofday (&overs, NULL);
	int num_of_partitions = mst->num_partitions[did + 1] - mst->num_partitions[did];
	voff_t * index_offset = mst->index_offset[did];
	int i;

	voff_t receive_start = mst->roff[did][num_of_partitions];
#ifdef MEASURE_MEMCPY_
	evaltime_t start, end;
	gettimeofday(&start, NULL);
#endif
	CUDA_CHECK_RETURN (cudaMemcpy(cm->receive_offsets, mst->soff[did], sizeof(voff_t) * (num_of_partitions + 1), cudaMemcpyHostToDevice));
	voff_t inter_size = mst->soff[did][num_of_partitions];
	if (inter_size == 0)
		return ((void *) 0);
	CUDA_CHECK_RETURN (cudaMemcpy((path_t*)cm->send + receive_start, mst->send[did], sizeof(path_t) * inter_size, cudaMemcpyHostToDevice));
#ifdef MEASURE_MEMCPY_
	gettimeofday(&end, NULL);
	memcpyhd_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
#endif

#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&start, NULL);
#endif

	for (i=0; i<num_of_partitions; i++)
	{
		int poffset = mst->num_partitions[did];
		int pid = mst->partition_list[poffset+i];
		voff_t num_mssgs = mst->soff[did][i+1] - mst->soff[did][i];
		int num_of_blocks = (num_mssgs + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		pull_mssg_lr_async <<<block_size, THREADS_PER_BLOCK_NODES>>> (num_mssgs, pid, index_offset[i], receive_start, 0);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	pull_inter_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PULL MSSG FOR GPU LISTRANKING INTER PROCESSORS TIME: ");
#endif
	gettimeofday (&overe, NULL);
	over_time[did] += (float)((overe.tv_sec * 1000000 + overe.tv_usec) - (overs.tv_sec * 1000000 + overs.tv_usec)) / 1000;

	return ((void *) 0);
}

/*transfer subgraphs from host to device, set initiated values for vertices and paths*/
void init_graph_data_gpu (int did, master_t * mst, meta_t * dm, d_lvs_t * lvs)
{
	int * num_partitions = mst->num_partitions;
	int * partition_list = mst->partition_list;
	int num_of_partitions = num_partitions[did+1]-num_partitions[did];
	int total_num_partitions = mst->total_num_partitions;
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
	voff_t offset = 0;
	for (i=0; i<num_of_partitions; i++)
	{
		int poffset = num_partitions[did];
		int pid = partition_list[poffset+i] - start_partition_id;
		voff_t size = lvs[pid].asize + lvs[pid].esize;
		CUDA_CHECK_RETURN (cudaMemcpy(dm->edge.post+offset, lvs[pid].posts, sizeof(vid_t)*size, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN (cudaMemcpy(dm->edge.pre+offset, lvs[pid].pres, sizeof(vid_t)*size, cudaMemcpyHostToDevice));
		int num_of_blocks = (size + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		init_lr <<<block_size, THREADS_PER_BLOCK_NODES>>> (size, total_num_partitions, pid + start_partition_id, offset);
		index_offset[i] = offset;
		offset += size;
		if (size > maxx)
			maxx=size;
	}
	index_offset[i] = offset;
//	printf ("index offset on GPU %d: \n", did);
//	print_offsets (index_offset, num_of_partitions+1);
}

void * listrank_push_modifygraph_intra_push_gpu (void * arg)
{
	evaltime_t overs, overe;
	lr_arg * garg = (lr_arg *) arg;
	int did = garg->did;
	comm_t * cm = garg->cm;
	master_t * mst = garg->mst;
	if (mst->world_rank == 0)
		printf ("WORLD RANK %d: listrank push modifygraph intra push gpu %d:\n", mst->world_rank, did);

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
		int poffset = mst->num_partitions[did];
		int pid = mst->partition_list[poffset+i];
		voff_t size = mst->id_offsets[pid+1] - mst->id_offsets[pid] - mst->jid_offset[pid];
		int num_of_blocks = (size + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		push_mssg_offset_lr <<<block_size, THREADS_PER_BLOCK_NODES, sizeof(vid_t)*(total_num_partitions+1)>>> (size, total_num_partitions, pid, index_offset[i]);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	push_offset_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PUSH MSSG OFFSET FOR GPU LISTRANKING INTRA PROCESSOR TIME: ");
#endif

	inclusive_scan ((int *)(cm->send_offsets), total_num_partitions + 1, NULL);

	CUDA_CHECK_RETURN (cudaMemset(cm->tmp_send_offsets, 0, sizeof(voff_t) * (total_num_partitions + 1)));

#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&start, NULL);
#endif
	for (i=0; i< num_of_partitions; i++)
	{
		int poffset = mst->num_partitions[did];
		int pid = mst->partition_list[poffset+i];
		voff_t size = mst->id_offsets[pid+1] - mst->id_offsets[pid] - mst->jid_offset[pid];
		int num_of_blocks = (size + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		push_mssg_lr <<<block_size, THREADS_PER_BLOCK_NODES, sizeof(vid_t)*(total_num_partitions+1)>>> (size, total_num_partitions, pid, index_offset[i]);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	push_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PUSH MSSG FOR GPU LISTRANKING INTRA PROCESSOR TIME: ");
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
	CUDA_CHECK_RETURN (cudaMemcpy(mst->receive[did], (path_t *)cm->send + inter_start, sizeof(path_t) * (inter_end-inter_start), cudaMemcpyDeviceToHost));
	gettimeofday(&end, NULL);
	memcpydh_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
#else
	CUDA_CHECK_RETURN (cudaMemcpyAsync(mst->receive[did], (path_t *)cm->send + inter_start, sizeof(path_t) * (inter_end-inter_start), cudaMemcpyDeviceToHost, streams[did]));
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
		push_selfloop_offset <<<block_size, THREADS_PER_BLOCK_NODES>>> (num_mssgs, pid, index_offset[i], total_num_partitions, 0, 1);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	pull_intra_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PUSH SELFLOOP OFFSET GPU INTRA PROCESSOR TIME: ");
#endif
	gettimeofday (&overe, NULL);
	over_time[did] += (float)((overe.tv_sec * 1000000 + overe.tv_usec) - (overs.tv_sec * 1000000 + overs.tv_usec)) / 1000;

	return ((void *) 0);
}

void * listrank_pull_modifygraph_push_gpu (void * arg)
{
	lr_arg * carg = (lr_arg *) arg;
	int did = carg->did;
	comm_t * cm = carg->cm;
	master_t * mst = carg->mst;
	int num_of_devices = mst->num_of_devices;
	int world_rank = mst->world_rank;
	if (world_rank == 0)
		printf ("WORLD RANK %d, listrank pull modifygraph push gpu %d\n", world_rank, did);

#ifdef SINGLE_NODE
	CUDA_CHECK_RETURN(cudaSetDevice (world_rank * num_of_devices + did));
#else
	CUDA_CHECK_RETURN(cudaSetDevice (did + DEVICE_SHIFT));
#endif
	int total_num_partitions = mst->total_num_partitions;
	int num_of_partitions = mst->num_partitions[did + 1] - mst->num_partitions[did];
	int poffset = mst->num_partitions[did];
	voff_t * index_offset = mst->index_offset[did];
	CUDA_CHECK_RETURN (cudaMemset (cm->extra_send_offsets, 0, sizeof(voff_t) * (total_num_partitions+1)));

	voff_t receive_start = mst->roff[did][num_of_partitions];
#ifdef MEASURE_TIME_
	evaltime_t start, end;
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&start, NULL);
#endif
	int i;
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset + i];
		voff_t num_mssgs = mst->roff[did][i+1] - mst->roff[did][i];
		int num_of_blocks = (num_mssgs + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		push_selfloop_offset <<<block_size, THREADS_PER_BLOCK_NODES>>> (num_mssgs, pid, index_offset[i], total_num_partitions, 0, 1);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PUSH SELFLOOP OFFSET GPU INTRA PROCESSOR TIME: ");
#endif

	CUDA_CHECK_RETURN (cudaMemcpy(cm->receive_offsets, mst->soff[did], sizeof(voff_t) * (num_of_partitions + 1), cudaMemcpyHostToDevice));
	voff_t inter_size = mst->soff[did][num_of_partitions];
	CUDA_CHECK_RETURN (cudaMemcpy((path_t*)(cm->send) + receive_start, mst->send[did], sizeof(path_t) * inter_size, cudaMemcpyHostToDevice));

#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&start, NULL);
#endif
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset + i];
		voff_t num_mssgs = mst->soff[did][i+1] - mst->soff[did][i];
		int num_of_blocks = (num_mssgs + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		push_selfloop_offset <<<block_size, THREADS_PER_BLOCK_NODES>>> (num_mssgs, pid, index_offset[i], total_num_partitions, receive_start, 0);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PUSH SELFLOOP OFFSET GPU INTER PROCESSORS TIME: ");
#endif

	inclusive_scan ((int *)(cm->extra_send_offsets), total_num_partitions + 1, NULL);

	// *************** malloc (send and) receive buffer for pull and push mode
	voff_t rcv_size;
	CUDA_CHECK_RETURN (cudaMemcpy (&rcv_size, cm->extra_send_offsets + num_of_partitions, sizeof(voff_t), cudaMemcpyDeviceToHost));
	if (rcv_size == 0)
	{
		printf ("CCCCCCCCCcccareful:::::::::: receive size from intra selfloop push is 0!!!!!!!!\n");
		rcv_size = 200;
	}
	cm->temp_size = malloc_pull_push_receive_device (&cm->receive, sizeof(selfloop_t), did, rcv_size, 2*(total_num_partitions+num_of_partitions-1)/num_of_partitions, world_rank, num_of_devices);
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
		int num_of_blocks = (num_mssgs + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		push_selfloop <<<block_size, THREADS_PER_BLOCK_NODES>>> (num_mssgs, pid, index_offset[i], total_num_partitions, 0, 1);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PUSH SELFOOLP GPU INTRA PROCESSOR TIME: ");
#endif

#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&start, NULL);
#endif
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset+i];
		voff_t num_mssgs = mst->soff[did][i+1] - mst->soff[did][i];
		int num_of_blocks = (num_mssgs + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		push_selfloop <<<block_size, THREADS_PER_BLOCK_NODES>>> (num_mssgs, pid, index_offset[i], total_num_partitions, receive_start, 0);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PUSH SELFOOLP GPU INTER PROCESSORS TIME: ");
#endif

	CUDA_CHECK_RETURN (cudaMemcpy(mst->roff[did], cm->extra_send_offsets, sizeof(voff_t)*(total_num_partitions + 1), cudaMemcpyDeviceToHost));
	voff_t inter_start = mst->roff[did][num_of_partitions];
	voff_t inter_end = mst->roff[did][total_num_partitions];
	CUDA_CHECK_RETURN (cudaMemcpy(mst->receive[did], (selfloop_t *)cm->receive+inter_start, (inter_end-inter_start)*sizeof(selfloop_t), cudaMemcpyDeviceToHost));

	return ((void *) 0);
}

void * listrank_pull_modifygraph_inter_push_intra_pull_gpu (void * arg)
{
	evaltime_t overs, overe;
	lr_arg * carg = (lr_arg *) arg;
	int did = carg->did;
	comm_t * cm = carg->cm;
	master_t * mst = carg->mst;
	int num_of_devices = mst->num_of_devices;
	int world_rank = mst->world_rank;
	if (world_rank == 0)
		printf ("WORLD RANK %d: listrank pull modifygraph push gpu %d\n", world_rank, did);

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
//	CUDA_CHECK_RETURN (cudaMemset (cm->extra_send_offsets, 0, sizeof(voff_t) * (total_num_partitions+1)));

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
	CUDA_CHECK_RETURN (cudaMemcpy((path_t*)(cm->send) + receive_start, mst->send[did], sizeof(path_t) * inter_size, cudaMemcpyHostToDevice));
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
		int num_of_blocks = (num_mssgs + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		push_selfloop_offset <<<block_size, THREADS_PER_BLOCK_NODES>>> (num_mssgs, pid, index_offset[i], total_num_partitions, receive_start, 0);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	pull_inter_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PUSH SELFLOOP OFFSET GPU INTER PROCESSORS TIME: ");
#endif

	inclusive_scan ((int *)(cm->extra_send_offsets), total_num_partitions + 1, NULL);

	// *************** malloc (send and) receive buffer for pull and push mode
	voff_t rcv_size;
	CUDA_CHECK_RETURN (cudaMemcpy (&rcv_size, cm->extra_send_offsets + num_of_partitions, sizeof(voff_t), cudaMemcpyDeviceToHost));
	if (rcv_size == 0)
	{
		printf ("WORD_RANK %d:GPU::::::::::CCCCCCCCCcccareful:::::::::: receive size from intra selfloop push is 0!!!!!!!!\n", mst->world_rank);
		CUDA_CHECK_RETURN (cudaMemcpy (&rcv_size, cm->extra_send_offsets + total_num_partitions, sizeof(voff_t), cudaMemcpyDeviceToHost));
		printf ("WORD_RANK %d:GPU::::::::::CCCCCCCCCCcheck number of messages pushed to inter selfloop: %u\n", rcv_size, mst->world_rank);
		rcv_size = 200;
//		exit(0);
	}
	cm->temp_size = malloc_pull_push_receive_device (&cm->receive, sizeof(selfloop_t), did, rcv_size, 20*(total_num_partitions+num_of_partitions-1)/num_of_partitions, world_rank, num_of_devices);
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
		int num_of_blocks = (num_mssgs + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		push_selfloop <<<block_size, THREADS_PER_BLOCK_NODES>>> (num_mssgs, pid, index_offset[i], total_num_partitions, 0, 1);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	push_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PUSH SELFOOLP GPU INTRA PROCESSOR TIME: ");
#endif

#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&start, NULL);
#endif
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset+i];
		voff_t num_mssgs = mst->soff[did][i+1] - mst->soff[did][i];
		int num_of_blocks = (num_mssgs + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		push_selfloop <<<block_size, THREADS_PER_BLOCK_NODES>>> (num_mssgs, pid, index_offset[i], total_num_partitions, receive_start, 0);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	push_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PUSH SELFOOLP GPU INTER PROCESSORS TIME: ");
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
#ifdef MEASURE_TIME_
	gettimeofday(&start, NULL);
	CUDA_CHECK_RETURN (cudaMemcpy(mst->receive[did], (selfloop_t *)cm->receive+inter_start, (inter_end-inter_start)*sizeof(selfloop_t), cudaMemcpyDeviceToHost));
	gettimeofday(&end, NULL);
	memcpydh_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
#else
	CUDA_CHECK_RETURN (cudaMemcpyAsync(mst->receive[did], (selfloop_t *)cm->receive+inter_start, (inter_end-inter_start)*sizeof(selfloop_t), cudaMemcpyDeviceToHost, streams[did]));
#endif

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
		pull_selfloop <<<block_size, THREADS_PER_BLOCK_NODES>>> (num_mssgs, pid, index_offset[i], (selfloop_t*)cm->receive, 1);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	pull_intra_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PULL SELFLOOP GPU INTRA PROCESSOR TIME: ");
#endif
	gettimeofday (&overe, NULL);
	over_time[did] += (float)((overe.tv_sec * 1000000 + overe.tv_usec) - (overs.tv_sec * 1000000 + overs.tv_usec)) / 1000;

	return ((void *) 0);
}

void * modifygraph_inter_pull_gpu (void * arg)
{
	evaltime_t overs, overe;
	lr_arg * carg = (lr_arg *) arg;
	comm_t * cm = carg->cm;
	master_t * mst = carg->mst;
	int did = carg->did;
	int num_of_partitions = mst->num_partitions[did + 1] - mst->num_partitions[did];
	int poffset = mst->num_partitions[did];
	voff_t * index_offset = mst->index_offset[did];
	int num_of_devices = mst->num_of_devices;
	int world_rank = mst->world_rank;
	if (world_rank == 0)
		printf ("WORLD RANK %d: modigygraph inter pull gpu %d:\n", world_rank, did);

#ifdef SINGLE_NODE
	CUDA_CHECK_RETURN(cudaSetDevice (world_rank * num_of_devices + did));
#else
	CUDA_CHECK_RETURN(cudaSetDevice (did + DEVICE_SHIFT));
#endif

	gettimeofday (&overs, NULL);
	voff_t receive_start = mst->roff[did][num_of_partitions];
	voff_t inter_size = mst->soff[did][num_of_partitions];
	if (cm->temp_size < (inter_size+receive_start)*sizeof(selfloop_t))
	{
		printf("Error:::::::: malloced receive buffer size %lu smaller than actual receive buffer size %lu!\n", cm->temp_size, (inter_size+receive_start)*sizeof(selfloop_t));
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
	CUDA_CHECK_RETURN (cudaMemcpy((selfloop_t *)(cm->receive) + receive_start, mst->send[did], sizeof(selfloop_t) * inter_size, cudaMemcpyHostToDevice));
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
		int num_of_blocks = (num_mssgs + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		pull_selfloop <<<block_size, THREADS_PER_BLOCK_NODES>>> (num_mssgs, pid, index_offset[i], (selfloop_t*)(cm->receive) + receive_start, 0);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	pull_inter_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PULL SELFLOOP GPU INTER PROCESSORS TIME: ");
#endif

	printf ("+++++++++ DST NOT FOUND IN FIRST ITERATION: %u\n", dst_not_found);
	gettimeofday (&overe, NULL);
	over_time[did] += (float)((overe.tv_sec * 1000000 + overe.tv_usec) - (overs.tv_sec * 1000000 + overs.tv_usec)) / 1000;

	// *************** free (send and) receive buffer for pull and push mode
	free_pull_push_receive_device (did, cm, world_rank, num_of_devices);

	return ((void *) 0);
}

void * modifygraph_pull_gpu (void * arg)
{
	lr_arg * carg = (lr_arg *) arg;
	comm_t * cm = carg->cm;
	master_t * mst = carg->mst;
	int did = carg->did;
	int num_of_partitions = mst->num_partitions[did + 1] - mst->num_partitions[did];
	int poffset = mst->num_partitions[did];
	voff_t * index_offset = mst->index_offset[did];
	int num_of_devices = mst->num_of_devices;
	int world_rank = mst->world_rank;
	if (world_rank == 0)
		printf ("WORLD RANK %d: modigygraph pull gpu %d:\n", world_rank, did);

#ifdef SINGLE_NODE
	CUDA_CHECK_RETURN(cudaSetDevice (world_rank * num_of_devices + did));
#else
	CUDA_CHECK_RETURN(cudaSetDevice (did + DEVICE_SHIFT));
#endif

#ifdef MEASURE_TIME_
	evaltime_t start, end;
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&start, NULL);
#endif
	int i;
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset + i];
		voff_t num_mssgs = mst->roff[did][i+1] - mst->roff[did][i];
		int num_of_blocks = (num_mssgs + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		pull_selfloop <<<block_size, THREADS_PER_BLOCK_NODES>>> (num_mssgs, pid, index_offset[i], (selfloop_t*)cm->receive, 1);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PULL SELFLOOP GPU INTRA PROCESSOR TIME: ");
#endif

	voff_t receive_start = mst->roff[did][num_of_partitions];
	CUDA_CHECK_RETURN (cudaMemcpy(cm->receive_offsets, mst->soff[did], sizeof(voff_t) * (num_of_partitions+1), cudaMemcpyHostToDevice));
	voff_t inter_size = mst->soff[did][num_of_partitions];
	CUDA_CHECK_RETURN (cudaMemcpy((selfloop_t *)(cm->receive) + receive_start, mst->send[did], sizeof(selfloop_t) * inter_size, cudaMemcpyHostToDevice));

#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&start, NULL);
#endif

	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset + i];
		voff_t num_mssgs = mst->soff[did][i+1] - mst->soff[did][i];
		int num_of_blocks = (num_mssgs + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		pull_selfloop <<<block_size, THREADS_PER_BLOCK_NODES>>> (num_mssgs, pid, index_offset[i], (selfloop_t*)(cm->receive) + receive_start, 0);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PULL SELFLOOP GPU INTER PROCESSORS TIME: ");
#endif

	printf ("+++++++++ DST NOT FOUND IN FIRST ITERATION: %u\n", dst_not_found);

	// *************** free (send and) receive buffer for pull and push mode
	free_pull_push_receive_device (did, cm, world_rank, num_of_devices);

	return ((void *) 0);
}


void
listrank_hetero_workflow (int num_of_partitions, subgraph_t * subgraph, d_jvs_t * djs, d_lvs_t * dls, master_t * mst, int k, int world_size, int world_rank)
{
	evaltime_t start, end;
	evaltime_t overs, overe; // measure big step in-memory processing time
	evaltime_t totals, totale; // measure total time including initialization and finalization

	int num_of_cpus = mst->num_of_cpus;
	int num_of_devices = mst->num_of_devices;
	mst->mssg_size = sizeof(path_t);
	meta_t * dm = (meta_t *) malloc (sizeof(meta_t) * (num_of_devices + num_of_cpus));

	// the following should be deleted if mst is correct
	int np_per_node;
	int np_node;
	get_np_node (&np_per_node, &np_node, num_of_partitions, world_size, world_rank);

	gettimeofday (&totals, NULL);
	gettimeofday(&start, NULL);
	if (world_rank == 0)
		printf ("WORLD RANK %d: IIIIIIIIIII initialize distributing partitions: \n", world_rank);
	init_distribute_partitions (num_of_partitions, mst, world_size);
	get_subgraph_sizes (subgraph, np_node);
	double unit_vsize = sizeof(path_t)*2 + sizeof(vid_t)*2 + sizeof(voff_t)*2 + sizeof(vid_t)*2 + sizeof(edge_type)*3 \
			+ junction_factor*(sizeof(ull)+EDGE_DIC_SIZE*(sizeof(vid_t)+sizeof(voff_t)+sizeof(path_t)+sizeof(char)*(k+1))+sizeof(size_t)+sizeof(kmer_t));
	distribute_partitions (num_of_partitions, mst, subgraph, uneven, world_size, world_rank, subgraph->total_graph_size, unit_vsize);
	gettimeofday(&end, NULL);
	if (world_rank == 0)
		print_exec_time (start, end, "WORLD RANK %d: #################### distributing partitions time: ", world_rank);

	int num_partitions_node = mst->num_partitions[num_of_cpus+num_of_devices];
	if (num_partitions_node != np_node)
	{
		printf ("error in mst->num_partitions, num_partitions_node=%d, np_node=%d!!!\n", num_partitions_node, np_node);
//		exit(0);
	}

	init_device_graph_compute (dm, mst);
	set_globals_graph_compute_gpu (dm, mst);

	init_host_graph_compute (dm, mst);

	// **************** malloc writing offset buffer for pull and push mode
	int i;
	for (i=0; i<num_of_devices; i++)
	{
		malloc_pull_push_offset_gpu (&dm[i].comm.extra_send_offsets, mst, i);
		set_extra_send_offsets_gpu (&dm[i].comm.extra_send_offsets, mst, i);
	}
	for (i=num_of_devices; i<num_of_devices+num_of_cpus; i++)
	{
		malloc_pull_push_offset_cpu(&dm[i].comm.extra_send_offsets, mst);
	}

	gettimeofday (&start, NULL);
	for (i=0; i<num_of_devices; i++)
	{
		init_graph_data_gpu (i, mst, &dm[i], dls);
	}
	for (i=num_of_devices; i<num_of_devices+num_of_cpus; i++)
	{
		init_graph_data_cpu (i, mst, &dm[i], dls);
	}
	gettimeofday (&end, NULL);

	if (world_rank == 0)
		print_exec_time (start, end, "WORLD RANK %d: IIIIIIIIIIIIIIInit graph data input time: \n", world_rank);

	pthread_t cpu_threads[NUM_OF_CPUS];
	pthread_t gpu_threads[NUM_OF_DEVICES];

	lr_arg arg[NUM_OF_DEVICES + NUM_OF_CPUS];
	for (i = 0; i < num_of_devices + num_of_cpus; i++)
	{
		arg[i].did = i;
		arg[i].cm = &dm[i].comm;
		arg[i].mst = mst;
		arg[i].dm = &dm[i];
		arg[i].k = k;
	}
//	gettimeofday (&end, NULL);
	if (world_rank == 0)
		print_exec_time (start, end, "WORLD RANK %d: #################### global initiation time: ", world_rank);


	float all2all_time = 0;
	//************** Modify graph structure here: ***************
	gettimeofday (&overs, NULL);

	int iters=0;
	mst->mssg_size = sizeof(path_t);

	uint intra_mssgs[NUM_OF_PROCS*MAX_NUM_ITERATION];
	uint inter_mssgs[NUM_OF_PROCS*MAX_NUM_ITERATION];
	init_mssg_count (intra_mssgs, inter_mssgs);
	long long prev_num_mssgs = 0;

	gettimeofday(&overs, NULL);
	while(1)
	{
		iters++;
		dst_not_found = 0;
		selfloop_made = 0;
		if(iters > MAX_NUM_ITERATION)
			break;
		if (world_rank == 0)
			printf("\n++++++++++++++++ Iteration %d: WORLD RANK %d ++++++++++++++++++\n", iters, world_rank);
		for (i = 0; i < num_of_devices; i++)
		{
			if (pthread_create (&gpu_threads[i], NULL, listrank_push_intra_pull_gpu, &arg[i]) != 0)
			{
				printf ("create thread for listranking push on gpu %d error!\n", i);
			}
		}
		for (i = 0; i < num_of_cpus; i++)
		{
			if (pthread_create (&cpu_threads[i], NULL, listrank_push_intra_pull_cpu, &arg[num_of_devices + i]) != 0)
			{
				printf ("create thread for listranking push on cpu %d error!\n", i);
			}
		}
#ifndef SYNC_ALL2ALL_
		if (pthread_create (&comm_thread, NULL, master_all2all_async, &cm_arg) != 0)
		{
			printf ("Create thread for communication error!\n");
		}
#endif

		for (i = 0; i < num_of_devices; i++)
		{
			if (pthread_join (gpu_threads[i], NULL) != 0)
			{
				printf ("Join thread on listranking push on gpu %d failure!\n", i);
			}
		}
		for (i = 0; i < num_of_cpus; i++)
		{
			if (pthread_join (cpu_threads[i], NULL) != 0)
			{
				printf ("Join thread on listranking push on cpu %d failure!\n", i);
			}
		}
#ifndef SYNC_ALL2ALL_
		if (pthread_join (comm_thread, NULL) != 0)
		{
			printf ("Join communication thread failure!\n");
		}

		int curr_num_mssgs = cm_arg.num_mssgs;
		if (curr_num_mssgs - prev_num_mssgs < THRESHOLD)
			break;
		prev_num_mssgs = curr_num_mssgs;
		get_mssg_count(mst, intra_mssgs, inter_mssgs, iters);
#else
		// ************* DATA COMMUNICATION: if number of messages for communication is 0, break iterations ***********
		gettimeofday (&start, NULL);
		long long curr_num_mssgs = master_all2all(mst);
//		printf ("current global number of messages: %ld\n", curr_num_mssgs);
		if (curr_num_mssgs == prev_num_mssgs)
		{
			gettimeofday (&end, NULL);
			all2all_time += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
//			print_exec_time (start, end, "TICKTOCK TICKTOCK TICKTOCK:: master all to all time in INTERATION %d for listranking push and pull: ", iters);
			break;
		}
		prev_num_mssgs = curr_num_mssgs;
		get_mssg_count(mst, intra_mssgs, inter_mssgs, iters);
		gettimeofday (&end, NULL);
		all2all_time += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
//		print_exec_time (start, end, "TICKTOCK TICKTOCK TICKTOCK:: master all to all time INTERATION %d for listranking push and pull: ", iters);
#endif

		for (i = 0; i < num_of_devices; i++)
		{
			if (pthread_create (&gpu_threads[i], NULL, listrank_inter_pull_gpu, &arg[i]) != 0)
			{
				printf ("create thread for listrankng pull on gpu %d error!\n", i);
			}
		}
		for (i = 0; i < num_of_cpus; i++)
		{
			if (pthread_create (&cpu_threads[i], NULL, listrank_inter_pull_cpu, &arg[num_of_devices + i]) != 0)
			{
				printf ("create thread for listranking pull on cpu %d error!\n", i);
			}
		}
		for (i = 0; i < num_of_devices; i++)
		{
			if (pthread_join (gpu_threads[i], NULL) != 0)
			{
				printf ("Join thread on listranking pull on gpu %d failure!\n", i);
			}
		}
		for (i = 0; i < num_of_cpus; i++)
		{
			if (pthread_join (cpu_threads[i], NULL) != 0)
			{
				printf ("Join thread on listranking pull on cpu %d failure!\n", i);
			}
		}
		if (world_rank == 0)
		{
//		printf("~~~~~~~~~~~~ WORLD RANK %d:::::::::: DST NOT FOUND: %u\n", world_rank, dst_not_found);
//		printf ("SSSSSSSSSSSSSSSSSSSselfloop made in total of rank %d: %u SSSSSSSSSSSSSSSSS\n", world_rank, selfloop_made);
		}
	}
//	while (debug==0) {}

	gettimeofday(&overe, NULL);
	inmemory_time += (float)((overe.tv_sec * 1000000 + overe.tv_usec) - (overs.tv_sec * 1000000 + overs.tv_usec)) / 1000; // recording in memory processing time

	if (world_rank == 0)
	{
		printf ("WORLD RANK %d: SSSSUMARY:::::~~~~~~~~~~~~~~~~ ALLTOALL TIME MEASURED: %f\n", world_rank, all2all_time);
		printf("WORLD RANK %d: SSSSUMARY:::::~~~~~~~~~~~~ NUMBER OF ITERATIONS FOR LIST RANKING: %d\n", world_rank, iters);
		print_exec_time (overs, overe, "WORLD RANK %d: DINGDONG DINGDONG DINGDONG DINGDONG :: uniPath time measured: ", world_rank);
	}
	for (i=0; i<num_of_devices; i++)
	{
#ifdef SINGLE_NODE
		CUDA_CHECK_RETURN (cudaSetDevice(world_rank * num_of_devices + i));
#else
		CUDA_CHECK_RETURN (cudaSetDevice(i + DEVICE_SHIFT));
#endif
		CUDA_CHECK_RETURN (cudaDeviceSynchronize());
#ifdef MEASURE_TIME_
		if (world_rank == 0)
		{
		printf ("WORLD RANK %d: SSSSUMARY:::::&&&&&&&&&&& PUSH_OFFSET TIME FOR GPU %d: %f &&&&&&&&&&&&&\n", world_rank, i, push_offset_time[i]);
		printf ("WORLD RANK %d: SSSSUMARY:::::&&&&&&&&&&& PUSH TIME FOR GPU %d: %f &&&&&&&&&&&&&&&&&&\n", world_rank, i, push_time[i]);
		printf ("WORLD RANK %d: SSSSUMARY:::::&&&&&&&&&&& PULL INTRA TIME FOR GPU %d: %f &&&&&&&&&&&&&&\n", world_rank, i, pull_intra_time[i]);
		printf ("WORLD RANK %d: SSSSUMARY:::::&&&&&&&&&&& PULL INTER TIME FOR GPU %d: %f &&&&&&&&&&&&&&\n", world_rank, i, pull_inter_time[i]);
		printf ("WORLD RANK %d: SSSSUMARY:::::&&&&&&&&&&& memcpy device to host time for GPU %d: %f &&&&&&&&&&&&&\n", world_rank, i, memcpydh_time[i]);
		printf ("WORLD RANK %d: SSSSUMARY:::::&&&&&&&&&&& memcpy host to device time for GPU %d: %f &&&&&&&&&&&&&\n", world_rank, i, memcpyhd_time[i]);
		}
#endif
	}
#ifdef MEASURE_TIME_
	if (world_rank == 0)
	{
	printf ("WORLD RANK %d: SSSSUMARY:::::&&&&&&&&&&& PUSH_OFFSET TIME FOR CPU: %f &&&&&&&&&&&&&\n", world_rank, push_offset_time[i]);
	printf ("WORLD RANK %d: SSSSUMARY:::::&&&&&&&&&&& PUSH TIME FOR CPU: %f &&&&&&&&&&&&&&&&&&\n", world_rank, push_time[i]);
	printf ("WORLD RANK %d: SSSSUMARY:::::&&&&&&&&&&& PULL INTRA TIME FOR CPU: %f &&&&&&&&&&&&&&\n", world_rank, pull_intra_time[i]);
	printf ("WORLD RANK %d: SSSSUMARY:::::&&&&&&&&&&& PULL INTER TIME FOR CPU: %f &&&&&&&&&&&&&&\n", world_rank, pull_inter_time[i]);
	}
#endif
	for (i=0; i<num_of_cpus+num_of_devices; i++)
	{
		if (world_rank == 0)
			printf ("+++++++++++ WORLD RANK %d: overall computation time on device %d::::::::::: %f\n", world_rank, i, over_time[i]);
	}

	for (i=num_of_devices; i<num_of_devices+num_of_cpus; i++)
	{
		finalize_graph_data_cpu ();
	}

	gettimeofday (&totale, NULL);
	if (world_rank == 0)
		print_exec_time (totals, totale, "WORLD RANK %d: TTTTTTTTOTAL TIME MEASURED:: uniPath time measured: ", world_rank);

	// ********** compact dbgraph with junction updates and contig output: implemented in contig.c and contig.cu ************
	gettimeofday (&totals, NULL);
	compact_dbgraph_contig (num_of_partitions, subgraph, djs, dls, dm, mst, k, world_size, world_rank);

	finalize_device_graph_compute (dm, mst);
	finalize_host_graph_compute (dm, mst);
	finalize_distribute_partitions (mst);
	finalize_receive_all2all(world_size);

	gettimeofday (&totale, NULL);
	if (world_rank == 0)
		print_exec_time (totals, totale, "WORLD RANK %d: TTTTTTTTOTAL TIME MEASURED:: compacting and gathering contigs including contig output time measured: ", world_rank);

	free(dm);
	if (world_rank == 0)
	{
	printf ("WWWWWWWWWWWWWWWWWorld rank %d:::::::::::::::\n", world_rank);
	print_mssg_count(mst->num_of_cpus+mst->num_of_devices, intra_mssgs, inter_mssgs, iters);
	}
}


void
traverse_dbgraph (int num_of_partitions, subgraph_t * subgraph, d_jvs_t * djs, d_lvs_t * dls, master_t * mst, int k, int world_size, int world_rank)
{
	listrank_hetero_workflow (num_of_partitions, subgraph, djs, dls, mst, k, world_size, world_rank);
}
}
