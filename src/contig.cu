/*
 * contig.cu
 *
 *  Created on: 2018-5-10
 *      Author: qiushuang
 *
 *  This file updates neighbors of junctions, overpassing the linear vertices; gathers contigs
 */

#include <pthread.h>
#include "../include/dbgraph.h"
#include "../include/comm.h"
#include "../include/contig.h"
#include "../include/distribute.h"
#include "../include/hash.h"
#include "malloc.cuh"
#include "contig.cuh"
#include "../include/scan.cu"

extern int cutoff;
extern uint dst_not_found;
extern float push_offset_time[NUM_OF_PROCS];
extern float push_time[NUM_OF_PROCS];
extern float pull_intra_time[NUM_OF_PROCS];
extern float pull_inter_time[NUM_OF_PROCS];
extern float over_time[NUM_OF_PROCS];

extern float memcpydh_time[NUM_OF_PROCS];
extern float memcpyhd_time[NUM_OF_PROCS];
extern float all2all_time_async;
extern float inmemory_time;

extern int lock_flag[NUM_OF_PROCS];

//#define SYNC_ALL2ALL_

extern "C"
{
void * compact_push_update_intra_push_gpu (void * arg)
{
	evaltime_t overs, overe;
	lr_arg * garg = (lr_arg *) arg;
	int did = garg->did;
	comm_t * cm = garg->cm;
	master_t * mst = garg->mst;
	if (mst->world_rank == 0)
		printf ("WORLD RANK %d: Compact push update intra push gpu %d:\n", mst->world_rank, did);

	CUDA_CHECK_RETURN(cudaSetDevice (did + DEVICE_SHIFT));

	gettimeofday (&overs, NULL);

	int total_num_partitions = mst->total_num_partitions;
	int num_of_partitions = mst->num_partitions[did + 1] - mst->num_partitions[did];
	int poffset = mst->num_partitions[did];
	voff_t * lindex_offset = mst->index_offset[did];
	voff_t * jindex_offset = mst->jindex_offset[did];
	voff_t * jnb_index_offset = mst->jnb_index_offset[did];

	CUDA_CHECK_RETURN (cudaMemset(cm->send_offsets, 0, sizeof(voff_t) * (total_num_partitions + 1)));

#ifdef MEASURE_TIME_
	evaltime_t start, end;
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&start, NULL);
#endif
	int i;
	voff_t spid_offset = 0;
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset+i];
		voff_t size = mst->jid_offset[pid];
		if(i>0) spid_offset += (jnb_index_offset[i] - jnb_index_offset[i-1])/2 + (jnb_index_offset[i] - jnb_index_offset[i-1])%2;
		int num_of_blocks = (size + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		push_mssg_offset_compact <<<block_size, THREADS_PER_BLOCK_NODES, sizeof(vid_t)*(total_num_partitions+1), streams[did][0]>>> (size, total_num_partitions, pid, jindex_offset[i], jnb_index_offset[i], spid_offset);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	push_offset_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PUSH MSSG OFFSET FOR GPU COMPACT INTRA PROCESSOR TIME: ");
#endif

	inclusive_scan ((int *)(cm->send_offsets), total_num_partitions + 1, NULL);

	CUDA_CHECK_RETURN (cudaMemset(cm->tmp_send_offsets, 0, sizeof(voff_t) * (total_num_partitions + 1)));

#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&start, NULL);
#endif
	spid_offset=0;
	for (i=0; i< num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset+i];
		voff_t size = mst->jid_offset[pid];
		if(i>0) spid_offset += (jnb_index_offset[i] - jnb_index_offset[i-1])/2 + (jnb_index_offset[i] - jnb_index_offset[i-1])%2;
		int num_of_blocks = (size + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		push_mssg_compact <<<block_size, THREADS_PER_BLOCK_NODES, sizeof(vid_t)*(total_num_partitions+1), streams[did][0]>>> (size, total_num_partitions, pid, jindex_offset[i], jnb_index_offset[i], spid_offset);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	push_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PUSH MSSG FOR GPU COMPACT INTRA PROCESSOR TIME: ");
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

#ifndef SYNC_ALL2ALL_
	if (atomic_set_value(&lock_flag[did], 1, 0) == false)
		printf ("!!!!!!!!!!! CAREFUL: ATOMIC SET VALUE ERROR IN GPU %d\n", did);
#endif

#ifdef MEASURE_MEMCPY_
	gettimeofday(&start, NULL);
	CUDA_CHECK_RETURN (cudaMemcpy(mst->receive[did], (query_t *)cm->send + inter_start, sizeof(query_t) * (inter_end-inter_start), cudaMemcpyDeviceToHost));
	gettimeofday(&end, NULL);
	memcpydh_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
#else
#ifndef SYNC_ALL2ALL_
	CUDA_CHECK_RETURN (cudaMemcpyAsync(mst->receive[did], (query_t *)cm->send + inter_start, sizeof(query_t) * (inter_end-inter_start), cudaMemcpyDeviceToHost, streams[did][1]));
#else
	CUDA_CHECK_RETURN (cudaMemcpy(mst->receive[did], (query_t *)cm->send + inter_start, sizeof(query_t) * (inter_end-inter_start), cudaMemcpyDeviceToHost));
#endif
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
		push_update_offset <<<block_size, THREADS_PER_BLOCK_NODES, sizeof(vid_t)*(total_num_partitions+1), streams[did][0]>>> (num_mssgs, pid, lindex_offset[i], total_num_partitions, 0, 1);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	pull_intra_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PUSH UPDATE OFFSET GPU INTRA PROCESSOR TIME: ");
#endif
	gettimeofday (&overe, NULL);
	over_time[did] += (float)((overe.tv_sec * 1000000 + overe.tv_usec) - (overs.tv_sec * 1000000 + overs.tv_usec)) / 1000;

	return ((void *) 0);
}

void * compact_pull_update_inter_push_intra_pull_gpu (void * arg)
{
	evaltime_t overs, overe;
	lr_arg * carg = (lr_arg *) arg;
	int did = carg->did;
	comm_t * cm = carg->cm;
	master_t * mst = carg->mst;
	int num_of_devices = mst->num_of_devices;
	int world_rank = mst->world_rank;
	if (world_rank == 0)
		printf ("WORLD RANK %d: Compact pull junction update push gpu %d\n", world_rank, did);

#ifdef SINGLE_NODE
	CUDA_CHECK_RETURN(cudaSetDevice (world_rank * num_of_devices + did));
#else
	CUDA_CHECK_RETURN(cudaSetDevice (did + DEVICE_SHIFT));
#endif
	gettimeofday (&overs, NULL);
	int total_num_partitions = mst->total_num_partitions;
	int num_of_partitions = mst->num_partitions[did + 1] - mst->num_partitions[did];
	int poffset = mst->num_partitions[did];
	voff_t * lindex_offset = mst->index_offset[did];
	voff_t * jindex_offset = mst->jindex_offset[did];
	voff_t * jnb_index_offset = mst->jnb_index_offset[did];
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
	CUDA_CHECK_RETURN (cudaMemcpy((query_t*)(cm->send) + receive_start, mst->send[did], sizeof(query_t) * inter_size, cudaMemcpyHostToDevice));
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
		push_update_offset <<<block_size, THREADS_PER_BLOCK_NODES, sizeof(vid_t)*(total_num_partitions+1), streams[did][0]>>> (num_mssgs, pid, lindex_offset[i], total_num_partitions, receive_start, 0);
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
	double remote_factor;
	voff_t init_mssg_size;
	if (rcv_size == 0)
	{
		printf ("WORLD RANK %d: did %d: CCCCCCCCCcccareful:::::::::: receive size from intra junction update push is 0!!!!!!!!\n", mst->world_rank, did);
		remote_factor = 1;
	}
	else
	{
		remote_factor = ceil((double)(jnb_index_offset[num_of_partitions] - rcv_size)/rcv_size);
	}
	init_mssg_size = (jnb_index_offset[num_of_partitions] - rcv_size) > rcv_size ? (jnb_index_offset[num_of_partitions] - rcv_size) : rcv_size;
	cm->temp_size = malloc_pull_push_receive_device (&cm->receive, sizeof(compact_t), did, init_mssg_size, ((int)remote_factor) * 2, world_rank, num_of_devices);
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
		push_update <<<block_size, THREADS_PER_BLOCK_NODES, sizeof(vid_t)*(total_num_partitions+1), streams[did][0]>>> (num_mssgs, pid, lindex_offset[i], total_num_partitions, 0, 1);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	push_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PUSH UPDATE GPU INTRA PROCESSOR TIME: ");
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
		push_update <<<block_size, THREADS_PER_BLOCK_NODES, sizeof(vid_t)*(total_num_partitions+1), streams[did][0]>>> (num_mssgs, pid, lindex_offset[i], total_num_partitions, receive_start, 0);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	push_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PUSH UPDATE GPU INTER PROCESSORS TIME: ");
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

#ifndef SYNC_ALL2ALL_
	if (atomic_set_value(&lock_flag[did], 1, 0) == false)
		printf ("!!!!!!!!!!! CAREFUL: ATOMIC SET VALUE ERROR IN GPU %d\n", did);
#endif

#ifdef MEASURE_TIME_
	gettimeofday(&start, NULL);
	CUDA_CHECK_RETURN (cudaMemcpy(mst->receive[did], (compact_t *)cm->receive+inter_start, (inter_end-inter_start)*sizeof(compact_t), cudaMemcpyDeviceToHost));
	gettimeofday(&end, NULL);
	memcpydh_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
#else
#ifndef SYNC_ALL2ALL_
	CUDA_CHECK_RETURN (cudaMemcpyAsync(mst->receive[did], (compact_t *)cm->receive+inter_start, (inter_end-inter_start)*sizeof(compact_t), cudaMemcpyDeviceToHost, streams[did][1]));
#else
	CUDA_CHECK_RETURN (cudaMemcpy(mst->receive[did], (compact_t *)cm->receive+inter_start, (inter_end-inter_start)*sizeof(compact_t), cudaMemcpyDeviceToHost));
#endif
#endif

#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&start, NULL);
#endif
	voff_t spid_offset=0;
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset + i];
		voff_t num_mssgs = mst->roff[did][i+1] - mst->roff[did][i];
		if (i>0) spid_offset += (jnb_index_offset[i] - jnb_index_offset[i-1])/2 + (jnb_index_offset[i] - jnb_index_offset[i-1])%2;
		int num_of_blocks = (num_mssgs + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		pull_update <<<block_size, THREADS_PER_BLOCK_NODES, 0, streams[did][0]>>> (num_mssgs, pid, jindex_offset[i], jnb_index_offset[i], spid_offset, cm->receive, 1);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	pull_intra_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PULL UPDATE GPU INTRA PROCESSOR TIME: ");
#endif
	gettimeofday (&overe, NULL);
	over_time[did] += (float)((overe.tv_sec * 1000000 + overe.tv_usec) - (overs.tv_sec * 1000000 + overs.tv_usec)) / 1000;

	return ((void *) 0);
}

void * update_inter_pull_gpu (void * arg)
{
	evaltime_t overs, overe;
	lr_arg * carg = (lr_arg *) arg;
	comm_t * cm = carg->cm;
	master_t * mst = carg->mst;
	int did = carg->did;
	int k = carg->k;
	int num_of_partitions = mst->num_partitions[did + 1] - mst->num_partitions[did];
	int total_num_partitions = mst->total_num_partitions;
	int poffset = mst->num_partitions[did];
	voff_t * jindex_offset = mst->jindex_offset[did];
	voff_t * jnb_index_offset = mst->jnb_index_offset[did];
	int num_of_devices = mst->num_of_devices;
	int world_rank = mst->world_rank;

	if (world_rank == 0)
		printf ("WORLD RANK %d: Junction update inter pull gpu %d:\n", world_rank, did);

	CUDA_CHECK_RETURN(cudaSetDevice (did + DEVICE_SHIFT));

	gettimeofday (&overs, NULL);
	voff_t receive_start = mst->roff[did][num_of_partitions];
	voff_t inter_size = mst->soff[did][num_of_partitions];
	if (cm->temp_size < (inter_size+receive_start)*sizeof(compact_t))
	{
		printf("WORLD RANK %d: did %d: Error:::::::: malloced receive buffer size smaller than actual receive buffer size!\n", world_rank, did);
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
	CUDA_CHECK_RETURN (cudaMemcpy((compact_t *)(cm->receive) + receive_start, mst->send[did], sizeof(compact_t) * inter_size, cudaMemcpyHostToDevice));
#ifdef MEASURE_MEMCPY_
	gettimeofday(&end, NULL);
	memcpyhd_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
#endif

#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&start, NULL);
#endif
	int i;
	voff_t spid_offset = 0;
	for (i=0; i<num_of_partitions; i++)
	{
		if (inter_size == 0)
			break;
		int pid = mst->partition_list[poffset + i];
		voff_t num_mssgs = mst->soff[did][i+1] - mst->soff[did][i];
		if(i>0) spid_offset += (jnb_index_offset[i] - jnb_index_offset[i-1])/2 + (jnb_index_offset[i] - jnb_index_offset[i-1])%2;
		int num_of_blocks = (num_mssgs + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		pull_update <<<block_size, THREADS_PER_BLOCK_NODES>>> (num_mssgs, pid, jindex_offset[i], jnb_index_offset[i], spid_offset, (char*)cm->receive + sizeof(compact_t) * receive_start, 0);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	pull_inter_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PULL SELFLOOP GPU INTER PROCESSORS TIME: ");
#endif

//	printf ("+++++++++ DST NOT FOUND IN FIRST ITERATION: %u\n", dst_not_found);
	gettimeofday (&overe, NULL);
	over_time[did] += (float)((overe.tv_sec * 1000000 + overe.tv_usec) - (overs.tv_sec * 1000000 + overs.tv_usec)) / 1000;

	// *************** free (send and) receive buffer for pull and push mode
	free_pull_push_receive_device (did, cm, world_rank, num_of_devices);
	spid_offset=0;
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset + i];
		voff_t size = mst->jid_offset[pid];
		if (i>0) spid_offset += (jnb_index_offset[i] - jnb_index_offset[i-1])/2 + (jnb_index_offset[i] - jnb_index_offset[i-1])%2;
		int num_of_blocks = (size + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		update_ulens_with_kplus1 <<<block_size, THREADS_PER_BLOCK_NODES, sizeof(vid_t)*(total_num_partitions+1)>>> (size, pid, jindex_offset[i], jnb_index_offset[i], spid_offset, k, total_num_partitions);
	}

	return ((void *) 0);
}

void * gather_contig_push_intra_pull_gpu (void * arg)
{
	evaltime_t overs, overe;
	lr_arg * carg = (lr_arg *) arg;
	comm_t * cm = carg->cm;
	master_t * mst = carg->mst;
	int did = carg->did;
	int k = carg->k;
	if (mst->world_rank == 0)
		printf ("WORLD RANK %d: Gather contig push intra pull gpu %d:\n", mst->world_rank, did);

	gettimeofday (&overs, NULL);
	int total_num_partitions = mst->total_num_partitions;
	int num_of_partitions = mst->num_partitions[did + 1] - mst->num_partitions[did];
	int poffset = mst->num_partitions[did];
	voff_t * lindex_offset = mst->index_offset[did];
	voff_t * jindex_offset = mst->jindex_offset[did];
	voff_t * jnb_index_offset = mst->jnb_index_offset[did];
	int i;

#ifdef SINGLE_NODE
	int world_rank = mst->world_rank;
	int num_of_devices = mst->num_of_devices;
	CUDA_CHECK_RETURN(cudaSetDevice (world_rank * num_of_devices + did));
#else
	CUDA_CHECK_RETURN(cudaSetDevice (did + DEVICE_SHIFT));
#endif
	gettimeofday (&overs, NULL);

	CUDA_CHECK_RETURN (cudaMemset(cm->send_offsets, 0, sizeof(voff_t) * (total_num_partitions + 1)));

#ifdef MEASURE_TIME_
	evaltime_t start, end;
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&start, NULL);
#endif
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset+i];
		voff_t size = mst->id_offsets[pid+1] - mst->id_offsets[pid] - mst->jid_offset[pid];
		int num_of_blocks = (size + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		push_mssg_offset_contig <<<block_size, THREADS_PER_BLOCK_NODES, sizeof(vid_t)*(total_num_partitions+1), streams[did][0]>>> (size, pid, lindex_offset[i], total_num_partitions);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	push_offset_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PUSH MSSG OFFSET FOR GPU GATHERING CONTIGS INTRA PROCESSOR TIME: ");
#endif

	inclusive_scan ((int *)(cm->send_offsets), total_num_partitions + 1, NULL);

	CUDA_CHECK_RETURN (cudaMemset(cm->tmp_send_offsets, 0, sizeof(voff_t) * (total_num_partitions + 1)));

#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&start, NULL);
#endif

	for (i=0; i< num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset+i];
		voff_t size = mst->id_offsets[pid+1] - mst->id_offsets[pid] - mst->jid_offset[pid];
		int num_of_blocks = (size + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		push_mssg_contig <<<block_size, THREADS_PER_BLOCK_NODES, sizeof(vid_t)*(total_num_partitions+1), streams[did][0]>>> (size, pid, lindex_offset[i], total_num_partitions);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	push_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PUSH MSSG FOR GPU GATHERING CONTIGS INTRA PROCESSOR TIME: ");
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
	CUDA_CHECK_RETURN (cudaMemcpy(mst->receive[did], (unitig_t *)cm->send + inter_start, sizeof(unitig_t) * (inter_end-inter_start), cudaMemcpyDeviceToHost));
	gettimeofday(&end, NULL);
	memcpydh_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
#else
#ifndef SYNC_ALL2ALL_
	CUDA_CHECK_RETURN (cudaMemcpyAsync(mst->receive[did], (unitig_t *)cm->send + inter_start, sizeof(unitig_t) * (inter_end-inter_start), cudaMemcpyDeviceToHost, streams[did][1]));
#else
	CUDA_CHECK_RETURN (cudaMemcpy(mst->receive[did], (unitig_t *)cm->send + inter_start, sizeof(unitig_t) * (inter_end-inter_start), cudaMemcpyDeviceToHost));
#endif
#endif
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&start, NULL);
#endif

	voff_t spid_offset = 0;
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset+i];
		voff_t num_mssgs = mst->roff[did][i+1] - mst->roff[did][i];
		if(i>0) spid_offset += (jnb_index_offset[i] - jnb_index_offset[i-1])/2 + (jnb_index_offset[i] - jnb_index_offset[i-1])%2;
		int num_of_blocks = (num_mssgs + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		pull_mssg_contig <<<block_size, THREADS_PER_BLOCK_NODES, 0, streams[did][0]>>> (num_mssgs, pid, jindex_offset[i], jnb_index_offset[i], spid_offset, 0, 1, k);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	pull_intra_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PULL MSSG FOR GPU GATHERING CONTIGS INTRA PROCESSOR TIME: ");
#endif
	gettimeofday (&overe, NULL);
	over_time[did] += (float)((overe.tv_sec * 1000000 + overe.tv_usec) - (overs.tv_sec * 1000000 + overs.tv_usec)) / 1000;

	return ((void*)0);
}


void * gather_contig_inter_pull_gpu (void * arg)
{
	evaltime_t overs, overe;
	lr_arg * carg = (lr_arg *) arg;
	comm_t * cm = carg->cm;
	master_t * mst = carg->mst;
	int k = carg->k;
	int did = carg->did;
	int num_of_partitions = mst->num_partitions[did + 1] - mst->num_partitions[did];
	int poffset = mst->num_partitions[did];
	voff_t * jindex_offset = mst->jindex_offset[did];
	voff_t * jnb_index_offset = mst->jnb_index_offset[did];
	if (mst->world_rank == 0)
		printf ("WORLD RANK %d: gather contig inter pull gpu %d:\n", mst->world_rank, did);

	CUDA_CHECK_RETURN(cudaSetDevice (did + DEVICE_SHIFT));

	gettimeofday (&overs, NULL);
	voff_t receive_start = mst->roff[did][num_of_partitions];
#ifdef MEASURE_MEMCPY_
	evaltime_t start, end;
	gettimeofday(&start, NULL);
#endif
	CUDA_CHECK_RETURN (cudaMemcpy(cm->receive_offsets, mst->soff[did], sizeof(voff_t) * (num_of_partitions + 1), cudaMemcpyHostToDevice));
	voff_t inter_size = mst->soff[did][num_of_partitions];
	printf ("WORLD RANK %d:::::::::::::inter size received for gathering contigs: %u\n", mst->world_rank, inter_size);
	CUDA_CHECK_RETURN (cudaMemcpy((unitig_t*)cm->send + receive_start, mst->send[did], sizeof(unitig_t) * inter_size, cudaMemcpyHostToDevice));
#ifdef MEASURE_MEMCPY_
	gettimeofday(&end, NULL);
	memcpyhd_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
#endif

#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&start, NULL);
#endif

	int i;
	voff_t spid_offset=0;
	for (i=0; i<num_of_partitions; i++)
	{
		if (inter_size == 0)
			break;
		int pid = mst->partition_list[poffset+i];
		voff_t num_mssgs = mst->soff[did][i+1] - mst->soff[did][i];
		if(i>0) spid_offset += (jnb_index_offset[i] - jnb_index_offset[i-1])/2 + (jnb_index_offset[i] - jnb_index_offset[i-1])%2;
		int num_of_blocks = (num_mssgs + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		pull_mssg_contig <<<block_size, THREADS_PER_BLOCK_NODES>>> (num_mssgs, pid, jindex_offset[i], jnb_index_offset[i], spid_offset, receive_start, 0, k);
	}
#ifdef MEASURE_TIME_
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	pull_inter_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& PULL MSSG FOR GPU GATHERING CONTIGS INTER PROCESSORS TIME: ");
#endif

	gettimeofday (&overe, NULL);
	over_time[did] += (float)((overe.tv_sec * 1000000 + overe.tv_usec) - (overs.tv_sec * 1000000 + overs.tv_usec)) / 1000;
	spid_offset = 0;
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset + i];
		voff_t size = mst->jid_offset[pid];
		if (i>0) spid_offset += (jnb_index_offset[i] - jnb_index_offset[i-1])/2 + (jnb_index_offset[i] - jnb_index_offset[i-1])%2;
		int num_of_blocks = (size + THREADS_PER_BLOCK_NODES - 1) / THREADS_PER_BLOCK_NODES;
		int block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;
		complete_contig_with_junction_gpu <<<block_size, THREADS_PER_BLOCK_NODES>>> (size, pid, jindex_offset[i], jnb_index_offset[i], spid_offset, k, cutoff);
	}

	return ((void*)0);
}

void init_contig_data_gpu (int did, d_jvs_t * jvs, d_lvs_t * lvs, meta_t * dm, master_t * mst)
{
	int * num_partitions = mst->num_partitions;
	int * partition_list = mst->partition_list;
	int num_of_partitions = num_partitions[did+1]-num_partitions[did];
	int total_num_partitions = mst->total_num_partitions;
	voff_t * jindex_offset = mst->jindex_offset[did];
	voff_t * jnb_index_offset = mst->jnb_index_offset[did];
	voff_t * joffsets = jvs->csr_offs_offs;
	vid_t * jnboffsets = jvs->csr_nbs_offs;
	voff_t * joffs = jvs->csr_offs;
	vid_t * jnbs = jvs->csr_nbs;
	uint * spids = jvs->csr_spids;
	voff_t * spid_offs = jvs->csr_spids_offs;

	int world_size = mst->world_size;
	int world_rank = mst->world_rank;
	int np_per_node;
	int np_node;
	get_np_node (&np_per_node, &np_node, total_num_partitions, world_size, world_rank);
	int start_partition_id = np_per_node*world_rank;

#ifdef SINGLE_NODE
	int num_of_devices = mst->num_of_devices;
	CUDA_CHECK_RETURN(cudaSetDevice (world_rank * num_of_devices + did));
#else
	CUDA_CHECK_RETURN(cudaSetDevice (did + DEVICE_SHIFT));
#endif
	int i;
	voff_t loffset = 0;
	voff_t joffset = 0;
	voff_t jnb_offset = 0;
	voff_t spids_offset = 0;
	for (i=0; i<num_of_partitions; i++)
	{
		int poffset = num_partitions[did];
		int pid = partition_list[poffset+i] - start_partition_id;
		voff_t lsize = lvs[pid].esize + lvs[pid].asize;
		voff_t jsize = jvs[pid].size;
		voff_t jnbsize = jnboffsets[pid+1] - jnboffsets[pid];
		voff_t spids_size = jnbsize/2 + jnbsize%2;
		CUDA_CHECK_RETURN (cudaMemcpy(dm->edge.post_edges+loffset, lvs[pid].post_edges, sizeof(edge_type)*lsize, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN (cudaMemcpy(dm->edge.pre_edges+loffset, lvs[pid].pre_edges, sizeof(edge_type)*lsize, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN (cudaMemcpy(dm->junct.kmers+joffset, jvs[pid].kmers, sizeof(kmer_t)*jsize, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN (cudaMemcpy(dm->junct.edges+joffset, jvs[pid].edges, sizeof(ull)*jsize, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN (cudaMemcpy(dm->junct.offs+joffset+i, joffs + joffsets[pid] + pid, sizeof(voff_t)*(jsize+1), cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN (cudaMemcpy(dm->junct.nbs+jnb_offset, jnbs + jnboffsets[pid], sizeof(vid_t)*jnbsize, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN (cudaMemcpy(dm->junct.spids+spids_offset, spids + spid_offs[pid], sizeof(uint)*spids_size, cudaMemcpyHostToDevice));
		jindex_offset[i] = joffset;
		jnb_index_offset[i] = jnb_offset;
		loffset += lsize;
		joffset += jsize;
		jnb_offset += jnbsize;
		spids_offset += spids_size;
	}
	jindex_offset[i] = joffset; // total number of junctions
	jnb_index_offset[i] = jnb_offset; // total number of neighbors of junctions

}

void
compact_dbgraph_contig (int num_of_partitions, subgraph_t * subgraph, d_jvs_t * djs, d_lvs_t * dls, meta_t * dm, master_t * mst, int k, int world_size, int world_rank)
{
	evaltime_t start, end;
	evaltime_t overs, overe; // measure big step in-memory processing time

	int num_of_cpus = mst->num_of_cpus;
	int num_of_devices = mst->num_of_devices;

	// the following should be deleted if mst is correct
	int np_per_node;
	int np_node;
	get_np_node (&np_per_node, &np_node, num_of_partitions, world_size, world_rank);

	int num_partitions_node = mst->num_partitions[num_of_cpus+num_of_devices];
	if (num_partitions_node != np_node)
	{
		printf ("error in mst->num_partitions, num_partitions_node=%d, np_node=%d!!!\n", num_partitions_node, np_node);
//		exit(0);
	}

	gettimeofday(&start, NULL);

	read_kmers_edges_for_gather_contig (num_of_partitions, djs, dls, mst, world_size, world_rank);

	junction_csr (djs, num_partitions_node, mst, subgraph);
	get_junction_info_processors (mst, subgraph); // get info of mst->jnbs and mst->js

	set_globals_graph_compute_gpu (dm, mst); // set global variables again, same as comm.cu
	realloc_device_edges (dm, mst);
	set_edges_gpu (dm, mst);

	realloc_host_edges (dm, mst);

	realloc_device_junctions (dm, mst);
	set_junctions_gpu (dm, mst);

	realloc_host_junctions (dm, mst);

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
		init_contig_data_gpu (i, djs, dls, &dm[i], mst);
	}
	for (i=num_of_devices; i<num_of_devices+num_of_cpus; i++)
	{
		init_contig_data_cpu (i, djs, dls, &dm[i], mst);
	}
	gettimeofday (&end, NULL);
	if (mst->world_rank == 0)
		print_exec_time (start, end, "WORLD RANK %d: IIIIIIIIIIIIIIInit graph data input time: \n", mst->world_rank);

	pthread_t cpu_threads[NUM_OF_CPUS];
	pthread_t gpu_threads[NUM_OF_DEVICES];
	lr_arg arg[NUM_OF_DEVICES + NUM_OF_CPUS];

#ifndef SYNC_ALL2ALL_
	pthread_t comm_thread;
	comm_arg cm_arg;
	cm_arg.mst=mst;
#endif
	create_streams(num_of_devices, 2);
	init_lock_flag();

	mst->mssg_size = sizeof(query_t);
	for (i = 0; i < num_of_devices + num_of_cpus; i++)
	{
		arg[i].did = i;
		arg[i].cm = &dm[i].comm;
		arg[i].mst = mst;
		arg[i].dm = &dm[i];
		arg[i].k = k;
#ifndef SYNC_ALL2ALL_
		cm_arg.cm[i] = &dm[i].comm;
#endif
	}
//	gettimeofday (&end, NULL);
	if (mst->world_rank == 0)
		print_exec_time (start, end, "WORLD RANK %d: #################### global initiation time: ", mst->world_rank);


	float all2all_time = 0;
	//************** Compact de bruijn graph and update junctions here: ***************
	gettimeofday (&overs, NULL);
	for (i = 0; i < num_of_devices; i++)
	{
		if (pthread_create (&gpu_threads[i], NULL, compact_push_update_intra_push_gpu, &arg[i]) != 0)
		{
			printf ("create thread for compact push update intra push on gpu %d error!\n", i);
		}
	}
	for (i = 0; i < num_of_cpus; i++)
	{
		if (pthread_create (&cpu_threads[i], NULL, compact_push_update_intra_push_cpu, &arg[num_of_devices + i]) != 0)
		{
			printf ("create thread for compact push update intra push on cpu %d error!\n", i);
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
			printf ("join thread on compact push update intra push on gpu %d failure!\n", i);
		}
	}
	for (i = 0; i < num_of_cpus; i++)
	{
		if (pthread_join (cpu_threads[i], NULL) != 0)
		{
			printf ("Join thread on compact push update intra push on cpu %d failure!\n", i);
		}
	}
#ifndef SYNC_ALL2ALL_
		if (pthread_join (comm_thread, NULL) != 0)
		{
			printf ("Join communication thread failure!\n");
		}
#else
	gettimeofday (&start, NULL);
	master_all2all(mst);
	gettimeofday (&end, NULL);
	all2all_time += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
#endif
	if (mst->world_rank == 0)
		print_exec_time (start, end, "WORLD RANK %d: TICKTOCK TICKTOCK TICKTOCK:: master all to all time after listrank push: ", mst->world_rank);

//	while(debug) {}
	mst->mssg_size = sizeof(compact_t); // reset mssg size
	for (i = 0; i < num_of_devices; i++)
	{
		if (pthread_create (&gpu_threads[i], NULL, compact_pull_update_inter_push_intra_pull_gpu, &arg[i]) != 0)
		{
			printf ("create thread for compact pull update inter upsh intra pull on gpu %d error!\n", i);
		}
	}
	for (i = 0; i < num_of_cpus; i++)
	{
		if (pthread_create (&cpu_threads[i], NULL, compact_pull_update_inter_push_intra_pull_cpu, &arg[num_of_devices + i]) != 0)
		{
			printf ("create thread for compact pull update inter push intra pull on cpu %d error!\n", i);
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
			printf ("join thread on compact pull update inter push intra pull on gpu %d failure!\n", i);
		}
	}
	for (i = 0; i < num_of_cpus; i++)
	{
		if (pthread_join (cpu_threads[i], NULL) != 0)
		{
			printf ("Join thread on compact pull update inter push intra pull on cpu %d failure!\n", i);
		}
	}
#ifndef SYNC_ALL2ALL_
		if (pthread_join (comm_thread, NULL) != 0)
		{
			printf ("Join communication thread failure!\n");
		}
#else
	gettimeofday (&start, NULL);
	master_all2all(mst);
	gettimeofday (&end, NULL);
	all2all_time += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
#endif
	if (mst->world_rank == 0)
		print_exec_time (start, end, "WORLD RANK %d: TICKTOCK TICKTOCK TICKTOCK:: master all to all time after compact_pull_update_inter_push_intra_pull: ", mst->world_rank);

	for (i = 0; i < num_of_devices; i++)
	{
		if (pthread_create (&gpu_threads[i], NULL, update_inter_pull_gpu, &arg[i]) != 0)
		{
			printf ("create thread on update inter pull on gpu %d error!\n", i);
		}
	}
	for (i = 0; i < num_of_cpus; i++)
	{
		if (pthread_create (&cpu_threads[i], NULL, update_inter_pull_cpu, &arg[num_of_devices + i]) != 0)
		{
			printf ("create thread for update inter pull on cpu %d error!\n", i);
		}
	}
	for (i = 0; i < num_of_devices; i++)
	{
		if (pthread_join (gpu_threads[i], NULL) != 0)
		{
			printf ("Join thread on update inter pull on gpu %d error!\n", i);
		}
	}
	for (i = 0; i < num_of_cpus; i++)
	{
		if (pthread_join (cpu_threads[i], NULL) != 0)
		{
			printf ("Join thread on update inter pull on cpu %d failure!\n", i);
		}
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
		if (mst->world_rank == 0)
		{
		printf ("WORLD RANK %d: &&&&&&&&&&& PUSH_OFFSET TIME FOR GPU %d: %f &&&&&&&&&&&&&\n", mst->world_rank, i, push_offset_time[i]);
		printf ("WORLD RANK %d: &&&&&&&&&&& PUSH TIME FOR GPU %d: %f &&&&&&&&&&&&&&&&&&\n", mst->world_rank, i, push_time[i]);
		printf ("WORLD RANK %d: &&&&&&&&&&& PULL INTRA TIME FOR GPU %d: %f &&&&&&&&&&&&&&\n", mst->world_rank, i, pull_intra_time[i]);
		printf ("WORLD RANK %d: &&&&&&&&&&& PULL INTER TIME FOR GPU %d: %f &&&&&&&&&&&&&&\n", mst->world_rank, i, pull_inter_time[i]);
		printf ("WORLD RANK %d: &&&&&&&&&&& memcpy device to host time for GPU %d: %f &&&&&&&&&&&&&\n", mst->world_rank, i, memcpydh_time[i]);
		printf ("WORLD RANK %d: &&&&&&&&&&& memcpy host to device time for GPU %d: %f &&&&&&&&&&&&&\n", mst->world_rank, i, memcpyhd_time[i]);
		}
#endif
		push_offset_time[i]=0;
		push_time[i]=0;
		pull_intra_time[i]=0;
		pull_inter_time[i]=0;
		memcpydh_time[i]=0;
		memcpyhd_time[i]=0;
	}
#ifdef MEASURE_TIME_
	if (mst->world_rank == 0)
	{
	printf ("WORLD RANK %d: &&&&&&&&&&& PUSH_OFFSET TIME FOR CPU: %f &&&&&&&&&&&&&\n", mst->world_rank, push_offset_time[i]);
	printf ("WORLD RANK %d: &&&&&&&&&&& PUSH TIME FOR CPU: %f &&&&&&&&&&&&&&&&&&\n", mst->world_rank, push_time[i]);
	printf ("WORLD RANK %d: &&&&&&&&&&& PULL INTRA TIME FOR CPU: %f &&&&&&&&&&&&&&\n", mst->world_rank, pull_intra_time[i]);
	printf ("WORLD RANK %d: &&&&&&&&&&& PULL INTER TIME FOR CPU: %f &&&&&&&&&&&&&&\n", mst->world_rank, pull_inter_time[i]);
	}
#endif
	push_offset_time[i]=0;
	push_time[i]=0;
	pull_intra_time[i]=0;
	pull_inter_time[i]=0;
	gettimeofday(&overe, NULL);
	inmemory_time += (float)((overe.tv_sec * 1000000 + overe.tv_usec) - (overs.tv_sec * 1000000 + overs.tv_usec)) / 1000; // recording in-memory processing time

	if (mst->world_rank == 0)
	{
		print_exec_time(overs, overe, "WORLD RANK %d: CCCCCCCCCCCCCCompact graph update junction time: ", mst->world_rank);
		printf ("WORLD RANK %d: ~~~~~~~~~~~~~~~~ ALLTOALL TIME MEASURED: %f\n", mst->world_rank, all2all_time);
	}

	for (i=0; i<num_of_cpus+num_of_devices; i++)
	{
		if (mst->world_rank == 0)
			printf ("+++++++++++ WORLD RANK %d: overall computation time on device %d::::::::::: %f\n", world_rank, i, over_time[i]);
		over_time[i]=0;
	}
	 // ************ free writing offset buffer for pull and push mode
	for (i=0; i<num_of_devices; i++)
		free_pull_push_offset_gpu(dm[i].comm.extra_send_offsets);
	for (i=num_of_devices; i<num_of_devices+num_of_cpus; i++)
		free_pull_push_offset_cpu(dm[i].comm.extra_send_offsets);

	// *********** get max_num_unitigs for prefix sum of ulens **************
	voff_t max_num_unitigs = 0;
	size_t max_total_len = 0;
	for (i=0; i<num_of_devices; i++)
	{
		int num_partitions_device = mst->num_partitions[i+1] - mst->num_partitions[i];
		int j;
		for (j=0; j<num_partitions_device; j++)
		{
			voff_t num_unitigs = mst->jnb_index_offset[i][j+1] - mst->jnb_index_offset[i][j];
			if (max_num_unitigs < num_unitigs)
				max_num_unitigs = num_unitigs;
		}
	}
	size_t * tmp_ulens = (size_t *) malloc (sizeof(size_t) * max_num_unitigs);
	size_t * tmp_offs = (size_t *) malloc (sizeof(size_t) * num_of_partitions);
	for (i=0; i<num_of_devices; i++)
	{
		tmp_offs[0] = 0;
		// gpu inclusive prefix sum for unitig lengths, record max_num_nbs and max_total_ulens at the same time!
		int num_partitions_device = mst->num_partitions[i+1] - mst->num_partitions[i];
		size_t total_lens = 0;
		int j;
		for (j=0; j<num_partitions_device; j++)
		{
			voff_t num_unitigs = mst->jnb_index_offset[i][j+1] - mst->jnb_index_offset[i][j];
			CUDA_CHECK_RETURN (cudaMemcpy(tmp_ulens, dm[i].junct.ulens + mst->jnb_index_offset[i][j], sizeof(size_t) * num_unitigs, cudaMemcpyDeviceToHost));
			tbb_scan_long (tmp_ulens, tmp_ulens, num_unitigs);
			CUDA_CHECK_RETURN (cudaMemcpy(dm[i].junct.ulens + mst->jnb_index_offset[i][j], tmp_ulens, sizeof(size_t) * num_unitigs, cudaMemcpyHostToDevice));
			total_lens += tmp_ulens[num_unitigs-1];
			tmp_offs[j+1] = total_lens;
			if (max_total_len < tmp_ulens[num_unitigs-1])
				max_total_len = tmp_ulens[num_unitigs-1];
		}
		malloc_unitig_buffer_gpu (&dm[i], total_lens, i, num_of_devices, num_partitions_device);
		set_unitig_pointer_gpu (&dm[i], i, mst);
		CUDA_CHECK_RETURN (cudaMemcpy(dm[i].junct.unitig_offs, tmp_offs, sizeof(size_t)*(num_partitions_device+1), cudaMemcpyHostToDevice));
	}
	free (tmp_ulens);

	for (i=num_of_devices; i<num_of_devices+num_of_cpus; i++)
	{
		tmp_offs[0] = 0;
		int num_partitions_device = mst->num_partitions[i+1] - mst->num_partitions[i];
		size_t total_lens = 0;
		int j;
		for (j=0; j<num_partitions_device; j++)
		{
			voff_t num_unitigs = mst->jnb_index_offset[i][j+1] - mst->jnb_index_offset[i][j];
			tbb_scan_long (dm[i].junct.ulens + mst->jnb_index_offset[i][j], dm[i].junct.ulens + mst->jnb_index_offset[i][j], num_unitigs);
			total_lens += (dm[i].junct.ulens + mst->jnb_index_offset[i][j])[num_unitigs-1];
			tmp_offs[j+1] = total_lens;
		}
		malloc_unitig_buffer_cpu (&dm[i], total_lens, num_partitions_device);
		set_unitig_pointer_cpu (&dm[i]);
		memcpy (dm[i].junct.unitig_offs, tmp_offs, sizeof(size_t) * (num_partitions_device+1));
	}

	// **************** GATHERING CONTIGS BEGINS ******************
	gettimeofday (&overs, NULL);
	mst->mssg_size = sizeof(unitig_t);
	for (i=0; i<num_of_devices; i++)
	{
		if (pthread_create (&gpu_threads[i], NULL, gather_contig_push_intra_pull_gpu, &arg[i]) != 0)
		{
			printf ("Create thread for gather contig push intra pull on gpu %d error!\n", i);
		}
	}
	for (i=0; i<num_of_cpus; i++)
	{
		if (pthread_create (&cpu_threads[i], NULL, gather_contig_push_intra_pull_cpu, &arg[num_of_devices+i]) != 0)
		{
			printf ("Create thread for gather contig push intra pull on cpu error!\n");
		}
	}
#ifndef SYNC_ALL2ALL_
		if (pthread_create (&comm_thread, NULL, master_all2all_async, &cm_arg) != 0)
		{
			printf ("Create thread for communication error!\n");
		}
#endif
	for (i=0; i<num_of_devices; i++)
	{
		if (pthread_join (gpu_threads[i], NULL) != 0)
		{
			printf ("Join thread on gather contig push intra pull on gpu %d error!\n", i);
		}
	}
	for (i=0; i<num_of_cpus; i++)
	{
		if (pthread_join (cpu_threads[i], NULL) != 0)
		{
			printf ("Join thread on gather contig push intra pull on cpu error!\n");
		}
	}
#ifndef SYNC_ALL2ALL_
		if (pthread_join (comm_thread, NULL) != 0)
		{
			printf ("Join communication thread failure!\n");
		}
#else
	gettimeofday (&start, NULL);
	master_all2all(mst);
	gettimeofday (&end, NULL);
	all2all_time += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
#endif
	if (mst->world_rank == 0)
		print_exec_time (start, end, "WORLD RANK %d: TICKTOCK TICKTOCK TICKTOCK:: master all to all time after gather_contig_push_intra_pull push: ", mst->world_rank);

	for (i=0; i<num_of_devices; i++)
	{
		if (pthread_create (&gpu_threads[i], NULL, gather_contig_inter_pull_gpu, &arg[i]) != 0)
		{
			printf ("Create thread for gather contig inter pull on gpu %d error!\n", i);
		}
	}
	for (i=0; i<num_of_cpus; i++)
	{
		if (pthread_create (&cpu_threads[i], NULL, gather_contig_inter_pull_cpu, &arg[num_of_devices+i]) != 0)
		{
			printf ("Create thread for gather contig inter pull on cpu error!\n");
		}
	}
	for (i=0; i<num_of_devices; i++)
	{
		if (pthread_join (gpu_threads[i], NULL) != 0)
		{
			printf ("Join thread on gather contig inter pull on gpu %d error!\n", i);
		}
	}
	for (i=0; i<num_of_cpus; i++)
	{
		if (pthread_join (cpu_threads[i], NULL) != 0)
		{
			printf ("Join thread on gather contig inter pull on cpu error!\n");
		}
	}
	gettimeofday (&overe, NULL);
	inmemory_time += (float)((overe.tv_sec * 1000000 + overe.tv_usec) - (overs.tv_sec * 1000000 + overs.tv_usec)) / 1000;

	// ************** write contigs to disk files: ***************
	for (i=0; i<num_of_devices; i++)
	{
		write_contigs_gpu (&dm[i], mst, i, max_num_unitigs, max_total_len, k);
	}
	for (i=num_of_devices; i<num_of_devices+num_of_cpus; i++)
	{
		write_contigs_cpu (&dm[i], mst, i, k);
		finalize_contig_data_cpu ();
	}

	free_junction_csr (djs, subgraph);
	free_device_realloc (dm, mst);
	free_host_realloc (dm, mst);

	// ********* malloc was in reading files in io **********
	free_adj_junction (np_node, djs);
	free_adj_linear (np_node, dls);
	free_kmers_edges_after_contig (np_node, djs, dls);
	for (i=0; i<num_of_cpus; i++)
	{
		free_unitig_buffer_cpu (dm, mst);
	}
	for (i=0; i<num_of_cpus; i++)
	{
		free_unitig_buffer_gpu (dm, mst);
	}

	destroy_streams(num_of_devices, 2);
}
}
