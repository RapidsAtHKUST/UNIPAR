/*
 * distribute.h
 *
 *  Created on: 2018-4-16
 *      Author: qiushuang
 */

#ifndef DISTRIBUTE_H_
#define DISTRIBUTE_H_

#include "comm.h"


#ifdef __cplusplus
extern "C"
{
void get_np_node (int * np_per_node, int * np_node, int total_num_partitions, int world_size, int world_rank);

void mpi_barrier (void);
int mpi_allsum (const void * send, void * receive);

void init_counts_displs (int total_num_of_partitions, int world_size, int world_rank);
void finalize_counts_displs (void);

void init_distribute_partitions (int total_num_of_partitions, master_t * mst, int world_size);
void init_temp_offset_counts (int total_num_of_partitions, master_t * mst);
void * init_count_displs_receive_all2all (int total_num_of_partitions, master_t * mst, voff_t * gathered_offsets);
void * mpi_all2allv (void * send, void * receive, int total_num_of_partitions, int mssg_size, int world_size, int world_rank);
void finalize_receive_all2all (int);
void * mpi_allgatherv (void * send, void * receive, int total_num_of_partitions, int world_size, int world_rank);
void * mpi_allgatherv_inplace (void * send, void * receive, int total_num_of_partitions, int world_size, int world_rank);
void * mpi_allgather_offsets (int total_num_of_partitions, int world_size, int world_rank);
void distribute_partitions (int total_num_of_partitions, master_t * mst, subgraph_t * subgraph, ptype partitioning, int world_size, int world_rank, ull graph_size, double mssg_size);
void finalize_distribute_partitions (master_t * mst);

size_t get_device_memory_size (int did); // in malloc.cuh

// ******* from all2all.cu file ***************
long long master_all2all (master_t * mst);
void * master_all2all_async (void * arg);
}
#endif

void get_np_node (int * np_per_node, int * np_node, int total_num_partitions, int world_size, int world_rank);
void init_counts_displs (int total_num_of_partitions, int world_size, int world_rank);
void finalize_counts_displs (void);
void mpi_barrier (void);
void * mpi_allgatherv (void * send, void * receive, int total_num_of_partitions, int world_size, int world_rank);
size_t get_device_memory_size (int did); // in malloc.cuh
int mpi_init (int * provided, int * world_size, int * world_rank);
void mpi_finalize (void);

#endif /* DISTRIBUTE_H_ */
