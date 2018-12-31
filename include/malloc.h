/*
 * malloc.h
 *
 *  Created on: 2018-10-7
 *      Author: qiushuang
 */

#ifndef MALLOC_H_
#define MALLOC_H_

#include "comm.h"
#include "dbgraph.h"
#include "graph.h"
#include "preprocess.h"


#define MSSG_FACTOR 1.1
#ifdef BINARY_GPU_
#define HASH_LOAD_FACTOR_GPU 1
#define BINARY_FACTOR 1//0.7
#else
#define HASH_LOAD_FACTOR_GPU 1
#define BINARY_FACTOR 1//0.7 //IT IS THE INPUT HASH TABLE LOAD_FACTOR
#endif

#ifdef BINARY_CPU_
#define HASH_LOAD_FACTOR 1
#define BINARY_FACTOR_CPU 1//0.7
#else
#define HASH_LOAD_FACTOR 1
#define BINARY_FACTOR_CPU 1//0.7
#endif


#ifdef __cplusplus
extern "C"
{
void malloc_subgraphs (subgraph_t * subgraph, int num_of_partitions);
size_t get_total_size_subgraphs (subgraph_t * subgraph);
void free_subgraphs (subgraph_t * subgraph);
void init_host_filter2 (dbmeta_t * dbm, master_t * mst, vid_t max_subsize);
void finalize_host_filter2 (dbmeta_t *dbm, master_t *mst);
void set_id_offsets_cpu (dbmeta_t * dbm, master_t * mst);
void init_host_preprocessing (dbmeta_t * dbm, master_t * mst);
void finalize_host_preprocessing (dbmeta_t * dbm, master_t * mst);
void init_host_gather (dbmeta_t * dbm, master_t * mst, vid_t max_subsize, vid_t max_jsize, vid_t max_lsize);
void finalize_host_gather2 (dbmeta_t * dbm, master_t * mst);
void init_host_graph_compute (meta_t * dm, master_t * mst);
void finalize_host_graph_compute (meta_t * dm, master_t * mst);
void malloc_pull_push_offset_cpu (voff_t ** extra_send_offsets, master_t * mst);
void free_pull_push_offset_cpu (voff_t * extra_send_offsets);
void malloc_pull_push_receive_cpu (comm_t * dm, uint mssg_size, int did, voff_t num_of_mssgs, int expand);
void free_pull_push_receive_cpu (comm_t * dm);
void realloc_host_edges (meta_t * dm, master_t * mst);
void realloc_host_junctions (meta_t * dm, master_t * mst);
void free_host_realloc (meta_t * dm, master_t * mst);
void collect_free_memory_cpu (meta_t * dm, master_t * mst);
void malloc_unitig_buffer_cpu (meta_t * dm, size_t size, int num_of_partitions);
void free_unitig_buffer_cpu (meta_t * dm, master_t * mst);
void free_pull_push_receive_cpu (comm_t * dm);
void free_subgraph_num_jnbs (subgraph_t * subgraph);
void malloc_subgraph_subgraphs (subgraph_t * subgraph, int num_of_partitions);
void free_subgraph_subgraphs (subgraph_t * subgraph);
int get_device_config (void);
}
#endif

void malloc_pull_push_receive_cpu (comm_t * dm, uint mssg_size, int did, voff_t num_of_mssgs, int expand);
void free_pull_push_receive_cpu (comm_t * dm);
void free_subgraph_num_jnbs (subgraph_t * subgraph);
void malloc_subgraph_subgraphs (subgraph_t * subgraph, int num_of_partitions);
void free_subgraph_subgraphs (subgraph_t * subgraph);
int get_device_config (void);

#endif /* MALLOC_H_ */
