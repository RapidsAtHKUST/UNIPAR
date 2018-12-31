/*
 * preprocess.h
 *
 *  Created on: 2018-4-11
 *      Author: qiushuang
 */

#ifndef PREPROCESS_H_
#define PREPROCESS_H_

#include "comm.h"
#include "graph.h"

#define INT_BYTES 4 // BE careful: if the number of bytes for a int is not 4, code of assid_t should be changed!
#define REVERSE_FLAG 0xff000000 // flag to set the direction of an edge
#define ID_BITS 0xffff // last 16 bits to store msp id
#define DEADEND 0xffffffff

#define INTER_BUF_FACTOR 1 //0.2

typedef uch edge_type;


#ifdef __cplusplus
extern "C"
{
void init_write_buffer (int num_of_devices);
void finalize_write_buffer (int num_of_devices);
void write_junctions_gpu (dbmeta_t * dbm, master_t * mst, uint jsize, uint lsize, int pid, int total_num_partitions, int did);
void write_linear_vertices_gpu (dbmeta_t * dbm, master_t * mst, uint jsize, uint lsize, int pid, int total_num_partitions, int did);
void write_ids_gpu (dbmeta_t * dbm, master_t * mst, uint total_num_partitions, int did);
void init_preprocessing_data_cpu (dbmeta_t * dbm, int num_of_partitions);
void output_vertices_gpu (dbmeta_t * dbm, master_t * mst, voff_t jsize, voff_t lsize, int pid, int total_num_partitions, int did, d_jvs_t * djs, d_lvs_t * dls, subgraph_t * subgraph);
void write_kmers_edges_gpu (dbmeta_t * dbm, master_t * mst, voff_t jsize, voff_t lsize, int pid, int total_num_partitions, int did);
void reset_globals_gather_cpu (dbmeta_t * dbm);
void finalize_preprocessing_data_cpu (void);
void init_binary_data_cpu (int did, master_t * mst, dbmeta_t * dbm, dbtable_t * tbs);
void init_hashtab_data_cpu (int did, master_t * mst, dbmeta_t * dbm, dbtable_t * tbs);
void init_binary_data_cpu_sorted (int did, master_t * mst, dbmeta_t * dbm, dbtable_t * tbs);
void finalize_hashtab_data_cpu (void);

void * shakehands_push_respond_intra_push_cpu (void * arg);
void * shakehands_pull_respond_inter_push_intra_pull_cpu (void * arg);
void * respond_inter_pull_cpu (void * arg);
void * neighbor_push_intra_pull_cpu (void * arg);
void * neighbor_inter_pull_cpu (void * arg);
void * identify_vertices_cpu (void * arg);
void * assign_vertex_ids_cpu (void * arg);
void * gather_vertices_cpu (void * arg);

void free_dbgraph_hashtab (int num_of_partitions, dbtable_t * tbs);
void tbb_pair_sort (pair_t * buf, uint size);
}
#endif
#endif /* PREPROCESS_H_ */
