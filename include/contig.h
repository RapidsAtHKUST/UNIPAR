/*
 * contig.h
 *
 *  Created on: 2018-10-4
 *      Author: qiushuang
 */

#ifndef CONTIG_H_
#define CONTIG_H_

#include <math.h>
#include "comm.h"
#include "graph.h"


#ifdef __cplusplus
extern "C"
{
void set_unitig_pointer_cpu (meta_t * dm);
void finalize_contig_data_cpu (void);
void get_junction_info_processors (master_t * mst, subgraph_t * subgraph);
void init_contig_data_cpu (int did, d_jvs_t * jvs, d_lvs_t * lvs, meta_t * dm, master_t * mst);
void * compact_push_update_intra_push_cpu (void * arg);
void * compact_pull_update_inter_push_intra_pull_cpu (void * arg);
void * update_inter_pull_cpu (void * arg);
void * gather_contig_push_intra_pull_cpu (void * arg);
void * gather_contig_inter_pull_cpu (void * arg);
void write_contigs_gpu (meta_t * dm, master_t * mst, int did, voff_t max_num, size_t max_total_len, int k);
void read_kmers_edges_for_gather_contig (int total_num_partitions, d_jvs_t * js, d_lvs_t * ls, master_t * mst, int world_size, int world_rank);
void junction_csr (d_jvs_t * js, int num_of_partitions, master_t * mst, subgraph_t * subgraph);
void get_junction_info_processors (master_t * mst, subgraph_t * subgraph);
void tbb_scan_long (size_t * input, size_t * output, size_t num);
void write_contigs_cpu (meta_t * dm, master_t * mst, int did, int k);
void free_junction_csr (d_jvs_t * js, subgraph_t * subgraph);
void free_adj_junction (int num_of_partitions, d_jvs_t * djs);
void free_kmers_edges_after_contig (int num_of_partitions, d_jvs_t * djs, d_lvs_t * dls);
}
#endif
#endif /* CONTIG_H_ */
