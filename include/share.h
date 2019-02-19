/*
 * share.h
 *
 *  Created on: 2017-8-23
 *      Author: qiushuang
 */

#include "dbgraph.h"
#include "graph.h"
#include "comm.h"

#ifndef SHARE_H_
#define SHARE_H_

#ifdef LONG_KMER
#define get_kmer(kmer, record) {\
	kmer.x = record.x;			\
	kmer.y = record.y;			\
	kmer.z = record.z;			\
	kmer.w = record.w;			\
}

#define equal_kmer(kmer1, kmer2) { (kmer1.x == kmer2.x) && (kmer1.y == kmer2.y) && (kmer1.z == kmer2.z) && (kmer1.w == kmer2.w) }

#else
#define get_kmer(kmer, record) {\
	kmer.x = record.x;			\
	kmer.y = record.y;			\
}
#define equal_kmer(kmer1, kmer2)  ((kmer1.x == kmer2.x) && (kmer1.y == kmer2.y))
#endif

#define find_max(array, num, max_val) {\
	int i;								\
	max_val = array[0];					\
	for (i = 1; i < num; i++) {			\
		if (array[i] > max_val)	{		\
			max_val = array[i]; }		\
	}}


#ifdef __cplusplus
extern "C"
{
void inclusive_prefix_sum (int * array, int num);
msp_id_t get_partition_id_cpu (minstr_t minstr, int p, int num_of_partitions);
voff_t get_max (subsize_t * subs, goffset_t * joff, goffset_t * toff, voff_t * max_sub_size, voff_t * max_jsize, voff_t * max_lsize, int intra_num_of_partitions, int total_num_of_partitions);
void get_subgraph_sizes (subgraph_t * subgraph, int num_of_partitions);
int query_partition_id_from_idoffsets (vid_t id, int num_of_partitions, goffset_t * id_offsets);
void get_global_offsets (voff_t * goff, voff_t * loff, int num_of_partitions, int cpu_threads);
void init_mssg_count (voff_t * intra_mssgs, voff_t * inter_mssgs);
void get_mssg_count (master_t * mst, voff_t * intra_mssgs, voff_t * inter_mssgs, int iter);
void print_mssg_count (int num_of_procs, voff_t * intra_mssgs, voff_t * inter_mssgs, int iters);
void print_offsets (voff_t * array, int num);
void write_hashtab (char * file_dir, node_t * tab, voff_t size, voff_t elem_hashed, int pid, int total_num_partitions, int did);
void write_ids_cpu (dbmeta_t * dbm, master_t * mst, voff_t total_num_partitions, int did);
void output_vertices_cpu (dbmeta_t * dbm, master_t * mst, voff_t jsize, voff_t lsize, int pid, int total_num_partitions, int did, d_jvs_t * djs, d_lvs_t * dls, subgraph_t * subgraph);
void write_kmers_edges_cpu (dbmeta_t * dbm, master_t * mst, voff_t jsize, voff_t lsize, int pid, int total_num_partitions, int did);
void write_junctions_cpu (dbmeta_t * dbm, master_t * mst, voff_t jsize, voff_t lsize, int pid, int total_num_partitions, int did);
void write_linear_vertices_cpu (dbmeta_t * dbm, master_t * mst, voff_t jsize, voff_t lsize, int pid, int total_num_partitions, int did);
void write_contigs_cpu (meta_t * dm, master_t * mst, int did, int k);

/* TBB SORT FUNCTIONS: */
void inclusive_prefix_sum_long (unsigned long long * array, int num);
void tbb_kmer_vid_sort (kmer_vid_t * buf, voff_t size);
void tbb_vertex_sort (vertex_t * mssg_buf, uint size);
void tbb_entry_sort (entry_t * mssg_buf, uint size);
void tbb_scan_long (size_t * input, size_t * output, size_t num);
void tbb_scan_uint (voff_t * input, voff_t * output, size_t num);
}
#endif

void inclusive_prefix_sum (int * array, int num);
int query_partition_id_from_idoffsets (vid_t id, int num_of_partitions, goffset_t * id_offsets);
void get_global_offsets (voff_t * goff, voff_t * loff, int num_of_partitions, int cpu_threads);
void output_vertices_cpu (dbmeta_t * dbm, master_t * mst, voff_t jsize, voff_t lsize, int pid, int total_num_partitions, int did, d_jvs_t * djs, d_lvs_t * dls, subgraph_t * subgraph);
void write_kmers_edges_cpu (dbmeta_t * dbm, master_t * mst, voff_t jsize, voff_t lsize, int pid, int total_num_partitions, int did);

void inclusive_prefix_sum_long (unsigned long long * array, int num);
void tbb_kmer_vid_sort (kmer_vid_t * buf, voff_t size);
void tbb_vertex_sort (vertex_t * mssg_buf, uint size);
void tbb_entry_sort (entry_t * mssg_buf, uint size);
void tbb_scan_long (size_t * input, size_t * output, size_t num);
void tbb_scan_uint (voff_t * input, voff_t * output, size_t num);

#endif /* SHARE_H_ */
