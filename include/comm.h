/*
 * comm.h
 *
 *  Created on: 2017-7-14
 *      Author: qiushuang
 */

#ifndef COMM_H_
#define COMM_H_
#define SYNC_ALL2ALL_

#include "utility.h"
#include "dbgraph.h"
#include "graph.h"

#define NUM_OF_PROCS 5
#define MAX_NUM_NODES_DEVICE 188743680 // 180M (52428800*4) // 200M this macro determines the total number of vertices in each GPU
#define MAX_NUM_NODES_DEVICE_HASH 209715200 //188743680 //157286400 //83886080 //104857600
#define MAX_NUM_ITERATION 15
#define TOTAL_OVER_INTRA_FACTOR 20 //6 for random-human14 //10 for hg7
#define TOTAL_OVER_INTRA_FACTOR_CPU 12 //10 for random-human14 //2 for hg7
#define THRESHOLD 100

//#define MEASURE_TIME_
//#define MEASURE_MEMCPY_

//#define CONTIG_CHECK

#define MAX_VID 4294967295
#define MOST_BIT 0x80000000

#define DEFAULT_SEED 3735928559

#define max_in_array(arr, n, max) {\
	int i; 							\
	max = arr[0];					\
	for (i=1;i<n;i++) {				\
		if(max<arr[i]) {			\
			max=arr[i]; }}			\
}

typedef enum ptype {even, uneven} ptype; // choose for distributing subgraphs to processors in a computer node

typedef uint k_t;
typedef int v_t;
typedef struct kv {
	k_t k;
	v_t v;
}kv_t;

typedef struct path {
	vid_t dst; // destination vertex, being current post or pre
	vid_t opps; // opposite direction of neighbors
	vid_t jid; // if opps == MAX_VID; this direction meets a junction
	vid_t cid; // current vertex id
	voff_t dist; // distance of bwd or fwd
} path_t; // unipaths of each vertex

typedef struct selfloop {
	vid_t v;
	vid_t dst;
} selfloop_t;

typedef struct rank {
	path_t p1;
	path_t p2;
} rank_t; // rank of a vertex in a unipath

typedef struct query {
	vid_t jid; // junction id with this query
	vid_t nid; // linear neighbor id of the junction
} query_t; // query neighbor with nid from junction jid

typedef struct compact {
	vid_t nid; // linear neighbor of the junction
	vid_t jid; // junction id with this query
	vid_t ojid;
	int plen; // length of this unitig
//	int rank; // rank of this vertex in the unitig
} compact_t; // push back opposite junction id for a query from a junction

typedef struct unitig {
	vid_t jid; // junction id (an endpoint)
	vid_t ojid; // the opposite junction id (another endpoint)
	voff_t rank; // encode edge into the first two bits of rank, rank of the linear vertex in the unitig
	voff_t len; // total length of this unitig, used to identify different paths with the same jids
} unitig_t;

typedef struct cmm
{
	int * id2index;
	voff_t * send_offsets;
	voff_t * receive_offsets;
	voff_t * extra_send_offsets;
	voff_t * tmp_send_offsets;
	voff_t * index_offsets;
	void * send;
	void * receive;
	void * dtemp; // temporary device memory for gpu sort
	size_t temp_size;
} comm_t; //communication for a processor

typedef struct meta
{
	comm_t comm;
	edge_t edge; // record info for linear vertices
	csr_t junct; // record info for junctions
	vid_t * id_offsets; // vertex id offsets for partitions, copied from master; it points to the memory on device to store id_offsets
	vid_t * jid_offset; // junction size for partitions, copied from master; it points to the memory on device to store jid_offset
	vid_t * dvt;
} meta_t; // meta data to manage communication and graph computation, used in listranking and contig workflow

typedef struct master
{
	vid_t num_vs[NUM_OF_PROCS]; // total number of linear vertices in each processor
	vid_t num_js[NUM_OF_PROCS]; // total number of junctions in each processor
	vid_t num_jnbs[NUM_OF_PROCS]; // total number of neighbors for junctions in each processor
	voff_t * soff[NUM_OF_PROCS]; // send offsets
	voff_t * roff[NUM_OF_PROCS]; // receive offsets
	void * send[NUM_OF_PROCS]; // send buffer of master
	void * receive[NUM_OF_PROCS]; // receive buffer of master
	int * not_reside[NUM_OF_PROCS]; // partitions that not reside in a processor, initialized in distribute_partitions
	int * id2index[NUM_OF_PROCS]; // id2index for each processor, initialized in distribute_partitions
	int * index2id[NUM_OF_PROCS]; // index2id for each processor, initialized in distribute_partitions, used in communication across compute nodes
	voff_t * index_offset[NUM_OF_PROCS]; // index_offset to store the start position of linear vertices in each partition in a processor
	voff_t * jindex_offset[NUM_OF_PROCS]; // index_offset to store the start position of junctions in each partition in a processor
	voff_t * jnb_index_offset[NUM_OF_PROCS]; // index_offset to store the start positions of neighbors of junctions in each partition in a processor
	int * partition_list; // partition list, from device 0, ..., n, cpu 0, ..., m in a computer node; initialized in init_distribute_partitions
	int * r2s; // from the receive buffer to the send buffer; for all to all communication, initialized in distribute_partitions
	vid_t * id_offsets;
	vid_t * jid_offset;
	int * pfrom[NUM_OF_PROCS]; // for parallel memory copy
	int num_partitions[NUM_OF_PROCS + 1]; // prefix sum of the numbers of partitions in each processor
	int total_num_partitions; // total number of partitions in all processes
	int num_of_devices;
	int num_of_cpus;
	uint mssg_size; // unit message size
	int world_size; // number of processes
	int world_rank; // rank id of current process
	ull * gpu_not_found[NUM_OF_DEVICES];
	char * file_dir; // file directory to read the subgraphs
//	int flag[NUM_OF_PROCS];
} master_t; // master of communication of processors, managed in cpu

typedef struct pair
{
	ull kmer;
	vid_t vid;
} pair_t;

typedef struct kmer_vid
{
	kmer_t kmer;
	vid_t vid;
} kmer_vid_t;

typedef struct dbtable
{
	vid_t size;
	vid_t num_elems; // number of elems hashed, to get a prime number
	entry_t * buf; // hash table entries
} dbtable_t;

typedef struct dbmeta
{
	comm_t comm;
	vertices_t nds;
	vertex_t * vs;
	entry_t * ens;
//	ull * before_sort;
//	ull * sorted_kmers;
	kmer_t * before_sort;
	kmer_t * sorted_kmers;
	vid_t * before_vids;
	vid_t * sorted_vids;
	vid_t * nb[EDGE_DIC_SIZE]; // neighbor arrays for junctions
	voff_t * jvld; // valid junction flag array;
	voff_t * lvld; // valid linear vertex flag array;
	vid_t * id_offsets; // vertex id offsets for partitions, copied from master; it points to the memory on device to store id_offsets
	vid_t * jid_offset; // junction size for partitions, copied from master; it points to the memory on device to store jid_offset
	d_jvs_t djs; // structure array of junction vertices, for data transfers
	d_lvs_t dls; // structure array of linear vertices, for data transfers
} dbmeta_t; // meta data for data structures in each processor

typedef struct shakehands {
	kmer_t dst; // destination kmer
	int code; // encoded two edges and their directions
//	msp_id_t pid;
} shakehands_t; // two connected vertices shake hands via bidirected edges

typedef struct assid {
	kmer_t dst; // destination kmers
	int code; // encoded two edges and their directions
	vid_t srcid; // source vertex id
} assid_t; // assign neighors' ids

typedef struct pre {
	int did;
	master_t * mst;
	dbmeta_t * dbm;
	d_jvs_t * js; // all the junctions in a computer node
	d_lvs_t * ls; // all the linear vertices in a computer node
	subgraph_t * subgraph; // record the subgraph sizes and ids in a computer node
	int k;
	int p;
} pre_arg;

typedef struct lr
{
	int did;
	comm_t * cm;
	master_t * mst;
	meta_t * dm;
	int k; // kmer length
} lr_arg;

typedef struct comm_arg
{
	master_t * mst;
	comm_t * cm[NUM_OF_PROCS];
	voff_t num_mssgs;
} comm_arg;


#ifdef __cplusplus
extern "C"
{
void generate_id_offsets (int num_of_partitions);
void free_partition_list (master_t * mst);
void init_graph_data_cpu (int did, master_t * mst, meta_t * dm, d_lvs_t * lvs);
void finalize_graph_data_cpu (void);
void * listrank_push_cpu (void * arg);
void * listrank_pull_cpu (void * arg);
void * listrank_pull_modifygraph_push_cpu (void * arg);
void * modifygraph_pull_cpu (void * arg);
void free_adj_linear (int num_of_partitions, d_lvs_t * dls);
void * listrank_inter_pull_cpu (void * arg);
void * listrank_push_intra_pull_cpu (void * arg);
void * listrank_push_modifygraph_intra_push_cpu (void * arg);
void * listrank_pull_modifygraph_inter_push_intra_pull_cpu (void * arg);
void * modifygraph_inter_pull_cpu (void * arg);
void compact_contig_workflow (int num_of_partitions, d_jvs_t * djs, d_lvs_t * dls, meta_t * dm, master_t * mst, int world_size, int world_rank);
void compact_dbgraph_contig (int num_of_partitions, subgraph_t * subgraph, d_jvs_t * djs, d_lvs_t * dls, meta_t * dm, master_t * mst, int k, int world_size, int world_rank);
}
#endif
#endif /* COMM_H_ */
