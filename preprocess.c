/*
 * preprocess.c
 *
 *  Created on: 2018-3-29
 *      Author: qiushuang
 *
 *  This file preprocesses De Bruijn graph: removing one-directed edge, indexing vertices with their location index,
 *  using new index to replace the neighbors of vertices, splitting and gathering vertices to junctions and linear vertices
 */
#include <omp.h>
#include "include/dbgraph.h"
#include "include/graph.h"
#include "include/comm.h"
#include "include/hash.h"
#include "include/preprocess.h"
#include "include/bitkmer.h"
#include "include/hash.h"
#include "include/malloc.h"
#include "include/share.h"

#define CPU_THREADS 24
#define ASSID_THREADS 24
#define SHAKEHANDS_THREADS 24
extern thread_function shift_dictionary[];
#ifdef LITTLE_ENDIAN
static const ull zerotable[8] = { 0xffffffffffffff00, 0xffffffffffff00ff, 0xffffffffff00ffff, 0xffffffff00ffffff, 0xffffff00ffffffff, 0xffff00ffffffffff, 0xff00ffffffffffff, 0xffffffffffffff};
#else
static const ull zerotable[8] = { 0xffffffffffffff, 0xff00ffffffffffff, 0xffff00ffffffffff, 0xffffff00ffffffff, 0xffffffff00ffffff, 0xffffffffff00ffff, 0xffffffffffff00ff, 0xffffffffffffff00};
#endif

extern long cpu_threads;
extern float elem_factor;
extern voff_t max_ss;
extern int cutoff;
ull send_linear = 0;
ull send_junction = 0;
ull receive_linear = 0;
ull receive_junction = 0;
ull reverse_flag = 0;
ull stride_true = 0;

static uint * size_prime_index; // for hash table lookup

static vid_t * id_offsets; // used to assign global id for each node in subgraphs
static vid_t * jid_offset; // junction vertex id offsets, used to calculate id of each vertex from its index

static kmer_t * kmers; // temporary kmers for fast kmer binary search in assigning neighbor ids
static vertex_t * vertices; // vertex structure

static voff_t * jvalid; // freed after filtering
static voff_t * lvalid; // freed after filtering

static int * id2index; // partition id to the index of partition list

static voff_t * send_offsets; // used to locate the write position of messages for each partition in send buffer
static voff_t * receive_offsets;
static voff_t * extra_send_offsets;
static voff_t * send_offsets_th;
static voff_t * tmp_send_offsets_th;
static voff_t * index_offsets;

extern int lock_flag[NUM_OF_PROCS];
extern float push_offset_time[NUM_OF_PROCS];
extern float push_time[NUM_OF_PROCS];
extern float pull_intra_time[NUM_OF_PROCS];
extern float pull_inter_time[NUM_OF_PROCS];

static void * send;
static void * receive;

// ********** the following are used for output junctions and linear vertices
static kmer_t * jkmers; // for output sorted junction kmers
//static kmer_t * lkmers; // for output sorted linear kmers
static edge_type * post_edges; // for output sorted post edges of linear vertices
static edge_type * pre_edges; // for output sorted pre edges of linear vertices
static vid_t * posts; // for output post neighbor of a linear vertex
static vid_t * pres; // for output pre neighbors of linear vertices
static vid_t * adj_nbs[EDGE_DIC_SIZE]; // for output neigbhors of junctions
static ull * junct_edges; // for output edges of junctions

static voff_t not_found[CPU_THREADS];

voff_t gmax_lsize = 0;
voff_t gmax_jsize = 0;

#define get_reverse_edge(edge, kmer) {	\
	edge = (kmer.x >> (KMER_UNIT_BITS - 2)) ^ 0x3; }

static void set_globals_filter_cpu (dbmeta_t * dbm)
{
	vertices = dbm->vs;
	jvalid = dbm->jvld;
	lvalid = dbm->lvld;
	jid_offset = dbm->jid_offset;
	id_offsets = dbm->id_offsets;
	send = dbm->comm.send;
}

static void set_kmer_for_host_pull (dbmeta_t * dbm)
{
	kmers = dbm->nds.kmer;
}

static void set_pull_push_receive (comm_t * cm)
{
	receive = cm->receive;
}

static void set_globals_preprocessing (dbmeta_t * dbm, int num_of_partitions)
{
	send_offsets = dbm->comm.send_offsets;
	receive_offsets = dbm->comm.receive_offsets;
	send = dbm->comm.send;
	id2index = dbm->comm.id2index;
	index_offsets = dbm->comm.index_offsets;

	send_offsets_th = (voff_t*) malloc (sizeof(voff_t) * (num_of_partitions+1) * (CPU_THREADS+1));
	CHECK_PTR_RETURN (send_offsets_th, "malloc local send offsets for multi-threads in push mssg offset error!\n");
	memset (send_offsets_th, 0, sizeof(voff_t) * (num_of_partitions+1) * (CPU_THREADS+1));
	tmp_send_offsets_th = (voff_t*) malloc (sizeof(voff_t) * (num_of_partitions+1) * (CPU_THREADS+1));
	CHECK_PTR_RETURN (send_offsets_th, "malloc tmp send offsets for multi-threads in push mssg offset error!\n");
	memset (tmp_send_offsets_th, 0, sizeof(voff_t) * (num_of_partitions+1) * (CPU_THREADS+1));
}

void init_preprocessing_data_cpu (dbmeta_t * dbm, int num_of_partitions)
{
	set_globals_preprocessing (dbm, num_of_partitions);
}

static void set_globals_gather (dbmeta_t * dbm)
{
	int i;
	for (i=0; i<EDGE_DIC_SIZE; i++)
	{
		adj_nbs[i] = dbm->djs.nbs[i];
	}
	posts = dbm->dls.posts;
	pres = dbm->dls.pres;
	jkmers = dbm->djs.kmers;
//	lkmers = dbm->dls.kmers;
	post_edges = dbm->dls.post_edges;
	pre_edges = dbm->dls.pre_edges;
	junct_edges = dbm->djs.edges;
}

void reset_globals_gather_cpu (dbmeta_t * dbm)
{
	set_globals_gather (dbm);
}

static void release_globals_cpu (void)
{
	free(send_offsets_th);
	free(tmp_send_offsets_th);
}

void finalize_preprocessing_data_cpu (void)
{
	release_globals_cpu();
}

static void init_kmers (uint size, voff_t index_offset)
{
	entry_t * buf = (entry_t *) send;

	int r;
#pragma omp parallel for
	for (r=0; r<size; r++)
	{
		int index = r;
		if (buf[index].occupied)
		{
			lvalid[index+1] = 1;
		}
	}
}

static void gather_vs (uint size, voff_t index_offset, entry_t * send)
{
	entry_t * buf = (entry_t *) send;
	vertex_t * vs = vertices + index_offset;

	int r;
#pragma omp parallel for
	for (r=0; r<size; r++)
	{
		int index = r;
		if (buf[index].occupied)
		{
			vs[lvalid[index]].kmer = buf[index].kmer;
			vs[lvalid[index]].edge = buf[index].edge;
			vs[lvalid[index]].vid = buf[index].occupied;//2
		}
	}
}

static void gather_vs_sorted (uint size, voff_t index_offset, entry_t * send)
{
	entry_t * buf = send;
	vertex_t * vs = vertices + index_offset;

	int r;
#pragma omp parallel for
	for (r=0; r<size; r++)
	{
		int index = r;
		vs[index].kmer = buf[index].kmer;
		vs[index].edge = buf[index].edge;
		vs[index].vid = buf[index].occupied;//2
	}
}

static void set_mssg_offset_buffer (dbmeta_t * dbm)
{
	extra_send_offsets = dbm->comm.extra_send_offsets;
}

void init_binary_data_cpu (int did, master_t * mst, dbmeta_t * dbm, dbtable_t * tbs)
{
	int * num_partitions = mst->num_partitions;
	int * partition_list = mst->partition_list;
	int num_of_partitions = num_partitions[did+1]-num_partitions[did]; // number of partitions in this processor
	int total_num_partitions = mst->total_num_partitions; // total number of partitions in this compute node
	voff_t * index_offset = mst->index_offset[did];

	int world_size = mst->world_size;
	int world_rank = mst->world_rank;
	int np_per_node = (total_num_partitions + world_size - 1)/world_size;
	int start_partition_id = np_per_node*world_rank;

	set_globals_filter_cpu (dbm);
	set_mssg_offset_buffer (dbm);

	int i;
	voff_t offset = 0;
	for (i=0; i<num_of_partitions; i++)
	{
		int poffset = num_partitions[did];
		int pid = partition_list[poffset+i] - start_partition_id; // !!!be careful here, this pid is not the global partition id
		int pindex = mst->id2index[did][pid + start_partition_id];
		if (pindex != i)
		{
			printf ("ERROR IN DISTRIBUTING PARTITIONS!!!!!!!!\n");
			exit(0);
		}
		voff_t size = tbs[pid].size;
		memset (dbm->lvld, 0, sizeof(vid_t) * (size+1));
		memcpy(dbm->comm.send, tbs[pid].buf, sizeof(entry_t) * size);
		init_kmers (size, offset);
//		inclusive_prefix_sum (dbm->lvld, size+1);
		tbb_scan_uint (dbm->lvld, dbm->lvld, size+1);
		offset = dbm->lvld[size];
		gather_vs (size, index_offset[i], (entry_t*)(dbm->comm.send));
		if (offset > max_ss)
		{
			printf ("error!!!!!!\n");
			exit(0);
		}
		tbb_vertex_sort (dbm->vs + index_offset[i], offset);
		index_offset[i+1] = index_offset[i] + offset;
		free (tbs[pid].buf);
	}
}

void init_binary_data_cpu_sorted (int did, master_t * mst, dbmeta_t * dbm, dbtable_t * tbs)
{
	int * num_partitions = mst->num_partitions;
	int * partition_list = mst->partition_list;
	int num_of_partitions = num_partitions[did+1]-num_partitions[did]; // number of partitions in this processor
	int total_num_partitions = mst->total_num_partitions; // total number of partitions in this compute node
	voff_t * index_offset = mst->index_offset[did];

	int world_size = mst->world_size;
	int world_rank = mst->world_rank;
	int np_per_node = (total_num_partitions + world_size - 1)/world_size;
	int start_partition_id = np_per_node*world_rank;

	set_globals_filter_cpu (dbm);
	set_mssg_offset_buffer (dbm);

	int i;
	for (i=0; i<num_of_partitions; i++)
	{
		int poffset = num_partitions[did];
		int pid = partition_list[poffset+i] - start_partition_id; // !!!be careful here, this pid is not the global partition id
		int pindex = mst->id2index[did][pid + start_partition_id];
		if (pindex != i)
		{
			printf ("ERROR IN DISTRIBUTING PARTITIONS!!!!!!!!\n");
			exit(0);
		}
		voff_t size = tbs[pid].size;
		gather_vs_sorted (size, index_offset[i], tbs[pid].buf);
		index_offset[i+1] = index_offset[i] + size;
		free (tbs[pid].buf);
	}
}

static int
binary_search_vertex (vertex_t * vs, kmer_t * akmer, uint size)
{
	int begin = 0;
	int end = size - 1;
	int index = (begin + end) / 2;
	int ret;
	while (begin <= end)
	{
		ret = compare_2kmers_cpu (&vs[index].kmer, akmer);
		if (ret == 0)
			return index;
		else if (ret < 0)
		{
			end = index - 1;
		}
		else if (ret > 0)
		{
			begin = index + 1;
		}
		index = (begin + end) / 2;
	}
//	printf ("!!!!!!!!!!!!! Error occurs here: %u, %u\n", akmer->x, akmer->y);
	return -1;
}

static int lookup_with_hash_cpu (hashval_t hashval, kmer_t * akmer, kmer_t * kmers, hashsize_t size, int pid)
{
	hashval_t index, hash2;
	uint i;

	index = hashtab_mod_cpu (hashval, size_prime_index[pid]);
	if (index >= size)
	{
		  printf ("index %u is larger than size %u\n", index, size);
		  return -1;
	}

	kmer_t * entry = kmers + index;
	if (is_equal_kmer_cpu(entry, akmer))
	{
		return index;
	}

	hash2 = hashtab_mod_m2_cpu (hashval, size_prime_index[pid]);
	for (i=0; i<size; i++)
	{
	     index += hash2;
	     if (index >= size)
	    	 index -= size;
	     entry = kmers + index;
	 	 if (is_equal_kmer_cpu(entry, akmer))
	 	 {
	 		return index;
	 	 }
	}
	return -1;
}

static int lookup_kmer_assign_source_id_binary (assid_t * mssg, vertex_t * vs, uint size, uint * not_found)
{
	edge_type edge;
	int stride;
	edge = (mssg->code >> (8*2)) & 0xff;
	if (edge > 3)
	{
		printf ("Encoded edge error!!!!!!!!!! %u\n", edge);
		exit (0);
	}

	int index = binary_search_vertex (vs, &mssg->dst, size);
	if (mssg->code & REVERSE_FLAG)
	{
		stride = EDGE_DIC_SIZE / 2;
	}
	else
	{
		stride = 0;
	}

	if (index==-1)
	{
		return -1;
	}
	else
	{
		vs[index].nbs[edge + stride] = mssg->srcid;
	}

	return 0;
}

static int lookup_kmer_set_edge_zero_binary (shakehands_t * mssg, vertex_t * vs, uint size, uint * not_found)
{
	edge_type edge;
	int stride;
	edge = (mssg->code >> (8*2)) & 0xff;
	if (edge > 3)
	{
		printf ("Encoded edge error!!!!!!!!!! %u\n", edge);
		exit (0);
	}

	int index = binary_search_vertex (vs, &mssg->dst, size);
	if (mssg->code & REVERSE_FLAG)
	{
		stride = EDGE_DIC_SIZE / 2;
	}
	else
	{
		stride = 0;
	}

	if (index==-1)
	{
		return -1;
	}
	else
	{
		atomic_and (&vs[index].edge, zerotable[edge+stride]);
	}

	return 0;
}
static void init_hashtab_cpu (uint size, voff_t index_offset)
{
	vertex_t * vs = vertices + index_offset;
	entry_t * buf = (entry_t *)send;

	kmer_t * local_kmers = kmers + index_offset;

	int r;

	for (r = 0; r < size; r++)
	{
		int index = r;
		if (buf[index].occupied)
		{
			if (buf[index].occupied != 2)
			{
				printf ("ERROR IN INPUT DATA!!!!!!!!!!!!\n");
				exit (0);
			}
			vs[index].kmer = buf[index].kmer;
			vs[index].edge = buf[index].edge;
			vs[index].vid = buf[index].occupied;

			local_kmers[index] = buf[index].kmer;
		}
	}
}

void init_hashtab_data_cpu (int did, master_t * mst, dbmeta_t * dbm, dbtable_t * tbs)
{
	int * num_partitions = mst->num_partitions;
	int * partition_list = mst->partition_list;
	int num_of_partitions = num_partitions[did+1]-num_partitions[did]; // number of partitions in this processor
	int total_num_partitions = mst->total_num_partitions; // total number of partitions in this compute node
	voff_t * index_offset = mst->index_offset[did];

	int world_size = mst->world_size;
	int world_rank = mst->world_rank;
	int np_per_node = (total_num_partitions + world_size - 1)/world_size;
	int start_partition_id = np_per_node*world_rank;

	size_prime_index = (uint *) malloc (sizeof(uint) * num_of_partitions);
	set_globals_filter_cpu (dbm);

	set_kmer_for_host_pull (dbm);

	int i;
	voff_t offset = 0;
	for (i=0; i<num_of_partitions; i++)
	{
		int poffset = num_partitions[did];
		int pid = partition_list[poffset+i] - start_partition_id;
		int pindex = mst->id2index[did][pid+start_partition_id];
		if (pindex != i)
		{
			printf ("ERROR IN DISTRIBUTING PARTITIONS!!!!!!!!\n");
			//  exit(0);
		}
		voff_t size = tbs[pid].size;
//		printf ("partition size of hash table %d: %u\n", pid+start_partition_id, size);
		memcpy(dbm->comm.send, tbs[pid].buf, sizeof(entry_t) * size);
//		init_hashtab (size, offset);
		init_hashtab_cpu (size, offset);
		index_offset[i] = offset;
		offset += size;

		uint num_of_elems = tbs[pid].num_elems;
		size_prime_index[i] = higher_prime_index (num_of_elems * elem_factor);
		free (tbs[pid].buf);
	}
	index_offset[i] = offset;
}

void finalize_hashtab_data_cpu (void)
{
	free (size_prime_index);
}

static int lookup_kmer_assign_source_id_cpu2 (assid_t mssg, vertex_t * vs, kmer_t * kmers, uint size, int pindex, uint * not_found)
{
	int edge;
	int stride;
	edge = (mssg.code >> (8*2)) && 0xff;
	if (edge > 3)
	{
		printf ("Encoded edge error!!!!!!!!!! %u\n", edge);
		exit (0);
	}
	hashval_t seed = DEFAULT_SEED;

	hashval_t hash = murmur_hash3_32 ((uint *)&mssg.dst, seed);
	int index = lookup_with_hash_cpu (hash, &mssg.dst, kmers, size, pindex);
	if (mssg.code >> (8*3))
	{
		stride = EDGE_DIC_SIZE / 2;
	}
	else
	{
		stride = 0;
	}
	if (index==-1)
	{
		(*not_found)++;
		return -1;
	}
	else
	{
		vs[index].nbs[edge + stride] = mssg.srcid;
	}

	return 0;
}

static int get_adj_id_from_post (kmer_t * kmer, edge_type edge, int k, int p, int num_of_partitions)
{
	minstr_t minpstr = 0, rminpstr = 0;
	minstr_t curr = 0;
	minstr_t pstr = 0, rpstr = 0;

	unit_kmer_t * ptr;
	kmer_t node[2];

	node[0].x = kmer->x;
	node[0].y = kmer->y;
#ifdef LONG_KMER
	node[0].z = kmer->z;
	node[0].w = kmer->w;
#endif

	kmer_32bit_left_shift (&node[0], 2);
	ptr = (unit_kmer_t *)(&node[0]) + (k * 2) / KMER_UNIT_BITS;
	*ptr |= ((unit_kmer_t)edge) << (KMER_UNIT_BITS - (k * 2) % KMER_UNIT_BITS);
	get_reverse_kmer (&node[0], &node[1], k);

	/* get first minimum p-substring */
	get_first_pstr ((unit_kmer_t *)&node[0], &pstr, p);
	get_first_pstr ((unit_kmer_t *)&node[1], &rpstr, p);
	minpstr = pstr;
	rminpstr = rpstr;

	int j;
	for (j = 1; j < k - p + 1; j++)
	{
		right_shift_pstr ((unit_kmer_t *)&node[0], &pstr, p, j);
		right_shift_pstr ((unit_kmer_t *)&node[1], &rpstr, p, j);
		if (pstr < minpstr) minpstr = pstr;
		if (rpstr < rminpstr) rminpstr = rpstr;
	}
	curr = rminpstr < minpstr ? rminpstr : minpstr;
	msp_id_t mspid = get_partition_id_cpu (curr, p, num_of_partitions);

	return mspid;
}

static int get_adj_id_from_pre (kmer_t * kmer, edge_type edge, int k, int p, int num_of_partitions)
{
	minstr_t minpstr = 0, rminpstr = 0;
	minstr_t curr = 0;
	minstr_t pstr = 0, rpstr = 0;

	unit_kmer_t * ptr;
	kmer_t node[2];

	node[0].x = kmer->x;
	node[0].y = kmer->y;
#ifdef LONG_KMER
	node[0].z = kmer->z;
	node[0].w = kmer->w;
#endif

	get_reverse_kmer (&node[0], &node[1], k);

	kmer_32bit_left_shift (&node[1], 2);
	ptr = (unit_kmer_t *)(&node[1]) + (k*2)/KMER_UNIT_BITS;
	*ptr |= ((unit_kmer_t)edge) << (KMER_UNIT_BITS - (k*2)%KMER_UNIT_BITS);
	get_reverse_kmer (&node[1], &node[0], k);

	/* get first minimum p-substring */
	get_first_pstr ((unit_kmer_t *)&node[0], &pstr, p);
	get_first_pstr ((unit_kmer_t *)&node[1], &rpstr, p);
	minpstr = pstr;
	rminpstr = rpstr;

	int j;
	for (j = 1; j < k - p + 1; j++)
	{
		right_shift_pstr ((unit_kmer_t *)&node[0], &pstr, p, j);
		right_shift_pstr ((unit_kmer_t *)&node[1], &rpstr, p, j);
		if (pstr < minpstr) minpstr = pstr;
		if (rpstr < rminpstr) rminpstr = rpstr;
	}
	curr = rminpstr < minpstr ? rminpstr : minpstr;
	msp_id_t mspid = get_partition_id_cpu (curr, p, num_of_partitions);

	return mspid;
}

static void get_adj_mssg_from_post (kmer_t * kmer, edge_type edge, int k, assid_t * mssg, msp_id_t pid)
{
	uint seed = DEFAULT_SEED;
	unit_kmer_t * ptr;
	kmer_t node[2];
	edge_type edges[2];
	edges[0] = edge; // edges[0] from node[0]

	node[0].x = kmer->x;
	node[0].y = kmer->y;
#ifdef LONG_KMER
	node[0].z = kmer->z;
	node[0].w = kmer->w;
#endif

	get_reverse_edge (edges[1], node[0]); // edges[1] from node[1]
	kmer_32bit_left_shift (&node[0], 2);
	ptr = (unit_kmer_t *)(&node[0]) + (k * 2) / KMER_UNIT_BITS;
	*ptr |= ((unit_kmer_t)edge) << (KMER_UNIT_BITS - (k * 2) % KMER_UNIT_BITS);

	get_reverse_kmer (&node[0], &node[1], k);

	int flag;
	hashval_t hash[2];
	hash[0] = murmur_hash3_32 ((uint *)&node[0], seed);
	hash[1] = murmur_hash3_32 ((uint *)&node[1], seed);
	if (hash[0] == hash[1])
	{
		int ret = compare_2kmers_cpu (&node[0], &node[1]);
		if (ret >= 0)
			flag = 0;
		else flag = 1;
	}
	else if (hash[0] < hash[1]) flag = 0;
	else flag = 1;

	mssg->dst = node[flag];
	mssg->code = 0;
	if (flag == 0)
	{
		mssg->code = (((uint)edges[1]) << (8*2)) | REVERSE_FLAG | pid;
		reverse_flag++;
	}
	else
		mssg->code = (((uint)edges[1]) << (8*2)) | pid;
}

static void get_adj_mssg_from_pre (kmer_t * kmer, edge_type edge, int k, assid_t * mssg, msp_id_t pid)
{
	uint seed = DEFAULT_SEED;
	unit_kmer_t * ptr;
	kmer_t node[2];
	edge_type edges[2];
	edges[1] = edge; // edges[1] from kmer - node[1]

	node[0].x = kmer->x;
	node[0].y = kmer->y;
#ifdef LONG_KMER
	node[0].z = kmer->z;
	node[0].w = kmer->w;
#endif

	get_reverse_kmer (&node[0], &node[1], k);

	get_reverse_edge (edges[0], node[1]); // edges[0] from kmer node[0]
	kmer_32bit_left_shift (&node[1], 2);
	ptr = (unit_kmer_t *)(&node[1]) + (k*2)/KMER_UNIT_BITS;
	*ptr |= ((unit_kmer_t)edge) << (KMER_UNIT_BITS - (k*2)%KMER_UNIT_BITS);

	get_reverse_kmer (&node[1], &node[0], k);

	int flag;
	hashval_t hash[2];
	hash[0] = murmur_hash3_32 ((uint *)&node[0], seed);
	hash[1] = murmur_hash3_32 ((uint *)&node[1], seed);
	if (hash[0] == hash[1])
	{
		int ret = compare_2kmers_cpu (&node[0], &node[1]);
		if (ret >= 0)
			flag = 0;
		else flag = 1;
	}
	else if (hash[0] < hash[1]) flag = 0;
	else flag = 1;

	mssg->dst = node[flag];
	mssg->code = 0;
	if (flag==1)
	{
		mssg->code = (((uint)edges[0]) << (8*2)) | REVERSE_FLAG | pid;
		reverse_flag++;
	}
	else
		mssg->code = (((uint)edges[0]) << (8*2)) | pid;
}

static void push_mssg_offset_shakehands_cpu (voff_t size, int num_of_partitions, voff_t index_offset, int k, int p)
{
	int nump = omp_get_num_procs();
#pragma omp parallel num_threads(cpu_threads)
	{
		int nth = omp_get_num_threads();
		if (nth != cpu_threads)
		{
			printf ("warning: setting cpu threads failed! real number of threads: %d\n", nth);
			// exit(0);
		}
		voff_t size_per_th = (size + nth - 1)/nth;
		if (size_per_th <= nth)
			size_per_th = size/nth;
		voff_t size_th;
		int thid = omp_get_thread_num();
		if (thid == nth-1)
		{
			size_th = size - size_per_th * (nth - 1);
		}
		else
			size_th = size_per_th;

		vertex_t * local_vs = vertices + index_offset + thid * size_per_th;
		voff_t * local_send_offsets = send_offsets_th + (thid+1) * (num_of_partitions+1);

		uint r;
		for(r = 0; r < size_th; r++)
		{
			int index = r;
			int pindex;
			int i;
			for (i = 0; i < EDGE_DIC_SIZE / 2; i++)
			{
				if ((local_vs[index].edge >> (i*8) & 0xff) >= cutoff)
				{
					local_vs[index].nbs[i] = get_adj_id_from_post (&local_vs[index].kmer, (edge_type) i, k, p, num_of_partitions);
					if (local_vs[index].nbs[i] < 0 || local_vs[index].nbs[i] >= num_of_partitions)
					{
						printf ("ERROR IN GETTING MSP ID!!!!!!\n");
					}
					pindex = id2index[local_vs[index].nbs[i]];
					local_send_offsets[pindex+1]++;
				}
				else
					local_vs[index].nbs[i] = DEADEND;
				if ((local_vs[index].edge >> ((EDGE_DIC_SIZE / 2 + i) * 8) & 0xff) >= cutoff)
				{
					local_vs[index].nbs[EDGE_DIC_SIZE / 2 + i] = get_adj_id_from_pre (&local_vs[index].kmer, (edge_type) i, k, p, num_of_partitions);
					if (local_vs[index].nbs[EDGE_DIC_SIZE / 2 + i]  < 0 || local_vs[index].nbs[EDGE_DIC_SIZE / 2 + i] >= num_of_partitions)
					{
						printf ("ERROR IN GETTING MSP ID!!!!!!\n");
					}
					pindex = id2index[local_vs[index].nbs[EDGE_DIC_SIZE / 2 + i]];
					local_send_offsets[pindex+1]++;
				}
				else
					local_vs[index].nbs[EDGE_DIC_SIZE / 2 + i] = DEADEND; // dead end branch
			}
		}
	}
}

static void push_mssg_shakehands_cpu (voff_t size, int num_of_partitions, voff_t index_offset, int k, int p, int curr_id)
{
	omp_set_num_threads(cpu_threads);
#pragma omp parallel
	{
		int nth = omp_get_num_threads();
//		printf ("PUSH MSSG OFFSET THREADS: %d!!!!!!!!!\n", nth);
		if (nth != cpu_threads)
		{
			printf ("warning: setting cpu threads failed! real number of threads: %d\n", nth);
			// exit(0);
		}
		uint size_per_th = (size + nth - 1)/nth;
		if (size_per_th <= nth)
			size_per_th = size/nth;
		uint size_th;
		int thid = omp_get_thread_num();
		if (thid == nth-1)
		{
			size_th = size - size_per_th * (nth - 1);
		}
		else
			size_th = size_per_th;

		vertex_t * local_vs = vertices + index_offset + thid * size_per_th;
		voff_t * local_send_offsets = tmp_send_offsets_th + (thid+1) * (num_of_partitions+1);
		shakehands_t * buf = (shakehands_t *) send;

		uint r;
		for(r = 0; r < size_th; r++)
		{
			int index = r;
			int pindex;
			msp_id_t pid;
			voff_t local_offset;
			voff_t off;
			shakehands_t tmp;
			int i;
			for (i=0; i<EDGE_DIC_SIZE/2; i++)
			{
				if ((local_vs[index].edge >> (i*8) & 0xff) >= cutoff)
				{
					pid = local_vs[index].nbs[i];
					if (pid < 0 || pid >= num_of_partitions)
						printf("ERRORRRRRRRRRRRR\n");
					pindex = id2index[pid];
//					local_offset = local_send_offsets[pindex+1]++;
					off = local_send_offsets[pindex+1] + send_offsets_th[thid*(num_of_partitions+1)+(pindex+1)] + send_offsets[pindex];
					local_send_offsets[pindex+1]++;
					get_adj_mssg_from_post (&local_vs[index].kmer, (edge_type)i, k, (assid_t*)(&tmp), curr_id);
					buf[off] = tmp;
				}
				if ((local_vs[index].edge >> ((EDGE_DIC_SIZE / 2 + i) * 8) & 0xff) >= cutoff)
				{
					pid = local_vs[index].nbs[EDGE_DIC_SIZE / 2 + i];
					if (pid < 0 || pid >= num_of_partitions)
						printf("ERRORRRRRRRRRRRR\n");
					pindex = id2index[pid];
//					local_offset = local_send_offsets[pindex+1]++;
					off = local_send_offsets[pindex+1] + send_offsets_th[thid*(num_of_partitions+1)+(pindex+1)] + send_offsets[pindex];
					local_send_offsets[pindex+1]++;
					get_adj_mssg_from_pre (&local_vs[index].kmer, (edge_type)i, k, (assid_t*)(&tmp), curr_id);
					buf[off] = tmp;
				}
			}
		}

	}
}

static void push_mssg_offset_respond_cpu (voff_t num_mssgs, int pid, voff_t psize, voff_t index_offset, int num_of_partitions, voff_t receive_start, bool intra_inter, int k, int p)
{
	omp_set_num_threads(cpu_threads);
#pragma omp parallel
	{
		int nth = omp_get_num_threads();
		if (nth != cpu_threads)
		{
			printf ("warning: setting cpu thread number failed!\n");
//			exit(0);
		}
		uint size_per_th = (num_mssgs + nth - 1)/nth;
		if (size_per_th <= nth)
			size_per_th = num_mssgs/nth;
		uint size_th;
		int thid = omp_get_thread_num();
		if (thid == nth-1)
		{
			size_th = num_mssgs - size_per_th * (nth - 1);
		}
		else
			size_th = size_per_th;

		shakehands_t * buf;
		voff_t * local_send_offsets = send_offsets_th + (thid+1) * (num_of_partitions+1);

		if (intra_inter)
		{
			int pindex = id2index[pid];
			buf = (shakehands_t *)send + receive_start + send_offsets[pindex] + thid * size_per_th;
		}
		else
		{
			int pindex = id2index[pid];
			buf = (shakehands_t *)send + receive_start + receive_offsets[pindex] + thid * size_per_th;
		}

		vertex_t * local_vs = vertices + index_offset;

		voff_t r;
		for (r=0; r<size_th; r++)
		{
			int index=r;
			shakehands_t tmp = buf[index];
			int thid = omp_get_thread_num ();
			int ret = binary_search_vertex (local_vs, &buf[index].dst, psize);
			int match = 0;
			if (ret != -1)
			{
				edge_type edge = (tmp.code >> (8*2)) & 0xff;
				int stride;
				if (tmp.code >> (8*3))
				{
					stride = EDGE_DIC_SIZE / 2;
				}
				else
				{
					stride = 0;
				}
				if ((local_vs[ret].edge >> ((stride + edge) * 8) & 0xff) < cutoff)
				{
					match = -1;
				}
			}
			if (ret == -1 || match == -1)
			{
				msp_id_t pid = tmp.code & ID_BITS;
				int pindex = id2index[pid];
				local_send_offsets[pindex+1]++;
			}
		}
	}
}

static void push_mssg_respond_cpu (uint num_mssgs, int pid, voff_t psize, voff_t index_offset, int num_of_partitions, voff_t receive_start, bool intra_inter, int k, int p)
{
	omp_set_num_threads(cpu_threads);
#pragma omp parallel
	{
		int nth = omp_get_num_threads();
		if (nth != cpu_threads)
		{
			printf ("warning: setting cpu thread number failed!\n");
//			exit(0);
		}
		uint size_per_th = (num_mssgs + nth - 1)/nth;
		if (size_per_th <= nth)
			size_per_th = num_mssgs/nth;
		uint size_th;
		int thid = omp_get_thread_num();
		if (thid == nth-1)
		{
			size_th = num_mssgs - size_per_th * (nth - 1);
		}
		else
			size_th = size_per_th;

		shakehands_t * buf;
		shakehands_t * vs = (shakehands_t *)receive;
		voff_t * local_offsets = extra_send_offsets;
		voff_t * local_send_offsets = tmp_send_offsets_th + (thid+1) * (num_of_partitions+1);

		if (intra_inter)
		{
			int pindex = id2index[pid];
			buf = (shakehands_t *)send + receive_start + send_offsets[pindex] + thid * size_per_th;
		}
		else
		{
			int pindex = id2index[pid];
			buf = (shakehands_t *)send + receive_start + receive_offsets[pindex] + thid * size_per_th;
		}
		vertex_t * local_vs = vertices + index_offset;

		voff_t r;
		for (r=0; r<size_th; r++)
		{
			int index=r;
			shakehands_t tmp = buf[index];
			edge_type edge = (tmp.code >> (8*2)) & 0xff;
			int stride;
			if (tmp.code >> (8*3))
			{
				stride = EDGE_DIC_SIZE / 2;
			}
			else
			{
				stride = 0;
			}
			int thid = omp_get_thread_num ();
			int mspid;
			int ret = binary_search_vertex (local_vs, &buf[index].dst, psize);
			int match = 0;
			if (ret != -1)
			{
				if ((local_vs[ret].edge >> ((stride + edge) * 8) & 0xff) < cutoff)
				{
					match = -1;
				}
			}
			if (ret == -1 || match == -1)
			{
				mspid = tmp.code & ID_BITS;
				if (stride == 0)
					get_adj_mssg_from_post (&tmp.dst, edge, k, (assid_t*)(&tmp), mspid);
				else
					get_adj_mssg_from_pre (&tmp.dst, edge, k, (assid_t*)(&tmp), mspid);
				int pindex = id2index[mspid];
				voff_t local_send_offset = local_send_offsets[pindex+1]++;
				off_t off = local_send_offset + send_offsets_th[thid*(num_of_partitions+1)+(pindex+1)] + local_offsets[pindex];
				vs[off] = tmp;
			}
		}
	}
}

static void pull_mssg_respond_cpu (uint num_mssgs, int pid, voff_t psize, voff_t index_offset, void * local_receive, bool intra_inter)
{
	int pindex = id2index[pid];
	shakehands_t * buf;
	if (intra_inter) // true if intra partitions
	{
		buf = (shakehands_t *)local_receive + extra_send_offsets[pindex];
	}
	else
	{
		buf = (shakehands_t *)local_receive + receive_offsets[pindex];
	}
	vertex_t * local_vs = vertices + index_offset;

	int r;
#pragma omp parallel for num_threads(cpu_threads)
	for (r=0; r<num_mssgs; r++)
	{
		int index = r;
		int thid = omp_get_thread_num ();
		if (lookup_kmer_set_edge_zero_binary (&buf[index], local_vs, psize, &not_found[thid]) == -1)
		{
			printf ("RESPOND ERROR WITH SOURCE KMER: %u, %u, pindex=%d, pid=%d\n", buf[index].dst.x, buf[index].dst.y, pindex, pid);
		}
	}

}


static void push_mssg_offset_assign_id_cpu (voff_t size, int num_of_partitions, voff_t index_offset, int k, int p)
{
	omp_set_num_threads(cpu_threads);
#pragma omp parallel
	{
		int nth = omp_get_num_threads();
		if (nth != cpu_threads)
		{
			printf ("warning: setting cpu threads failed! real number of threads: %d\n", nth);
		}
		voff_t size_per_th = (size + nth - 1)/nth;
		if (size_per_th <= nth)
			size_per_th = size/nth;
		voff_t size_th;
		int thid = omp_get_thread_num();
		if (thid == nth-1)
		{
			size_th = size - size_per_th * (nth - 1);
		}
		else
			size_th = size_per_th;

		vertex_t * local_vs = vertices + index_offset + thid * size_per_th;
		voff_t * local_send_offsets = send_offsets_th + (thid+1) * (num_of_partitions+1);

		uint r;
		for(r = 0; r < size_th; r++)
		{
			int index = r;
			int pindex;
			int i;
			for (i = 0; i < EDGE_DIC_SIZE / 2; i++)
			{
				if ((local_vs[index].edge >> (i*8) & 0xff) >= cutoff)
				{
//					local_vs[index].nbs[i] = get_adj_id_from_post (&local_vs[index].kmer, (edge_type) i, k, p, num_of_partitions);
					if (local_vs[index].nbs[i] > num_of_partitions)
					{
						printf ("ERROR IN GETTING MSP ID!!!!!!\n");
					}
					pindex = id2index[local_vs[index].nbs[i]];
					local_send_offsets[pindex+1]++;
				}
				else
					local_vs[index].nbs[i] = DEADEND;
				if ((local_vs[index].edge >> ((EDGE_DIC_SIZE / 2 + i) * 8) & 0xff) >= cutoff)
				{
	//				local_vs[index].nbs[EDGE_DIC_SIZE / 2 + i] = get_adj_id_from_pre (&local_vs[index].kmer, (edge_type) i, k, p, num_of_partitions);
					if (local_vs[index].nbs[EDGE_DIC_SIZE / 2 + i] > num_of_partitions)
					{
						printf ("ERROR IN GETTING MSP ID!!!!!!\n");
					}
					pindex = id2index[local_vs[index].nbs[EDGE_DIC_SIZE / 2 + i]];
					local_send_offsets[pindex+1]++;
				}
				else
					local_vs[index].nbs[EDGE_DIC_SIZE / 2 + i] = DEADEND; // dead end branch
			}
		}
	}
}

static void push_mssg_assign_id_cpu (uint size, int num_of_partitions, voff_t index_offset, int k, int p, int curr_id)
{
	omp_set_num_threads(cpu_threads);
#pragma omp parallel
	{
		int nth = omp_get_num_threads();
		if (nth != cpu_threads)
		{
			printf ("warning: setting cpu threads failed! real number of threads: %d\n", nth);
		}
		uint size_per_th = (size + nth - 1)/nth;
		if (size_per_th <= nth)
			size_per_th = size/nth;
		uint size_th;
		int thid = omp_get_thread_num();
		if (thid == nth-1)
		{
			size_th = size - size_per_th * (nth - 1);
		}
		else
			size_th = size_per_th;

		vertex_t * local_vs = vertices + index_offset + thid * size_per_th;
		voff_t * local_send_offsets = tmp_send_offsets_th + (thid+1) * (num_of_partitions+1);
		assid_t * buf = (assid_t *) send;

		uint r;
		for(r = 0; r < size_th; r++)
		{
			int index = r;
			int pindex;
			msp_id_t pid;
			voff_t local_offset;
			voff_t off;
			assid_t tmp;
			kmer_t reverse;
			int i;
			for (i=0; i<EDGE_DIC_SIZE/2; i++)
			{
				if ((local_vs[index].edge >> (i*8) & 0xff) >= cutoff)
				{
					pid = local_vs[index].nbs[i];
					if (pid < 0 || pid > num_of_partitions)
						printf("ERRORRRRRRRRRRRR\n");
					pindex = id2index[pid];
//					local_offset = local_send_offsets[pindex+1]++;
					off = local_send_offsets[pindex+1] + send_offsets_th[thid*(num_of_partitions+1)+(pindex+1)] + send_offsets[pindex];
					local_send_offsets[pindex+1]++;
					get_adj_mssg_from_post (&local_vs[index].kmer, (edge_type)i, k, &tmp, pid);
					tmp.srcid = local_vs[index].vid - 1; // **************************** BE CAREFUL HERE !!!!!!!!!!! ***********************
					if (tmp.srcid >= id_offsets[curr_id] && tmp.srcid < id_offsets[curr_id] + jid_offset[curr_id]) // junction
						send_junction++;
					else
						send_linear++;
					buf[off] = tmp;
				}
				if ((local_vs[index].edge >> ((EDGE_DIC_SIZE / 2 + i) * 8) & 0xff) >= cutoff)
				{
					pid = local_vs[index].nbs[EDGE_DIC_SIZE / 2 + i];
					if (pid < 0 || pid > num_of_partitions)
						printf("ERRORRRRRRRRRRRR\n");
					pindex = id2index[pid];
//					local_offset = local_send_offsets[pindex+1]++;
					off = local_send_offsets[pindex+1] + send_offsets_th[thid*(num_of_partitions+1)+(pindex+1)] + send_offsets[pindex];
					local_send_offsets[pindex+1]++;
					get_adj_mssg_from_pre (&local_vs[index].kmer, (edge_type)i, k, &tmp, pid);
					tmp.srcid = local_vs[index].vid - 1; // **************************** BE CAREFUL HERE !!!!!!!!!!! ***********************
					if (tmp.srcid >= id_offsets[curr_id] && tmp.srcid < id_offsets[curr_id] + jid_offset[curr_id]) // junction
						send_junction++;
					else
						send_linear++;
					buf[off] = tmp;
				}
			}
		}

	}
}

static void pull_mssg_assign_id_cpu (uint num_mssgs, int pid, uint psize, voff_t index_offset, int num_of_partitions, voff_t receive_start, bool intra_inter)
{
	assid_t * buf;
	int pindex = id2index[pid];
	if (intra_inter)
		buf = (assid_t *)send + receive_start + send_offsets[pindex];
	else buf = (assid_t *)send + receive_start + receive_offsets[pindex];

	vertex_t * local_vs = vertices + index_offset;

//	kmer_t * local_kmer = kmers + index_offset;

	int r;
#pragma omp parallel for num_threads(cpu_threads)
	for (r=0; r<num_mssgs; r++)
	{
		int index = r;
		assid_t tmp = buf[index];
		msp_id_t id = tmp.code & ID_BITS;
		// CHECK
//		if (id != pid)
//			printf ("ERROR IN ID ENCODING! id = %d, pid = %d\n", id, pid);
		int thid = omp_get_thread_num ();
//		lookup_kmer_assign_source_id_cpu (tmp, local_vs, psize, pindex, &not_found[thid]); // assign the neigbhor id for kmer tmp.dst
//		if (lookup_kmer_assign_source_id_cpu2 (tmp, local_vs, local_kmer, psize, pindex, &not_found[thid]) == -1)
		if (lookup_kmer_assign_source_id_binary (&buf[index], local_vs, psize, &not_found[thid]) == -1)
		{
			printf ("CPU KMER NOT FOUND: %u, %u, pindex=%d, pid=%d\n", tmp.dst.x, tmp.dst.y, pindex, pid);
//			exit(0);
		}
	}
}

void * neighbor_push_intra_pull_cpu (void * arg)
{
	evaltime_t start, end;
	pre_arg * carg = (pre_arg *) arg;
	comm_t * cm = &carg->dbm->comm;
	master_t * mst = carg->mst;
	int did = carg->did;
	int k = carg->k;
	int p = carg->p;
//	int total_num_partitions = mst->num_partitions[num_of_cpus+num_of_devices];
	int total_num_partitions = mst->total_num_partitions;
	int num_of_partitions = mst->num_partitions[did + 1] - mst->num_partitions[did];
	int poffset = mst->num_partitions[did];
	voff_t * index_offset = mst->index_offset[did];
	if (mst->world_rank == 0)
		printf ("WORLD RANK %d: Neighboring push intra pull cpu:\n", mst->world_rank);

	memset(cm->send_offsets, 0, sizeof(voff_t) * (total_num_partitions + 1));
	memset (send_offsets_th, 0, sizeof(voff_t) * (total_num_partitions+1) * (cpu_threads+1));

#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	int i;
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset+i];
		voff_t size = index_offset[i+1] - index_offset[i];
		push_mssg_offset_assign_id_cpu (size, total_num_partitions, index_offset[i], k, p); // a better version than above
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	push_offset_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "***************** PUSH MSSG OFFSET FOR CPU *ASSIGNING ID* INTRA PROCESSOR TIME: ");
#endif

	get_global_offsets (cm->send_offsets, send_offsets_th, total_num_partitions, (cpu_threads+1));
	inclusive_prefix_sum (cm->send_offsets, total_num_partitions + 1);
	memset (tmp_send_offsets_th, 0, sizeof(voff_t) * (total_num_partitions+1) * (cpu_threads+1));

#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset+i];
		voff_t size = index_offset[i+1] - index_offset[i];
		push_mssg_assign_id_cpu (size, total_num_partitions, index_offset[i], k, p, i); // a better version than above
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	push_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "***************** PUSH MSSG FOR CPU *ASSIGNING ID* INTRA PROCESSOR TIME: ");
#endif

	memcpy(mst->roff[did], cm->send_offsets, sizeof(voff_t)*(total_num_partitions + 1));
	voff_t inter_start = mst->roff[did][num_of_partitions];
	voff_t inter_end = mst->roff[did][total_num_partitions];

	mst->receive[did] = (assid_t *)cm->send+inter_start;
//	pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
//	pthread_mutex_lock(&mutex);
//	lock_flag[did] = 1;
//	pthread_mutex_unlock(&mutex);
//	if (atomic_set_value (&lock_flag[did], 1, 0) == false)
//		printf("!!!!!!!!!!!!!!!!!! CAREFUL, SETTING VALUE DOES NOT WORK FINE!\n");

	uint total_not_found = 0;
	for (i=0; i<cpu_threads; i++)
	{
		total_not_found += not_found[i];
	}
//	printf ("~~~~~~~~~~~~~~~~~~~~~ Number of vertices not found: %u, number of mssgs: %u\n", total_not_found, cm->send_offsets[num_of_partitions]);

	memset (not_found, 0, sizeof(uint) * cpu_threads);

#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset + i];
		voff_t num_mssgs = cm->send_offsets[i+1] - cm->send_offsets[i];
		voff_t size = index_offset[i+1] - index_offset[i];
		pull_mssg_assign_id_cpu (num_mssgs, pid, size, index_offset[i], total_num_partitions, 0, 1); // a better version
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	pull_intra_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "***************** PULL MSSG FOR CPU *ASSIGNING ID* INTRA PROCESSOR TIME: ");
#endif
	for (i=0; i<cpu_threads; i++)
	{
		total_not_found += not_found[i];
	}
//	printf ("~~~~~~~~~~~~~~~~~~~~~ Number of vertices not found: %u, number of mssgs: %u\n", total_not_found, cm->send_offsets[num_of_partitions]);

	return ((void*) 0);
}

void * neighbor_inter_pull_cpu (void * arg)
{
	evaltime_t start, end;
	pre_arg * carg = (pre_arg *) arg;
	comm_t * cm = &carg->dbm->comm;
	master_t * mst = carg->mst;
	int num_of_cpus = mst->num_of_cpus;
	int num_of_devices = mst->num_of_devices;
	int did = carg->did;
	int total_num_partitions = mst->total_num_partitions;
	int num_of_partitions = mst->num_partitions[did + 1] - mst->num_partitions[did];
	voff_t * index_offset = mst->index_offset[did];
	if (mst->world_rank == 0)
		printf ("WORLD RANK %d: Neighboring inter pull cpu:\n", mst->world_rank);

	voff_t receive_start = mst->roff[did][num_of_partitions];
	memcpy(cm->receive_offsets, mst->soff[did], sizeof(voff_t) * (num_of_partitions + 1));
	voff_t inter_size = mst->soff[did][num_of_partitions];
	if(inter_size == 0)
		return ((void*) 0);
	memcpy((assid_t*)cm->send + receive_start, mst->send[did], sizeof(assid_t) * inter_size);

	int poffset = mst->num_partitions[did];
	int i;
#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset + i];
		voff_t num_mssgs = cm->receive_offsets[i+1] - cm->receive_offsets[i];
		voff_t size = index_offset[i+1] - index_offset[i];
		pull_mssg_assign_id_cpu (num_mssgs, pid, size, index_offset[i], total_num_partitions, receive_start, 0);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	pull_inter_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "***************** PULL MSSG FOR CPU *NEIGHBORING* INTER PROCESSORS TIME: ");
#endif
	uint total_not_found = 0;
	for (i=0; i<cpu_threads; i++)
	{
		total_not_found += not_found[i];
	}
//	printf ("~~~~~~~~~~~~~~~~~~~~~ Number of vertices not found: %u\n", total_not_found);
	return ((void *) 0);
}

void * shakehands_push_respond_intra_push_cpu (void * arg)
{
	evaltime_t start, end;
	pre_arg * carg = (pre_arg *) arg;
	comm_t * cm = &carg->dbm->comm;
	master_t * mst = carg->mst;
	int did = carg->did;
	int k = carg->k;
	int p = carg->p;
	int total_num_partitions = mst->total_num_partitions;
	int num_of_partitions = mst->num_partitions[did + 1] - mst->num_partitions[did];
	int poffset = mst->num_partitions[did];
	voff_t * index_offset = mst->index_offset[did];
	if (mst->world_rank == 0)
		printf ("WORLD RANK %d: shakehands push respond intra push cpu:\n", mst->world_rank);

	memset(cm->send_offsets, 0, sizeof(voff_t) * (total_num_partitions + 1));
	memset (send_offsets_th, 0, sizeof(voff_t) * (total_num_partitions+1) * (cpu_threads+1));

#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	int i;
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset+i];
		voff_t size = index_offset[i+1] - index_offset[i];
		push_mssg_offset_shakehands_cpu (size, total_num_partitions, index_offset[i], k, p); // a better version than above
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	push_offset_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "***************** PUSH MSSG OFFSET FOR CPU *SHAKEHANDS* INTRA PROCESSOR TIME: ");
#endif

	get_global_offsets (cm->send_offsets, send_offsets_th, total_num_partitions, (cpu_threads+1));
	inclusive_prefix_sum (cm->send_offsets, total_num_partitions + 1);
	memset (tmp_send_offsets_th, 0, sizeof(voff_t) * (total_num_partitions+1) * (cpu_threads+1));

#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset+i];
		voff_t size = index_offset[i+1] - index_offset[i];
		push_mssg_shakehands_cpu (size, total_num_partitions, index_offset[i], k, p, pid); // a better version than above
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	push_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "***************** PUSH MSSG FOR CPU *SHAKEHANDS* INTRA PROCESSOR TIME: ");
#endif

	memcpy(mst->roff[did], cm->send_offsets, sizeof(voff_t)*(total_num_partitions + 1));
	voff_t inter_start = mst->roff[did][num_of_partitions];
	voff_t inter_end = mst->roff[did][total_num_partitions];
	mst->receive[did] = (shakehands_t *)cm->send+inter_start;

//	pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
//	pthread_mutex_lock(&mutex);
//	lock_flag[did] = 1;
//	pthread_mutex_unlock(&mutex);
//	if (atomic_set_value (&lock_flag[did], 1, 0) == false)
//		printf("!!!!!!!!!!!!!!!!!! CAREFUL, SETTING VALUE DOES NOT WORK FINE!\n");

	uint total_not_found = 0;
	for (i=0; i<cpu_threads; i++)
	{
		total_not_found += not_found[i];
	}

	memset (cm->extra_send_offsets, 0, sizeof(voff_t) * (total_num_partitions+1));
	memset (send_offsets_th, 0, sizeof(voff_t) * (total_num_partitions+1) * (cpu_threads+1));

	memset (not_found, 0, sizeof(uint) * cpu_threads);

#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset + i];
		voff_t num_mssgs = cm->send_offsets[i+1] - cm->send_offsets[i];
		voff_t size = index_offset[i+1] - index_offset[i];
		push_mssg_offset_respond_cpu (num_mssgs, pid, size, index_offset[i], total_num_partitions, 0, 1, k, p); // a better version
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	pull_intra_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "***************** PUSH *RESPOND* OFFSET CPU INTRA PROCESSOR TIME: ");
#endif
	for (i=0; i<cpu_threads; i++)
	{
		total_not_found += not_found[i];
	}

	return ((void*) 0);
}

void * shakehands_pull_respond_inter_push_intra_pull_cpu (void * arg)
{
	evaltime_t start, end;
	pre_arg * carg = (pre_arg *) arg;
	comm_t * cm = &carg->dbm->comm;
	master_t * mst = carg->mst;
	int num_of_cpus = mst->num_of_cpus;
	int num_of_devices = mst->num_of_devices;
	int did = carg->did;
	int k = carg->k;
	int p = carg->p;
	int total_num_partitions = mst->total_num_partitions;
	int num_of_partitions = mst->num_partitions[did + 1] - mst->num_partitions[did];
	int poffset = mst->num_partitions[did];
	voff_t * index_offset = mst->index_offset[did];
	if (mst->world_rank == 0)
		printf ("WORLD RANK %d: Shakehands pull respond update inter push intra pull cpu:\n", mst->world_rank);

	voff_t receive_start = cm->send_offsets[num_of_partitions];
	memcpy(cm->receive_offsets, mst->soff[did], sizeof(voff_t) * (num_of_partitions + 1));
	voff_t inter_size = cm->receive_offsets[num_of_partitions];
	memcpy((shakehands_t*)(cm->send) + receive_start, mst->send[did], sizeof(shakehands_t) * inter_size);

#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	int i;
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset + i];
		voff_t num_mssgs = cm->receive_offsets[i+1] - cm->receive_offsets[i];
		voff_t size = index_offset[i+1] - index_offset[i];
		push_mssg_offset_respond_cpu (num_mssgs, pid, size, index_offset[i], total_num_partitions, receive_start, 0, k, p);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	pull_inter_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "*********** PUSH *RESPOND* OFFSET CPU INTER PROCESSORS TIME: ");
#endif
	get_global_offsets (cm->extra_send_offsets, send_offsets_th, total_num_partitions, (cpu_threads+1));
	inclusive_prefix_sum (cm->extra_send_offsets, total_num_partitions + 1);

	// *************** malloc (send and) receive buffer for pull and push mode, this is for junctions???????
	voff_t rcv_size = cm->extra_send_offsets[num_of_partitions];
	if (rcv_size == 0)
	{
		printf ("WORLD RANK %d: CPU::: CCCCCCCCCcccareful:::::::::: receive size from intra junction update push is 0!!!!!!!!\n", mst->world_rank);
		rcv_size = 1000;
	}
	malloc_pull_push_receive_cpu (cm, sizeof(shakehands_t), did, rcv_size, 2*(total_num_partitions+num_of_partitions-1)/num_of_partitions);
	set_pull_push_receive (cm);

	memset (tmp_send_offsets_th, 0, sizeof(voff_t) * (total_num_partitions+1) * (cpu_threads+1));

#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset+i];
		voff_t num_mssgs = cm->send_offsets[i+1] - cm->send_offsets[i];
		voff_t size = index_offset[i+1] - index_offset[i];
		push_mssg_respond_cpu (num_mssgs, pid, size, index_offset[i], total_num_partitions, 0, 1, k, p);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	push_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "************** PUSH *RESPOND* MSSG CPU INTRA PROCESSOR TIME: ");
#endif

#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset+i];
		voff_t num_mssgs = cm->receive_offsets[i+1] - cm->receive_offsets[i];
		voff_t size = index_offset[i+1] - index_offset[i];
		push_mssg_respond_cpu (num_mssgs, pid, size, index_offset[i], total_num_partitions, receive_start, 0, k, p);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	push_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "************** PUSH *RESPOND* MSSG CPU INTER PROCESSORS TIME: ");
#endif

	memcpy(mst->roff[did], cm->extra_send_offsets, sizeof(voff_t)*(total_num_partitions + 1));
	voff_t inter_start = cm->extra_send_offsets[num_of_partitions];
	voff_t inter_end = cm->extra_send_offsets[total_num_partitions];


	printf ("@@@@@@@@@@@@@@@@@@@@@ WORLD RANK %d:: total number of shakehands pushed in device %d: %lu\n", mst->world_rank, did, inter_end);
	printf ("############### WORLD RANK %d:: number of intra mssgs pulled for inter shakehands of device %d: %lu\n", mst->world_rank, did, inter_start);
	printf ("############### WORLD RANK %d:: number of slots malloced for receive buffer of device %d: %u\n", mst->world_rank, did, cm->temp_size/sizeof(shakehands_t));
	mst->receive[did] = (shakehands_t*)(cm->receive) + inter_start;

#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset + i];
		voff_t num_mssgs = cm->extra_send_offsets[i+1] - cm->extra_send_offsets[i];
		voff_t size = index_offset[i+1] - index_offset[i];
		pull_mssg_respond_cpu (num_mssgs, pid, size, index_offset[i], cm->receive, 1);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	pull_intra_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "WORLD RANK %d ***************** PULL *RESPOND* MSSG CPU INTRA PROCESSOR TIME: ", mst->world_rank);
#endif

	return ((void *) 0);
}

void * respond_inter_pull_cpu (void * arg)
{
	evaltime_t start, end;
	pre_arg * carg = (pre_arg *) arg;
	comm_t * cm = &carg->dbm->comm;
	master_t * mst = carg->mst;
	int num_of_cpus = mst->num_of_cpus;
	int num_of_devices = mst->num_of_devices;
	int did = carg->did;
	int total_num_partitions = mst->total_num_partitions;
	int num_of_partitions = mst->num_partitions[did + 1] - mst->num_partitions[did];
	voff_t * index_offset = mst->index_offset[did];
	if (mst->world_rank == 0)
		printf ("WORLD RANK %d: respond inter pull cpu:\n", mst->world_rank);

	voff_t receive_start = mst->roff[did][num_of_partitions]; //cm->extra_send_offsets[num_of_partitions];
	memcpy(cm->receive_offsets, mst->soff[did], sizeof(voff_t) * (num_of_partitions + 1));
	voff_t inter_size = mst->soff[did][num_of_partitions];
	if(inter_size == 0)
		return ((void*) 0);
	if (cm->temp_size <= (inter_size+receive_start)*sizeof(shakehands_t))
	{
		printf("WORLD RANK %d: CPU: Error:::::::: malloced receive buffer size smaller than actual receive buffer size!\n", mst->world_rank);
		exit(0);
	}
	memcpy((shakehands_t*)(cm->receive) + receive_start, mst->send[did], sizeof(shakehands_t) * inter_size);
	printf ("############### number of inter mssgs pulled for inter shakehands of device %d: %lu\n", did, inter_size);

	int poffset = mst->num_partitions[did];
	int i;
#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset + i];
		voff_t num_mssgs = cm->receive_offsets[i+1] - cm->receive_offsets[i];
		voff_t size = index_offset[i+1] - index_offset[i];
		pull_mssg_respond_cpu (num_mssgs, pid, size, index_offset[i], cm->receive + sizeof(shakehands_t) * receive_start, 0);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	pull_inter_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "***************** PULL *RESPOND* MSSG FOR CPU INTER PROCESSORS TIME: ");
#endif
	// *************** free (send and) receive buffer for pull and push mode
	free_pull_push_receive_cpu(cm);

	uint total_not_found = 0;
	for (i=0; i<cpu_threads; i++)
	{
		total_not_found += not_found[i];
	}
//	printf ("~~~~~~~~~~~~~~~~~~~~~ Number of vertices not found: %u\n", total_not_found);
	return ((void *) 0);
}

static void label_vertex_with_flags_cpu (uint size, voff_t index_offset)
{
	vertex_t * local_vs = vertices + index_offset;
	voff_t * local_jvalid = jvalid + index_offset;
	voff_t * local_lvalid = lvalid + index_offset;

	uint r;
#pragma omp parallel for num_threads(cpu_threads)
	for (r = 0; r < size; r++)
	{
		int ind = 0;
		int outd = 0;
		uint index = r;
		if (local_vs[index].vid == 0)
			continue;

		int i;
		for (i = 0; i < EDGE_DIC_SIZE / 2; i++)
		{
			if ((local_vs[index].edge >> ((EDGE_DIC_SIZE / 2 + i) * 8) & 0xff) >= cutoff)
			{
				ind++;
			}
			if ((local_vs[index].edge >> (i*8) & 0xff) >= cutoff)
			{
				outd++;
			}
		}

		if (ind==0 && outd==0) // filter isolated vertices
		{
			local_vs[index].vid=0;
			continue;
		}

		if (ind > 1 || outd > 1)
		{
			local_jvalid[index] = 1; // to pick out junction nodes
		}

		if (ind <= 1 && outd <= 1)
		{
			if (outd <= 1)
			{
				for (i = 0; i < EDGE_DIC_SIZE / 2; i++)
				{
					if ((local_vs[index].edge >> (i*8) & 0xff) >= cutoff)
					{
						break;
					}
				}
				if (i == EDGE_DIC_SIZE / 2)
				{
					local_jvalid[index] = 1; // set a vertex with only one edge or no edge to be a junction
				}
			}
			if (ind <= 1)
			{
				for (i = EDGE_DIC_SIZE / 2; i < EDGE_DIC_SIZE; i++)
				{
					if ((local_vs[index].edge >> (i*8) & 0xff) >= cutoff)
					{
						break;
					}
				}
				if (i == EDGE_DIC_SIZE)
				{
					local_jvalid[index] = 1; // set a vertex with only one edge or no edge to be a junction
				}
			}
			if (local_jvalid[index] != 1)
				local_lvalid[index] = 1; // a linear vertex
		}
	}
}

static void assid_vertex_with_flags_cpu (uint size, int pid, voff_t index_offset)
{
	uint r;

	vertex_t * local_vs = vertices + index_offset;
	voff_t * local_jvalid = jvalid + index_offset;
	voff_t * local_lvalid = lvalid + index_offset;

#pragma omp parallel for num_threads(cpu_threads)
	for (r = 0; r < size; r++)
	{
		uint index = r;
		bool jflag, lflag;

		if (index==0)
		{
			if (local_jvalid[index])
				jflag = true;
			else
				jflag = false;
			if (local_lvalid[index])
				lflag = true;
			else
				lflag = false;
		}
		else
		{
			if (local_jvalid[index] - local_jvalid[index-1])
				jflag = true;
			else
				jflag = false;
			if (local_lvalid[index] - local_lvalid[index-1])
				lflag = true;
			else
				lflag = false;
		}

		if (jflag==false && lflag==false) // empty slot
		{
			local_vs[index].vid = 0;
			continue;
		}

		// **************************** BE CAREFUL HERE !!!!!!!!!!! ***********************
		if (jflag==true) // a junction
			local_vs[index].vid = local_jvalid[index] + id_offsets[pid];
		else if (lflag) // a linear vertex
			local_vs[index].vid = jid_offset[pid] + id_offsets[pid] + local_lvalid[index];
	}
}


void * identify_vertices_cpu (void * arg)
{
	evaltime_t start, end;
	pre_arg * garg = (pre_arg *) arg;
	master_t * mst = garg->mst;
	dbmeta_t * dbm = garg->dbm;
	int did = garg->did;
	if (mst->world_rank == 0)
		printf ("WORLD RANK %d: identifying vertices cpu %d:\n", mst->world_rank, did);

	int num_of_partitions = mst->num_partitions[did + 1] - mst->num_partitions[did];
	voff_t * index_offset = mst->index_offset[did];
	memset (dbm->lvld, 0, sizeof(vid_t) * (max_ss + 1));
	int i;
#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	for (i=0; i<num_of_partitions; i++)
	{
		int poffset = mst->num_partitions[did];
		int pid = mst->partition_list[poffset+i];
		voff_t size = index_offset[i+1] - index_offset[i];
		label_vertex_with_flags_cpu (size, index_offset[i]);
//		inclusive_prefix_sum ((int *)(dbm->jvld + index_offset[i]), size);
		tbb_scan_uint (dbm->jvld + index_offset[i], dbm->jvld + index_offset[i], size);
//		inclusive_prefix_sum ((int *)(dbm->lvld + index_offset[i]), size);
		tbb_scan_uint (dbm->lvld + index_offset[i], dbm->lvld + index_offset[i], size);
		mst->jid_offset[pid] = (dbm->jvld + index_offset[i])[size-1];
		mst->id_offsets[pid+1] = (dbm->jvld + index_offset[i])[size-1] + (dbm->lvld + index_offset[i])[size-1]; // pid+1, for prefix-sum later
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& IDENTIFYING IDS OF VERTICES TIME: ");
#endif
//	print_offsets (mst->jid_offset, total_num_partitions);
//	print_offsets (mst->id_offsets, total_num_partitions+1);

	return ((void *)0);
}

void * assign_vertex_ids_cpu (void * arg)
{
	evaltime_t start, end;
	pre_arg * garg = (pre_arg *) arg;
	master_t * mst = garg->mst;
	int did = garg->did;
	if (mst->world_rank == 0)
		printf ("WORLD RANK %d: Assigning vertices cpu %d:\n", mst->world_rank, did);

	int num_of_partitions = mst->num_partitions[did + 1] - mst->num_partitions[did];
	voff_t * index_offset = mst->index_offset[did];

	int i;
#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	for (i=0; i<num_of_partitions; i++)
	{
		int poffset = mst->num_partitions[did];
		int pid = mst->partition_list[poffset+i];
		voff_t size = index_offset[i+1] - index_offset[i];
//		assid_vertex_with_flags (size, pid, index_offset[i]);
		assid_vertex_with_flags_cpu (size, pid, index_offset[i]);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	print_exec_time (start, end, "&&&&&&&&&&&&&&&&&&&& ASSIGNING IDS OF VERTICES TIME: ");
#endif
	return ((void *)0);
}

static void gather_vertex_cpu (uint size, int pid, voff_t index_offset, int total_num_partitions)
{
	vertex_t * local_vs = vertices + index_offset;
	uint count=0;
	uint dead=0;
	uint r;
#pragma omp parallel for num_threads(cpu_threads) firstprivate(pid, size, local_vs)
	for (r = 0; r < size; r++)
	{
		int i;
		uint index = r;
		if (local_vs[index].vid == 0)
			continue;

		voff_t off = local_vs[index].vid - id_offsets[pid] - 1;
		if (local_vs[index].vid < id_offsets[pid] + 1)
		{
			printf ("Gather vertex cpu: error in vertex id and id_offsets: local_vs[%u].vid=%lu, id_offsets[%d]=%lu!\n", \
					index, local_vs[index].vid, pid, id_offsets[pid]);
			exit(0);
		}
		if (local_vs[index].vid - id_offsets[pid] <= jid_offset[pid]) // a junction here
		{
			for (i=0; i<EDGE_DIC_SIZE; i++)
			{
				adj_nbs[i][off] = local_vs[index].nbs[i];
				if (local_vs[index].nbs[i] != DEADEND)
				{
					int ppid = query_partition_id_from_idoffsets (local_vs[index].nbs[i], total_num_partitions, id_offsets);
					if (local_vs[index].nbs[i] - id_offsets[ppid] <= jid_offset[ppid])
						count++;
				}
				else
					dead++;
			}
			jkmers[off] = local_vs[index].kmer;
			junct_edges[off] = local_vs[index].edge;
		}
		else // a linear vertex
		{
			for (i = 0; i < EDGE_DIC_SIZE / 2; i++)
			{
				if ((local_vs[index].edge >> (i*8) & 0xff) >= cutoff)
				{
					posts[off - jid_offset[pid]] = local_vs[index].nbs[i];
					post_edges[off - jid_offset[pid]] = i;
					if (local_vs[index].nbs[i] != DEADEND)
					{
						int ppid = query_partition_id_from_idoffsets (local_vs[index].nbs[i], total_num_partitions, id_offsets);
						if (local_vs[index].nbs[i] - id_offsets[ppid] <= jid_offset[ppid])
							count++;
					}
					else
					{
						dead++;
						printf ("error in gathering neighbors of linear vertices here!\n");
						exit(0);
					}
				}
			}
			for (i = EDGE_DIC_SIZE / 2; i < EDGE_DIC_SIZE; i++)
			{
				if ((local_vs[index].edge >> (i*8) & 0xff) >= cutoff)
				{
					pres[off - jid_offset[pid]] = local_vs[index].nbs[i];
					pre_edges[off - jid_offset[pid]] = i - EDGE_DIC_SIZE/2;
					if (local_vs[index].nbs[i] != DEADEND)
					{
						int ppid = query_partition_id_from_idoffsets (local_vs[index].nbs[i], total_num_partitions, id_offsets);
						if (local_vs[index].nbs[i] - id_offsets[ppid] <= jid_offset[ppid])
							count++;
					}
					else
					{
						dead++;
						printf ("error in gathering neighbors of linear vertices here!\n");
						exit(0);
					}
					break;
				}
			}
//			lkmers[off - jid_offset[pid]] = local_vs[index].kmer;

		}
	}
//	printf ("partition %d: number of junctions in neighbors: %u, deadend = %u, size = %u\n", pid, count, dead, size);
}

void * gather_vertices_cpu (void * arg)
{
	evaltime_t start, end;
	pre_arg * garg = (pre_arg *) arg;
	master_t * mst = garg->mst;
	dbmeta_t * dbm = garg->dbm;
	d_jvs_t * js = garg->js;
	d_lvs_t * ls = garg->ls;
	subgraph_t * subgraph = garg->subgraph;
	int did = garg->did;
	if (mst->world_rank == 0)
		printf ("WORLD RANK %d: Assigning vertices cpu %d:\n", mst->world_rank, did);

	int total_num_partitions = mst->total_num_partitions;
	int num_of_partitions = mst->num_partitions[did + 1] - mst->num_partitions[did];
	voff_t * index_offset = mst->index_offset[did];

	int i;
#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	for (i=0; i<num_of_partitions; i++)
	{
		int poffset = mst->num_partitions[did];
		int pid = mst->partition_list[poffset+i];
		voff_t size = index_offset[i+1] - index_offset[i];
		gather_vertex_cpu (size, pid, index_offset[i], total_num_partitions);
		uint jsize = mst->jid_offset[pid];
		uint lsize = mst->id_offsets[pid+1] - mst->id_offsets[pid] - jsize;
		output_vertices_cpu (dbm, mst, jsize, lsize, pid, total_num_partitions, did, js, ls, subgraph);
		write_kmers_edges_cpu (dbm, mst, jsize, lsize, pid, total_num_partitions, did);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	print_exec_time (start, end, "CPU: &&&&&&&&&&&&&&&&&&&& ASSIGNING IDS OF VERTICES TIME: ");
#endif
	if (mst->world_rank == 0 && mst->num_of_devices == 0)
	{
	printf ("CPU: NUMBER OF VERTICES PROCESSED: %u\n", index_offset[num_of_partitions]);
	printf ("CPU: number of messages sent by junction: %lu\n"
			"number of messages sent by linear vertices: %lu\n"
			"number of messages received from junction: %lu\n"
			"number of messages received from linear vertices: %lu\n",\
			send_junction, send_linear, receive_junction, receive_linear);
	}

	return ((void *)0);
}

