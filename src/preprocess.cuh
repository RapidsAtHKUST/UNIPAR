/*
 * preprocess.cuh
 *
 *  Created on: 2018-3-29
 *      Author: qiushuang
 *
 *  Device kernels and functions for preprocess
 */
#ifndef PREPROCESS_CUH_
#define PREPROCESS_CUH_

#include "../include/hash.h"
#include "../include/preprocess.h"
#include "bitkmer.cuh"
#include "hash.cuh"

#define MAX_NUM_BLOCKS 1024 // for performance, setting it to be 1024; the maximum number of blocks in GPU can be at least 4096
#define THREADS_PER_BLOCK_NODES 1024
#define TOTAL_THREADS_NODES (THREADS_PER_BLOCK_NODES * MAX_NUM_BLOCKS)

#ifdef LITTLE_ENDIAN
__constant__ static ull zerotable[8] = { 0xffffffffffffff00, 0xffffffffffff00ff, 0xffffffffff00ffff, 0xffffffff00ffffff, 0xffffff00ffffffff, 0xffff00ffffffffff, 0xff00ffffffffffff, 0xffffffffffffff};
#else
__constant__ static ull zerotable[8] = { 0xffffffffffffff, 0xff00ffffffffffff, 0xffff00ffffffffff, 0xffffff00ffffffff, 0xffffffff00ffffff, 0xffffffffff00ffff, 0xffffffffffff00ff, 0xffffffffffffff00};
#endif


//********** If use hash table lookup: *********
__constant__ static uint * size_prime_index; // array in size of number of partitions

__constant__ static goffset_t * id_offsets; // used to assign global id for each node in subgraphs
__constant__ static goffset_t * jid_offset; // junction vertex id offsets, used to calculate id of each vertex from its index

__constant__ static entry_t * ens;
//__constant__ static ull * before_sort;
//__constant__ static ull * sorted_kmers;
__constant__ static kmer_t * before_sort;
__constant__ static kmer_t * sorted_kmers; // used with binary search
__constant__ static vid_t * before_vids;
__constant__ static kmer_t * kmers; // used with hashtable lookup
__constant__ static ull * edges;
__constant__ static vid_t * vids; // global id assigned to each vertex: scatter each vertex to an index calculated from this vid
__constant__ static vid_t * nbs[EDGE_DIC_SIZE]; // for hash table and adj arrays
//__constant__ static ull * spids;
//__constant__ static ull * spidsr;

__constant__ static voff_t * jvalid; // flag to denote valid junctions
__constant__ static voff_t * lvalid; // flag to denote valid linear vertices

__constant__ static int * id2index; // partition id to the index of partition list

__constant__ static voff_t * send_offsets; // used to locate the write position of messages for each partition in send buffer
__constant__ static voff_t * receive_offsets;
__constant__ static voff_t * extra_send_offsets; // used when pull message and push new messages happens simultaneously.
__constant__ static voff_t * tmp_send_offsets;
__constant__ static voff_t * index_offsets;

__constant__ static void * send;
__constant__ static void * receive;

// ********** the followings are used for output junctions and linear vertices
__constant__ static kmer_t * jkmers;
//__constant__ static kmer_t * lkmers;
__constant__ static vid_t * posts; // for gather vertices to adj arrays
__constant__ static vid_t * pres; // for gather vertices to adj arrays
__constant__ static edge_type * post_edges; // for output post edges of linear vertices
__constant__ static edge_type * pre_edges; // for output pre edges of linear vertices
__constant__ static ull * junct_edges ; // for output edges of junctions
__constant__ static vid_t * adj_nbs[EDGE_DIC_SIZE]; // for gather neigbhors of junctions
__constant__ static ull * spids; // for output shared pids of neighbors of kmers
__constant__ static ull * spidsr; // for output shared pids of neighbors of reverse complements
__constant__ static uint * spidlv; // for output shared pids of post and pre of linear vertices

__constant__ ull * gpu_not_found;

extern "C"
{
static void set_extra_send_offsets_gpu (voff_t ** extra_send_offsets_ptr, master_t * mst, int i)
{
#ifdef SINGLE_NODE
	int num_of_devices = mst->num_of_devices;
	int world_rank = mst->world_rank;
	CUDA_CHECK_RETURN (cudaSetDevice (world_rank * num_of_devices + i));
#else
	CUDA_CHECK_RETURN (cudaSetDevice (i + DEVICE_SHIFT));
#endif
	CUDA_CHECK_RETURN (cudaMemcpyToSymbol(extra_send_offsets, extra_send_offsets_ptr, sizeof(voff_t *)));
}

static void set_globals_filter1_gpu (dbmeta_t * dbm, master_t * mst)
{
	int num_of_devices = mst->num_of_devices;

	int i;
#ifdef SINGLE_NODE
	int world_rank = mst->world_rank;
#endif
	for (i = 0; i < num_of_devices; i++)
	{
#ifdef SINGLE_NODE
		cudaSetDevice(world_rank * num_of_devices + i);
#else
		cudaSetDevice(i + DEVICE_SHIFT);
#endif
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (sorted_kmers, &dbm[i].sorted_kmers, sizeof(kmer_t*)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (vids, &dbm[i].sorted_vids, sizeof(vid_t*)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (before_sort, &dbm[i].before_sort, sizeof(kmer_t*)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (before_vids, &dbm[i].before_vids, sizeof(vid_t*)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (edges, &dbm[i].nds.edge, sizeof(ull *)));
//		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (spids, &dbm[i].nds.spids, sizeof(ull *)));
//		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (spidsr, &dbm[i].nds.spidsr, sizeof(ull *)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (send, &dbm[i].comm.send, sizeof(void *)));

		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (lvalid, &dbm[i].lvld, sizeof(voff_t *)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (id_offsets, &dbm[i].id_offsets, sizeof(goffset_t *)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (jid_offset, &dbm[i].jid_offset, sizeof(goffset_t *)));

		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (nbs, &dbm[i].nb, sizeof(vid_t *) * 8));
	}
}

static void set_globals_filter2_gpu (dbmeta_t * dbm, master_t * mst)
{
	int num_of_devices = mst->num_of_devices;

	int i;
#ifdef SINGLE_NODE
	int world_rank = mst->world_rank;
#endif
	for (i = 0; i < num_of_devices; i++)
	{
#ifdef SINGLE_NODE
		cudaSetDevice(world_rank * num_of_devices + i);
#else
		cudaSetDevice(i + DEVICE_SHIFT);
#endif
//		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (nbs, &dbm[i].nb, sizeof(vid_t *) * 8));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (jvalid, &dbm[i].jvld, sizeof(voff_t *)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (lvalid, &dbm[i].lvld, sizeof(voff_t *)));
	}
}

static void set_globals_preprocessing_gpu (dbmeta_t * dbm, master_t * mst)
{
	int num_of_devices = mst->num_of_devices;
#ifdef SINGLE_NODE
	int world_rank = mst->world_rank;
#endif
	int i;
	for (i = 0; i < num_of_devices; i++)
	{
#ifdef SINGLE_NODE
		cudaSetDevice(world_rank * num_of_devices + i);
#else
		cudaSetDevice(i + DEVICE_SHIFT);
#endif
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (id2index, &dbm[i].comm.id2index, sizeof(int *)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (receive_offsets, &dbm[i].comm.receive_offsets, sizeof(voff_t *)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (send_offsets, &dbm[i].comm.send_offsets, sizeof(voff_t *)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (tmp_send_offsets, &dbm[i].comm.tmp_send_offsets, sizeof(voff_t*)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (send, &dbm[i].comm.send, sizeof(void *)));

		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (index_offsets, &dbm[i].comm.index_offsets, sizeof(voff_t *)));
	}
}

static void set_globals_gather_gpu (dbmeta_t * dbm, master_t * mst)
{
	int num_of_devices = mst->num_of_devices;
#ifdef SINGLE_NODE
	int world_rank = mst->world_rank;
#endif

	int i;
	for (i = 0; i < num_of_devices; i++)
	{
#ifdef SINGLE_NODE
		cudaSetDevice(world_rank * num_of_devices + i);
#else
		cudaSetDevice(i + DEVICE_SHIFT);
#endif
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (jvalid, &dbm[i].jvld, sizeof(voff_t *)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (lvalid, &dbm[i].lvld, sizeof(voff_t *)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (adj_nbs, &dbm[i].djs.nbs, sizeof(vid_t *) * 8));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (spids, &dbm[i].djs.spids, sizeof(ull *)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (spidsr, &dbm[i].djs.spidsr, sizeof(ull *)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (junct_edges, &dbm[i].djs.edges, sizeof(ull *)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (posts, &dbm[i].dls.posts, sizeof(vid_t *)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (pres, &dbm[i].dls.pres, sizeof(vid_t *)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (post_edges, &dbm[i].dls.post_edges, sizeof(edge_type *)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (pre_edges, &dbm[i].dls.pre_edges, sizeof(edge_type *)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (spidlv, &dbm[i].dls.spids, sizeof(uint *)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (jkmers, &dbm[i].djs.kmers, sizeof(kmer_t *)));
	}
}

static void set_receive_buffer_gpu (void ** recv, int did, int world_rank, int num_of_devices)
{
#ifdef SINGLE_NODE
	CUDA_CHECK_RETURN (cudaSetDevice(world_rank * num_of_devices + did));
#else
	CUDA_CHECK_RETURN (cudaSetDevice(did + DEVICE_SHIFT));
#endif
	CUDA_CHECK_RETURN (cudaMemcpyToSymbol(receive, recv, sizeof(void*)));
}

}


__global__ void init_hashtab (uint size, voff_t index_offset)
{
	kmer_t * local_kmers = kmers + index_offset;
	ull * local_edges = edges + index_offset;
	vid_t * local_vids = vids + index_offset;

	entry_t * buf = (entry_t *)send;

	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int w = (size + TOTAL_THREADS_NODES - 1) / TOTAL_THREADS_NODES;
	int r;

	for (r = 0; r < w; r++)
	{
		int index = gid + r * TOTAL_THREADS_NODES;
		if (index >= size)
		{
			break;
		}
		if (buf[index].occupied)
		{
			local_kmers[index] = buf[index].kmer;
			local_edges[index] = buf[index].edge;
			local_vids[index] = buf[index].occupied;
		}
	}
}


__global__ void init_hashtab_gpu (uint size, voff_t index_offset)
{
	ull * local_edges = edges + index_offset;
	vid_t * local_vids = vids + index_offset;
	entry_t * local_ens = ens + index_offset;

	entry_t * buf = (entry_t *)send;

	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int w = (size + TOTAL_THREADS_NODES - 1) / TOTAL_THREADS_NODES;
	int r;

	for (r = 0; r < w; r++)
	{
		int index = gid + r * TOTAL_THREADS_NODES;
		if (index >= size)
		{
			break;
		}
		if (buf[index].occupied)
		{
			local_ens[index].kmer = buf[index].kmer;
			local_ens[index].edge = buf[index].edge;
			local_ens[index].occupied = buf[index].occupied;
			local_edges[index] = buf[index].edge;
			local_vids[index] = buf[index].occupied;
		}
	}
}

__device__ static int lookup_with_hash_gpu (hashval_t hashval, kmer_t * akmer, kmer_t * kmers, hashsize_t size, int tbid)
{
	hashval_t index, hash2;
	uint i;
//	uint threshold = size / 2;
	index = hashtab_mod (hashval, size_prime_index[tbid]);
	kmer_t * entry = kmers + index;
	if ((is_equal_kmer)(entry, akmer))
	{
		return index;
	}

	hash2 = hashtab_mod_m2 (hashval, size_prime_index[tbid]);
	for (i=0; i<size; i++)
//	for (i=0; i<threshold; i++)
	{
	     index += hash2;
	     if (index >= size)
	    	 index -= size;
	     entry = kmers + index;
	 	if ((is_equal_kmer)(entry, akmer))
	 	{
	 		return index;
	 	}
	}
	return -1;
}

__device__ static int lookup_kmer_assign_source_id (assid_t mssg, kmer_t * kmers, uint size, int pindex, voff_t index_offset)
{
	int edge;
	int stride;
	edge = (mssg.code >> (8*2)) && 0xff;
	uint seed = DEFAULT_SEED;

	hashval_t hash = murmur_hash3_32 ((uint *)&mssg.dst, seed);
	int index = lookup_with_hash_gpu (hash, &mssg.dst, kmers, size, pindex);
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
//		printf ("PINDEX=%d, self_id=%d, CANNOT FIND THE KMER %u\t%u\n", pindex, self_id, mssg.dst.x, mssg.dst.y);
		return -1;
	}
	else
	{
		nbs[edge + stride][index_offset + index] = mssg.srcid;
	}

	return 0;
}

__device__ int lookup_with_hash_vertices_gpu (hashval_t hashval, kmer_t * akmer, entry_t * vs, hashsize_t size, int pid)
{
	hashval_t index, hash2;
	uint i;
//	uint threshold = size / 2;
	index = hashtab_mod (hashval, size_prime_index[pid]);
	if (index >= size)
	{
		  printf ("index %u is larger than size %u\n", index, size);
		  return -1;
	}

	entry_t * entry = vs + index;
	if ((is_equal_kmer)(&(entry->kmer), akmer))
	{
		return index;
	}

	hash2 = hashtab_mod_m2 (hashval, size_prime_index[pid]);
	for (i=0; i<size; i++)
//	for (i=0; i<threshold; i++)
	{
	     index += hash2;
	     if (index >= size)
	    	 index -= size;
	     entry = vs + index;
	 	 if ((is_equal_kmer)(&(entry->kmer), akmer))
	 	 {
	 		return index;
	 	 }
	}
	return -1;
}

__device__ static int lookup_kmer_assign_source_id_gpu (assid_t mssg, entry_t * kmers, uint size, int pindex, voff_t index_offset)
{
	int edge;
	int stride;
	edge = (mssg.code >> (8*2)) && 0xff;
	uint seed = DEFAULT_SEED;

	hashval_t hash = murmur_hash3_32 ((uint *)&mssg.dst, seed);
	int index = lookup_with_hash_vertices_gpu (hash, &mssg.dst, kmers, size, pindex);
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
//		printf ("PINDEX=%d, self_id=%d, CANNOT FIND THE KMER %u\t%u\n", pindex, self_id, mssg.dst.x, mssg.dst.y);
		return -1;
	}
	else
	{
		nbs[edge + stride][index_offset + index] = mssg.srcid;
	}

	return 0;
}

__device__ static int get_id_for_push (kmer_t * kmer, edge_type edge, int k, int p, int num_of_partitions)
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
	get_reverse (&node[0], &node[1]);

	int table_index;
#ifdef LONG_KMER
	table_index = (128 - k * 2) / 32;
	shift_dictionary[table_index] (&node[1], 128 - k * 2);
#else
	table_index = (64 - k * 2) / 32;
	shift_dictionary[table_index] (&node[1], 64 - k * 2);
#endif

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
	msp_id_t mspid = get_partition_id (curr, p, num_of_partitions);
	return mspid;
}

__device__ static void get_adj_kmer (kmer_t * kmer, edge_type edge, int k, assid_t * mssg, msp_id_t pid)
{
	uint seed = DEFAULT_SEED;
	unit_kmer_t * ptr;
	kmer_t node[2];
	edge_type edges[2];
	edges[0] = edge;

	node[0].x = kmer->x;
	node[0].y = kmer->y;
#ifdef LONG_KMER
	node[0].z = kmer->z;
	node[0].w = kmer->w;
#endif

	get_reverse_edge (edges[1], node[0]);
	kmer_32bit_left_shift (&node[0], 2);
	ptr = (unit_kmer_t *)(&node[0]) + (k * 2) / KMER_UNIT_BITS;
	*ptr |= ((unit_kmer_t)edge) << (KMER_UNIT_BITS - (k * 2) % KMER_UNIT_BITS);
	get_reverse (&node[0], &node[1]);

	int table_index;
#ifdef LONG_KMER
	table_index = (128 - k * 2) / 32;
	shift_dictionary[table_index] (&node[1], 128 - k * 2);
#else
	table_index = (64 - k * 2) / 32;
	shift_dictionary[table_index] (&node[1], 64 - k * 2);
#endif

	int flag;
	hashval_t hash[2];
	hash[0] = murmur_hash3_32 ((uint *)&node[0], seed);
	hash[1] = murmur_hash3_32 ((uint *)&node[1], seed);
	if (hash[0] == hash[1])
	{
		int ret = compare_2kmers (&node[0], &node[1]);
		if (ret >= 0)
			flag = 0;
		else flag = 1;
	}
	else if (hash[0] < hash[1]) flag = 0;
	else flag = 1;

	mssg->dst = node[flag];
	if (flag == 0)
		mssg->code = (edges[1] << (8*2)) | REVERSE_FLAG | pid; // reverse flag 1 on the most significant 8 bits
	else
		mssg->code = (edges[1] << (8*2)) | pid; // reverse flag 0 on the most significant 8 bits
}

__global__ static void push_mssg_offset_assign_id (uint size, int num_of_partitions, voff_t index_offset, int k, int p, int cutoff)
{
	kmer_t * local_kmer = kmers + index_offset;
	ull * local_edge = edges + index_offset;
	vid_t * local_nbs[EDGE_DIC_SIZE];
	int i;
	for (i=0; i<EDGE_DIC_SIZE; i++)
	{
		local_nbs[i] = nbs[i] + index_offset;
	}

	const int gid = blockIdx.x*blockDim.x+threadIdx.x;
//	const int tid = threadIdx.x;
	int r,w;
	w = (size + TOTAL_THREADS_NODES - 1) / TOTAL_THREADS_NODES;
	for(r = 0; r < w; r++)
	{
		int index = gid + r * TOTAL_THREADS_NODES;
		if(index >= size)
			break;
		int pindex;
		kmer_t reverse;
		int i;
		for (i = 0; i < EDGE_DIC_SIZE / 2; i++)
		{
			if ((local_edge[index] >> (i*8) & 0xff) >= cutoff)
			{
				local_nbs[i][index] = get_id_for_push (&local_kmer[index], (edge_type) i, k, p, num_of_partitions);
				pindex = id2index[local_nbs[i][index]];
				atomicAdd(&send_offsets[pindex+1], 1);
			}
			else
				local_nbs[i][index] = DEADEND;
			if ((local_edge[index] >> ((EDGE_DIC_SIZE / 2 + i) * 8) & 0xff) >= cutoff)
			{
				get_reverse (&local_kmer[index], &reverse);
#ifdef LONG_KMER
				int table_index = (128 - k * 2) / 32;
				shift_dictionary[table_index] (&reverse, 128 - k * 2);
#else
				int table_index = (64 - k * 2) / 32;
				shift_dictionary[table_index] (&reverse, 64 - k * 2);
#endif
				local_nbs[EDGE_DIC_SIZE / 2 + i][index] = get_id_for_push (&reverse, (edge_type) i, k, p, num_of_partitions);
				pindex = id2index[local_nbs[EDGE_DIC_SIZE / 2 + i][index]];
				atomicAdd(&send_offsets[pindex+1], 1);
			}
			else
				local_nbs[EDGE_DIC_SIZE / 2 + i][index] = DEADEND; // dead end branch
		}
	}
}

__global__ static void push_mssg_offset_assign_id_gpu (uint size, int num_of_partitions, voff_t index_offset, int k, int p, int cutoff)
{
//	kmer_t * local_kmer = kmers + index_offset;
	entry_t * local_ens = ens + index_offset;
	ull * local_edge = edges + index_offset;
	vid_t * local_nbs[EDGE_DIC_SIZE];
	int i;
	for (i=0; i<EDGE_DIC_SIZE; i++)
	{
		local_nbs[i] = nbs[i] + index_offset;
	}

	const int gid = blockIdx.x*blockDim.x+threadIdx.x;
//	const int tid = threadIdx.x;
	int r,w;
	w = (size + TOTAL_THREADS_NODES - 1) / TOTAL_THREADS_NODES;
	for(r = 0; r < w; r++)
	{
		int index = gid + r * TOTAL_THREADS_NODES;
		if(index >= size)
			break;
		int pindex;
		kmer_t reverse;
		int i;
		for (i = 0; i < EDGE_DIC_SIZE / 2; i++)
		{
			if ((local_edge[index] >> (i*8) & 0xff) >= cutoff)
			{
				local_nbs[i][index] = get_id_for_push (&local_ens[index].kmer, (edge_type) i, k, p, num_of_partitions);
				pindex = id2index[local_nbs[i][index]];
				atomicAdd(&send_offsets[pindex+1], 1);
			}
			else
				local_nbs[i][index] = DEADEND;
			if ((local_edge[index] >> ((EDGE_DIC_SIZE / 2 + i) * 8) & 0xff) >= cutoff)
			{
				get_reverse (&local_ens[index].kmer, &reverse);
#ifdef LONG_KMER
				int table_index = (128 - k * 2) / 32;
				shift_dictionary[table_index] (&reverse, 128 - k * 2);
#else
				int table_index = (64 - k * 2) / 32;
				shift_dictionary[table_index] (&reverse, 64 - k * 2);
#endif
				local_nbs[EDGE_DIC_SIZE / 2 + i][index] = get_id_for_push (&reverse, (edge_type) i, k, p, num_of_partitions);
				pindex = id2index[local_nbs[EDGE_DIC_SIZE / 2 + i][index]];
				atomicAdd(&send_offsets[pindex+1], 1);
			}
			else
				local_nbs[EDGE_DIC_SIZE / 2 + i][index] = DEADEND; // dead end branch
		}
	}
}

__global__ static void push_mssg_assign_id (uint size, int num_of_partitions, voff_t index_offset, int k, int p, int cutoff)
{
	kmer_t * local_kmer = kmers + index_offset;
	ull * local_edge = edges + index_offset;
	vid_t * local_nbs[EDGE_DIC_SIZE];
	int i;
	for (i=0; i<EDGE_DIC_SIZE; i++)
	{
		local_nbs[i] = nbs[i] + index_offset;
	}
	vid_t * local_vids = vids + index_offset;
	assid_t * buf = (assid_t *) send;

	const int gid = blockIdx.x*blockDim.x+threadIdx.x;
	int r, w;
	w = (size + TOTAL_THREADS_NODES - 1)/ TOTAL_THREADS_NODES;
	for (r = 0; r < w; r++)
	{
		int index = gid + r * TOTAL_THREADS_NODES;
		if(index >= size)
			break;
		int pindex;
		msp_id_t pid;
		voff_t local_offset;
		voff_t off;
		assid_t tmp;
		kmer_t reverse;
		for (i=0; i<EDGE_DIC_SIZE/2; i++)
		{
			if ((local_edge[index] >> (i*8) & 0xff) >= cutoff)
			{
				pid = local_nbs[i][index];
				pindex = id2index[pid];
				local_offset = atomicAdd(&tmp_send_offsets[pindex+1], 1);
				off = local_offset + send_offsets[pindex];
				get_adj_kmer (&local_kmer[index], (edge_type) i, k, &tmp, pid);
				tmp.srcid = local_vids[index] - 1; // **************************** BE CAREFUL HERE !!!!!!!!!!! ***********************
				buf[off] = tmp;
			}
			if ((local_edge[index] >> ((EDGE_DIC_SIZE / 2 + i) * 8) & 0xff) >= cutoff)
			{
				get_reverse (&local_kmer[index], &reverse);
#ifdef LONG_KMER
				int table_index = (128 - k * 2) / 32;
				shift_dictionary[table_index] (&reverse, 128 - k * 2);
#else
				int table_index = (64 - k * 2) / 32;
				shift_dictionary[table_index] (&reverse, 64 - k * 2);
#endif
				pid = local_nbs[EDGE_DIC_SIZE / 2 + i][index];
				pindex = id2index[pid];
				local_offset = atomicAdd(&tmp_send_offsets[pindex+1], 1);
				off = local_offset + send_offsets[pindex];
				get_adj_kmer (&reverse, (edge_type) i, k, &tmp, pid);
				tmp.srcid = local_vids[index] - 1; // **************************** BE CAREFUL HERE !!!!!!!!!!! ***********************
				buf[off] = tmp;
			}
		}
	}
}

__global__ static void push_mssg_assign_id_gpu (uint size, int num_of_partitions, voff_t index_offset, int k, int p, int cutoff)
{
//	kmer_t * local_kmer = kmers + index_offset;
	entry_t * local_ens = ens + index_offset;
	ull * local_edge = edges + index_offset;
	vid_t * local_nbs[EDGE_DIC_SIZE];
	int i;
	for (i=0; i<EDGE_DIC_SIZE; i++)
	{
		local_nbs[i] = nbs[i] + index_offset;
	}
	vid_t * local_vids = vids + index_offset;
	assid_t * buf = (assid_t *) send;

	const int gid = blockIdx.x*blockDim.x+threadIdx.x;
	int r, w;
	w = (size + TOTAL_THREADS_NODES - 1)/ TOTAL_THREADS_NODES;
	for (r = 0; r < w; r++)
	{
		int index = gid + r * TOTAL_THREADS_NODES;
		if(index >= size)
			break;
		int pindex;
		msp_id_t pid;
		voff_t local_offset;
		voff_t off;
		assid_t tmp;
		kmer_t reverse;
		for (i=0; i<EDGE_DIC_SIZE/2; i++)
		{
			if ((local_edge[index] >> (i*8) & 0xff) >= cutoff)
			{
				pid = local_nbs[i][index];
				pindex = id2index[pid];
				local_offset = atomicAdd(&tmp_send_offsets[pindex+1], 1);
				off = local_offset + send_offsets[pindex];
				get_adj_kmer (&local_ens[index].kmer, (edge_type) i, k, &tmp, pid);
				tmp.srcid = local_vids[index] - 1; // **************************** BE CAREFUL HERE !!!!!!!!!!! ***********************
				buf[off] = tmp;
			}
			if ((local_edge[index] >> ((EDGE_DIC_SIZE / 2 + i) * 8) & 0xff) >= cutoff)
			{
				get_reverse (&local_ens[index].kmer, &reverse);
#ifdef LONG_KMER
				int table_index = (128 - k * 2) / 32;
				shift_dictionary[table_index] (&reverse, 128 - k * 2);
#else
				int table_index = (64 - k * 2) / 32;
				shift_dictionary[table_index] (&reverse, 64 - k * 2);
#endif
				pid = local_nbs[EDGE_DIC_SIZE / 2 + i][index];
				pindex = id2index[pid];
				local_offset = atomicAdd(&tmp_send_offsets[pindex+1], 1);
				off = local_offset + send_offsets[pindex];
				get_adj_kmer (&reverse, (edge_type) i, k, &tmp, pid);
				tmp.srcid = local_vids[index] - 1; // **************************** BE CAREFUL HERE !!!!!!!!!!! ***********************
				buf[off] = tmp;
			}
		}
	}
}

__global__ static void pull_mssg_assign_id (uint num_mssgs, int pid, uint psize, voff_t index_offset, int num_of_partitions, voff_t receive_start, bool intra_inter, int did)
{
	assid_t * buf;
	int pindex = id2index[pid];
	if (intra_inter)
		buf = (assid_t *)send + receive_start + send_offsets[pindex];
	else buf = (assid_t *)send + receive_start + receive_offsets[pindex];

	kmer_t * local_kmer = kmers + index_offset;

	const int gid = blockIdx.x*blockDim.x+threadIdx.x;
	int r, w;
	w = (num_mssgs + TOTAL_THREADS_NODES - 1)/ TOTAL_THREADS_NODES;
	for (r=0; r<w; r++)
	{
		int index = gid + r * TOTAL_THREADS_NODES;
		if (index >= num_mssgs)
			break;
		assid_t tmp = buf[index];
		msp_id_t id = tmp.code & ID_BITS;
		// CHECK
		if (id != pid)
			printf ("ERROR IN ID ENCODING!\n");
		if (lookup_kmer_assign_source_id (tmp, local_kmer, psize, pindex, index_offset) == -1)
		{
	//		atomicAdd (gpu_not_found, 1);
			printf ("KMER NOT FOUND: %u, %u, pindex=%d, pid=%d\n", tmp.dst.x, tmp.dst.y, pindex, pid); // assign the neigbhor id for kmer tmp.dst
		}
	}

}

__global__ static void pull_mssg_assign_id_gpu (uint num_mssgs, int pid, uint psize, voff_t index_offset, int num_of_partitions, voff_t receive_start, bool intra_inter, int did)
{
	assid_t * buf;
	int pindex = id2index[pid];
	if (intra_inter)
		buf = (assid_t *)send + receive_start + send_offsets[pindex];
	else buf = (assid_t *)send + receive_start + receive_offsets[pindex];

//	kmer_t * local_kmer = kmers + index_offset;
	entry_t * local_ens = ens + index_offset;

	const int gid = blockIdx.x*blockDim.x+threadIdx.x;
	int r, w;
	w = (num_mssgs + TOTAL_THREADS_NODES - 1)/ TOTAL_THREADS_NODES;
	for (r=0; r<w; r++)
	{
		int index = gid + r * TOTAL_THREADS_NODES;
		if (index >= num_mssgs)
			break;
		assid_t tmp = buf[index];
		msp_id_t id = tmp.code & ID_BITS;
		// CHECK
		if (id != pid)
			printf ("ERROR IN ID ENCODING!\n");
		if (lookup_kmer_assign_source_id_gpu (tmp, local_ens, psize, pindex, index_offset) == -1)
		{
	//		atomicAdd (gpu_not_found, 1);
			printf ("KMER NOT FOUND: %u, %u, pindex=%d, pid=%d\n", tmp.dst.x, tmp.dst.y, pindex, pid); // assign the neigbhor id for kmer tmp.dst
		}
	}

}

__global__ static void pull_mssg_assign_id_inter (uint num_mssgs, int pid, uint psize, voff_t index_offset, int num_of_partitions, voff_t receive_start, bool intra_inter, int did)
{
	assid_t * buf;
	int pindex = id2index[pid];
	if (intra_inter)
		buf = (assid_t *)send + receive_start + send_offsets[pindex];
	else buf = (assid_t *)send + receive_start + receive_offsets[pindex];

	kmer_t * local_kmer = kmers + index_offset;
	vid_t * local_vid = vids + index_offset;

	const int gid = blockIdx.x*blockDim.x+threadIdx.x;
	int r, w;
	w = (num_mssgs + TOTAL_THREADS_NODES - 1)/ TOTAL_THREADS_NODES;
	for (r=0; r<w; r++)
	{
		int index = gid + r * TOTAL_THREADS_NODES;
		if (index >= num_mssgs)
			break;
		assid_t tmp = buf[index];
		msp_id_t id = tmp.code & ID_BITS;
		// CHECK
		if (id != pid)
			printf ("ERROR IN ID ENCODING!\n");

		if ((int)local_vid[index] - 1 < 0)
			printf ("INDEX ERROR!!!!!!!!!\n");
		if (lookup_kmer_assign_source_id (tmp, local_kmer, psize, pindex, index_offset) == -1)
		{
			atomicAdd (gpu_not_found, 1);
			printf ("KMER NOT FOUND: %u, %u, pindex=%d, pid=%d\n", tmp.dst.x, tmp.dst.y, pindex, pid); // assign the neigbhor id for kmer tmp.dst
		}
	}

}


__global__ static void label_vertex_with_flags (uint size, voff_t index_offset, int cutoff)
{
	const uint gid = blockIdx.x * blockDim.x + threadIdx.x;
	uint r, w;
	w = (size + TOTAL_THREADS_NODES - 1) / TOTAL_THREADS_NODES;

	ull * local_edge = edges + index_offset;
	vid_t * local_vid = vids + index_offset;
	voff_t * local_jvalid = jvalid + index_offset;
	voff_t * local_lvalid = lvalid + index_offset;

	for (r = 0; r < w; r++)
	{
		if (gid + r * TOTAL_THREADS_NODES >= size)
			break;

		int ind = 0;
		int outd = 0;
		int index = gid + r * TOTAL_THREADS_NODES;

		if (local_vid[index] == 0)
			continue;

		int i;
		for (i = 0; i < EDGE_DIC_SIZE / 2; i++)
		{
			if ((*(local_edge + index) >> ((EDGE_DIC_SIZE / 2 + i) * 8) & 0xff) >= cutoff)
			{
				ind++;
			}
			if ((*(local_edge + index) >> (i*8) & 0xff) >= cutoff)
			{
				outd++;
			}
		}

		if (ind==0 && outd==0)
		{
			local_vid[index] = 0;
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
					if ((*(local_edge + index) >> (i*8) & 0xff) >= cutoff)
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
					if ((*(local_edge + index) >> (i*8) & 0xff) >= cutoff)
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
/*
__global__ static void assid_vertex_with_flags2 (uint size, int pid, voff_t index_offset)
{
	const uint gid = blockIdx.x * blockDim.x + threadIdx.x;
	uint r, w;
	w = (size + TOTAL_THREADS_NODES - 1) / TOTAL_THREADS_NODES;

	vid_t * local_vid = vids + index_offset;
	voff_t * local_jvalid = jvalid + index_offset;
	voff_t * local_lvalid = lvalid + index_offset;

	for (r = 0; r < w; r++)
	{
		if (gid + r * TOTAL_THREADS_NODES >= size)
			break;

		int index = gid + r * TOTAL_THREADS_NODES;
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
			local_vid[index] = 0;
			continue;
		}

		// **************************** BE CAREFUL HERE !!!!!!!!!!! ***********************
		if (jflag==true) // a junction
			local_vid[index] = local_jvalid[index] + id_offsets[pid]; // index+1
		else if (lflag) // a linear vertex
			local_vid[index] = jid_offset[pid] + id_offsets[pid] + local_lvalid[index]; // index+1
	}
}
*/
__global__ static void label_vertex_with_flags_binary (uint size, voff_t index_offset, int cutoff)
{
	const uint gid = blockIdx.x * blockDim.x + threadIdx.x;
	uint r, w;
	w = (size + TOTAL_THREADS_NODES - 1) / TOTAL_THREADS_NODES;

	ull * local_edge = edges + index_offset;
	voff_t * local_jvalid = jvalid;
	voff_t * local_lvalid = lvalid;
	vid_t * local_vid = vids + index_offset;

	for (r = 0; r < w; r++)
	{
		if (gid + r * TOTAL_THREADS_NODES >= size)
			break;

		int ind = 0;
		int outd = 0;
		int index = gid + r * TOTAL_THREADS_NODES;

		if (local_vid[index] == 0)
			continue;

		int i;
		for (i = 0; i < EDGE_DIC_SIZE / 2; i++)
		{
			if ((*(local_edge + index) >> ((EDGE_DIC_SIZE / 2 + i) * 8) & 0xff) >= cutoff)
			{
				ind++;
			}
			if ((*(local_edge + index) >> (i*8) & 0xff) >= cutoff)
			{
				outd++;
			}
		}

		if (ind == 0 && outd == 0) // filter isolated vertices
		{
			local_vid[index] = 0;
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
					if ((*(local_edge + index) >> (i*8) & 0xff) >= cutoff)
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
					if ((*(local_edge + index) >> (i*8) & 0xff) >= cutoff)
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

__global__ static void assid_vertex_with_flags (uint size, int pid, voff_t index_offset)
{
	const uint gid = blockIdx.x * blockDim.x + threadIdx.x;
	uint r, w;
	w = (size + TOTAL_THREADS_NODES - 1) / TOTAL_THREADS_NODES;

	vid_t * local_vid = vids + index_offset;
	voff_t * local_jvalid = jvalid;
	voff_t * local_lvalid = lvalid;

	for (r = 0; r < w; r++)
	{
		if (gid + r * TOTAL_THREADS_NODES >= size)
			break;

		int index = gid + r * TOTAL_THREADS_NODES;
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
			local_vid[index] = 0;
			continue;
		}

		// **************************** BE CAREFUL HERE !!!!!!!!!!! ***********************
		if (jflag==true) // a junction
			local_vid[index] = local_jvalid[index]; // index+1
		else if (lflag) // a linear vertex
			local_vid[index] = jid_offset[pid] + local_lvalid[index]; // index+1
	}
}


__global__ static void gather_vertex (uint size, int pid, voff_t index_offset, int cutoff)
{
	const uint gid = blockIdx.x * blockDim.x + threadIdx.x;
	uint r, w;
	w = (size + TOTAL_THREADS_NODES - 1) / TOTAL_THREADS_NODES;

	ull * local_edge = edges + index_offset;
	vid_t * local_vid = vids + index_offset;
	vid_t * local_nbs[EDGE_DIC_SIZE];
	int i;
	for (i=0; i<EDGE_DIC_SIZE; i++)
	{
		local_nbs[i] = nbs[i] + index_offset;
	}

	for (r = 0; r < w; r++)
	{
		if (gid + r * TOTAL_THREADS_NODES >= size)
			break;

		int index = gid + r * TOTAL_THREADS_NODES;

		if (local_vid[index] == 0)
			continue;

		voff_t off = local_vid[index] - id_offsets[pid] - 1;
		if (local_vid[index] - id_offsets[pid] <= jid_offset[pid]) // a junction here
		{
			for (i=0; i<EDGE_DIC_SIZE; i++)
			{
				adj_nbs[i][off] = local_nbs[i][index];
			}
		}
		else // a linear vertex
		{
			for (i = 0; i < EDGE_DIC_SIZE / 2; i++)
			{
				if ((*(local_edge + index) >> (i*8) & 0xff) >= cutoff)
				{
					posts[off - jid_offset[pid]] = local_nbs[i][index];
					break;
				}
			}
			for (i = EDGE_DIC_SIZE / 2; i < EDGE_DIC_SIZE; i++)
			{
				if ((*(local_edge + index) >> (i*8) & 0xff) >= cutoff)
				{
					pres[off - jid_offset[pid]] = local_nbs[i][index];
					break;
				}
			}
		}
	}
}

__global__ void init_kmers (uint size, voff_t index_offset)
{
	entry_t * buf = (entry_t *) send;

	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int w = (size + TOTAL_THREADS_NODES - 1) / TOTAL_THREADS_NODES;
	int r;
	for (r=0; r<w; r++)
	{
		int index = gid + r * TOTAL_THREADS_NODES;
		if (index >= size)
		{
			break;
		}
		if (buf[index].occupied)
		{
			lvalid[index+1] = 1;
		}
	}
}

// ************ Only for kmers with length < 31
__global__ void gather_kmers (uint size, voff_t index_offset)
{
	entry_t * buf = (entry_t *) send;

	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int w = (size + TOTAL_THREADS_NODES - 1) / TOTAL_THREADS_NODES;
	int r;
	for (r=0; r<w; r++)
	{
		int index = gid + r * TOTAL_THREADS_NODES;
		if (index >= size)
			break;
		if (buf[index].occupied)
		{
			before_sort[lvalid[index]] = buf[index].kmer;
			before_vids[lvalid[index]] = index;
		}
	}
}

__global__ void gather_edges (uint size, voff_t index_offset)
{
	ull * local_edges = edges + index_offset;
	vid_t * local_vid = vids + index_offset;
	entry_t * buf = (entry_t *) send;

	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int w = (size + TOTAL_THREADS_NODES - 1) / TOTAL_THREADS_NODES;
	int r;
	for (r=0; r<w; r++)
	{
		int index = gid + r * TOTAL_THREADS_NODES;
		if (index >= size)
			break;
		uint hindex = local_vid[index];
		local_edges[index] = buf[hindex].edge;
	}
}

__global__ void gather_vs (uint size, voff_t index_offset)
{
	ull * local_edges = edges + index_offset;
	vid_t * local_vid = vids + index_offset;
	kmer_t * local_kmers = sorted_kmers + index_offset;
	entry_t * buf = (entry_t *) send;

	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int w = (size + TOTAL_THREADS_NODES - 1) / TOTAL_THREADS_NODES;
	int r;
	for (r=0; r<w; r++)
	{
		int index = gid + r * TOTAL_THREADS_NODES;
		if (index >= size)
			break;
		local_kmers[index] = buf[index].kmer;
		local_edges[index] = buf[index].edge;
		local_vid[index] = buf[index].occupied;
	}
}

__device__ static int get_adj_id_from_post (kmer_t * kmer, edge_type edge, int k, int p, uint size, int num_of_partitions)
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
	msp_id_t mspid = get_partition_id (curr, p, num_of_partitions);

	return mspid;
}

__device__ static int get_adj_id_from_pre (kmer_t * kmer, edge_type edge, int k, int p, uint size, int num_of_partitions)
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
	msp_id_t mspid = get_partition_id (curr, p, num_of_partitions);

	return mspid;
}

__device__ static void get_adj_mssg_from_post (kmer_t * kmer, edge_type edge, int k, assid_t * mssg, msp_id_t pid)
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
		int ret = compare_2kmers (&node[0], &node[1]);
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
	}
	else
		mssg->code = (((uint)edges[1]) << (8*2)) | pid;
}

__device__ static void get_adj_mssg_from_pre (kmer_t * kmer, edge_type edge, int k, assid_t * mssg, msp_id_t pid)
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
		int ret = compare_2kmers (&node[0], &node[1]);
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
	}
	else
		mssg->code = (((uint)edges[0]) << (8*2)) | pid;
}


__device__ static int binary_search_kmers (kmer_t * akmer, kmer_t * vs, uint size)
{
	int begin = 0;
	int end = size - 1;
	int index = (begin + end) / 2;
	int ret;
	while (begin <= end)
	{
		ret = compare_2kmers (&vs[index], akmer);
		if (ret == 0)
			return index; // RETURN INDEX of the kmer!!!
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

__global__ static void push_mssg_offset_assign_id_binary (voff_t size, int num_of_partitions, voff_t index_offset, int k, int p, int cutoff)
{
	ull * local_edge = edges + index_offset;
	vid_t * local_nbs[EDGE_DIC_SIZE];
	int i;
	for (i=0; i<EDGE_DIC_SIZE; i++)
	{
		local_nbs[i] = nbs[i] + index_offset;
	}

	const int gid = blockIdx.x*blockDim.x+threadIdx.x;
//	const int tid = threadIdx.x;
	voff_t r,w;
	w = (size + TOTAL_THREADS_NODES - 1) / TOTAL_THREADS_NODES;
	for(r = 0; r < w; r++)
	{
		int index = gid + r * TOTAL_THREADS_NODES;
		if(index >= size)
			break;
		int pindex;
		int i;
		for (i = 0; i < EDGE_DIC_SIZE / 2; i++)
		{
			if ((local_edge[index] >> (i*8) & 0xff) >= cutoff)
			{
//				local_nbs[i][index] = get_adj_id_from_post (&curr_kmer, (edge_type)i, k, p, size, num_of_partitions);
				if (local_nbs[i][index] > num_of_partitions)
				{
					printf ("ERROR IN GETTING MSP ID!!!!!!\n");
				}
				pindex = id2index[local_nbs[i][index]];
				atomicAdd(&send_offsets[pindex+1], 1);
			}
			else
				local_nbs[i][index] = DEADEND;
			if ((local_edge[index] >> ((EDGE_DIC_SIZE / 2 + i) * 8) & 0xff) >= cutoff)
			{
//				local_nbs[EDGE_DIC_SIZE / 2 + i][index] = get_adj_id_from_pre (&curr_kmer, (edge_type)i, k, p, size, num_of_partitions);
				if (local_nbs[EDGE_DIC_SIZE / 2 + i][index] > num_of_partitions)
				{
					printf ("ERROR IN GETTING MSP ID!!!\n");
				}
				pindex = id2index[local_nbs[EDGE_DIC_SIZE / 2 + i][index]];
				atomicAdd(&send_offsets[pindex+1], 1);
			}
			else
				local_nbs[EDGE_DIC_SIZE / 2 + i][index] = DEADEND; // dead end branch
		}
	}
}

__global__ static void push_mssg_assign_id_binary (uint size, int num_of_partitions, voff_t index_offset, int k, int p, int cutoff)
{
//	ull * local_kmer = sorted_kmers + index_offset;
	kmer_t * local_kmer = sorted_kmers + index_offset;
	ull * local_edge = edges + index_offset;
	vid_t * local_nbs[EDGE_DIC_SIZE];
	int i;
	for (i=0; i<EDGE_DIC_SIZE; i++)
	{
		local_nbs[i] = nbs[i] + index_offset;
	}
	vid_t * local_vids = vids + index_offset;
	assid_t * buf = (assid_t *) send;

	const int gid = blockIdx.x*blockDim.x+threadIdx.x;
	int r, w;
	w = (size + TOTAL_THREADS_NODES - 1)/ TOTAL_THREADS_NODES;
	for (r = 0; r < w; r++)
	{
		int index = gid + r * TOTAL_THREADS_NODES;
		if(index >= size)
			break;
		int pindex;
		msp_id_t pid;
		voff_t local_offset;
		voff_t off;
		assid_t tmp;
		kmer_t curr_kmer;
		curr_kmer = local_kmer[index];
		for (i=0; i<EDGE_DIC_SIZE/2; i++)
		{
			if ((local_edge[index] >> (i*8) & 0xff) >= cutoff)
			{
				pid = local_nbs[i][index];
				pindex = id2index[pid];
				local_offset = atomicAdd(&tmp_send_offsets[pindex+1], 1);
				off = local_offset + send_offsets[pindex];
				get_adj_mssg_from_post (&curr_kmer, (edge_type)i, k, &tmp, pid);
				tmp.srcid = local_vids[index] - 1; // **************************** BE CAREFUL HERE !!!!!!!!!!! ***********************
				buf[off] = tmp;
			}
			if ((local_edge[index] >> ((EDGE_DIC_SIZE / 2 + i) * 8) & 0xff) >= cutoff)
			{
				pid = local_nbs[EDGE_DIC_SIZE / 2 + i][index];
				pindex = id2index[pid];
				local_offset = atomicAdd(&tmp_send_offsets[pindex+1], 1);
				off = local_offset + send_offsets[pindex];
				get_adj_mssg_from_pre (&curr_kmer, (edge_type)i, k, &tmp, pid);
				tmp.srcid = local_vids[index] - 1; // **************************** BE CAREFUL HERE !!!!!!!!!!! ***********************
				buf[off] = tmp;
			}
		}
	}
}

__global__ static void push_mssg_offset_shakehands (voff_t size, int num_of_partitions, voff_t index_offset, int k, int p, int cutoff)
{
	kmer_t * local_kmer = sorted_kmers + index_offset;
	ull * local_edge = edges + index_offset;
	vid_t * local_nbs[EDGE_DIC_SIZE];
	int i;
	for (i=0; i<EDGE_DIC_SIZE; i++)
	{
		local_nbs[i] = nbs[i] + index_offset;
	}

	const int gid = blockIdx.x*blockDim.x+threadIdx.x;
	voff_t r,w;
	w = (size + TOTAL_THREADS_NODES - 1) / TOTAL_THREADS_NODES;
	for(r = 0; r < w; r++)
	{
		int index = gid + r * TOTAL_THREADS_NODES;
		if(index >= size)
			break;
		int pindex;
		kmer_t curr_kmer;
		curr_kmer = local_kmer[index];
		int i;
		for (i = 0; i < EDGE_DIC_SIZE / 2; i++)
		{
			if ((local_edge[index] >> (i*8) & 0xff) >= cutoff)
			{
				local_nbs[i][index] = get_adj_id_from_post (&curr_kmer, (edge_type)i, k, p, size, num_of_partitions);
				if (local_nbs[i][index] > num_of_partitions)
				{
					printf ("ERROR IN GETTING MSP ID!!!!!!\n");
				}
				pindex = id2index[local_nbs[i][index]];
				atomicAdd(&send_offsets[pindex+1], 1);
			}
			else
				local_nbs[i][index] = DEADEND;
			if ((local_edge[index] >> ((EDGE_DIC_SIZE / 2 + i) * 8) & 0xff) >= cutoff)
			{
				local_nbs[EDGE_DIC_SIZE / 2 + i][index] = get_adj_id_from_pre (&curr_kmer, (edge_type)i, k, p, size, num_of_partitions);
				if (local_nbs[EDGE_DIC_SIZE / 2 + i][index] > num_of_partitions)
				{
					printf ("ERROR IN GETTING MSP ID!!!\n");
				}
				pindex = id2index[local_nbs[EDGE_DIC_SIZE / 2 + i][index]];
				atomicAdd(&send_offsets[pindex+1], 1);
			}
			else
				local_nbs[EDGE_DIC_SIZE / 2 + i][index] = DEADEND; // dead end branch
		}
	}
}

__global__ static void push_mssg_shakehands (voff_t size, int num_of_partitions, voff_t index_offset, int k, int p, int curr_id, int cutoff)
{
//	ull * local_kmer = sorted_kmers + index_offset;
	kmer_t * local_kmer = sorted_kmers + index_offset;
	ull * local_edge = edges + index_offset;
	vid_t * local_nbs[EDGE_DIC_SIZE];
	int i;
	for (i=0; i<EDGE_DIC_SIZE; i++)
	{
		local_nbs[i] = nbs[i] + index_offset;
	}
	shakehands_t * buf = (shakehands_t *) send;

	const int gid = blockIdx.x*blockDim.x+threadIdx.x;
	int r, w;
	w = (size + TOTAL_THREADS_NODES - 1)/ TOTAL_THREADS_NODES;
	for (r = 0; r < w; r++)
	{
		int index = gid + r * TOTAL_THREADS_NODES;
		if(index >= size)
			break;
		int pindex;
		msp_id_t pid;
		voff_t local_offset;
		voff_t off;
		shakehands_t tmp;
		kmer_t curr_kmer;
		curr_kmer = local_kmer[index];
		for (i=0; i<EDGE_DIC_SIZE/2; i++)
		{
			if ((local_edge[index] >> (i*8) & 0xff) >= cutoff)
			{
				pid = local_nbs[i][index];
				pindex = id2index[pid];
				local_offset = atomicAdd(&tmp_send_offsets[pindex+1], 1);
				off = local_offset + send_offsets[pindex];
				get_adj_mssg_from_post (&curr_kmer, (edge_type)i, k, (assid_t*)(&tmp), curr_id);
				buf[off] = tmp;
			}
			if ((local_edge[index] >> ((EDGE_DIC_SIZE / 2 + i) * 8) & 0xff) >= cutoff)
			{
				pid = local_nbs[EDGE_DIC_SIZE / 2 + i][index];
				pindex = id2index[pid];
				local_offset = atomicAdd(&tmp_send_offsets[pindex+1], 1);
				off = local_offset + send_offsets[pindex];
				get_adj_mssg_from_pre (&curr_kmer, (edge_type)i, k, (assid_t*)(&tmp), curr_id);
				buf[off] = tmp;
			}
		}
	}
}

__global__ static void push_mssg_offset_respond (voff_t num_mssgs, int pid, voff_t psize, voff_t index_offset, int num_of_partitions, voff_t receive_start, bool intra_inter, int k, int p, int cutoff)
{
	shakehands_t * buf;
	if (intra_inter)
	{
		int pindex = id2index[pid];
		buf = (shakehands_t *)send + receive_start + send_offsets[pindex];
	}
	else
	{
		int pindex = id2index[pid];
		buf = (shakehands_t *)send + receive_start + receive_offsets[pindex];
	}

	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int r, w;
	w = (num_mssgs + TOTAL_THREADS_NODES - 1) / TOTAL_THREADS_NODES;
	kmer_t * local_kmer = sorted_kmers + index_offset;
	ull * local_edge = edges + index_offset;

	for (r=0; r<w; r++)
	{
		int index = gid + r * TOTAL_THREADS_NODES;
		if (index >= num_mssgs)
			break;
		shakehands_t tmp = buf[index];
		int ret = binary_search_kmers (&buf[index].dst, local_kmer, psize);
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
			if ((local_edge[ret] >> ((stride + edge) * 8) & 0xff) < cutoff)
			{
				match = -1;
			}
		}
		if (ret == -1 || match == -1)
		{
//			printf ("GPU KMER NOT FOUND: %u, %u, pid=%d\n", tmp.dst.x, tmp.dst.y, pid);
			msp_id_t pid = tmp.code & ID_BITS;
			int pindex = id2index[pid];
			atomicAdd(&extra_send_offsets[pindex + 1], 1);
		}
	}
}

__global__ static void push_mssg_respond (uint num_mssgs, int pid, voff_t psize, voff_t index_offset, int num_of_partitions, voff_t receive_start, bool intra_inter, int k, int p, int cutoff)
{
	shakehands_t * buf;
	shakehands_t * vs = (shakehands_t *)receive;
	if (intra_inter)
	{
		int pindex = id2index[pid];
		buf = (shakehands_t *)send + receive_start + send_offsets[pindex];
	}
	else
	{
		int pindex = id2index[pid];
		buf = (shakehands_t *)send + receive_start + receive_offsets[pindex];
	}

	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int r, w;
	w = (num_mssgs + TOTAL_THREADS_NODES - 1) / TOTAL_THREADS_NODES;
	kmer_t * local_kmer = sorted_kmers + index_offset;
	ull * local_edge = edges + index_offset;

	for (r=0; r<w; r++)
	{
		int index = gid + r * TOTAL_THREADS_NODES;
		if (index >= num_mssgs)
			break;
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
		int ret = binary_search_kmers (&buf[index].dst, local_kmer, psize);
		int match = 0;
		if (ret != -1)
		{
			if ((local_edge[ret] >> ((stride + edge) * 8) & 0xff) < cutoff)
			{
				match = -1;
			}
		}
		if (ret == -1 || match == -1)
		{
//			printf ("GPU KMER NOT FOUND: %u, %u, pid=%d\n", tmp.dst.x, tmp.dst.y, pid);
			msp_id_t pid = tmp.code & ID_BITS;
			if (stride == 0)
				get_adj_mssg_from_post (&tmp.dst, edge, k, (assid_t*)(&tmp), pid);
			else
				get_adj_mssg_from_pre (&tmp.dst, edge, k, (assid_t*)(&tmp), pid);
			int pindex = id2index[pid];
			voff_t local_offset = atomicAdd(&tmp_send_offsets[pindex+1], 1);
			voff_t off = local_offset + extra_send_offsets[pindex];
			vs[off] = tmp;
		}
	}
}

__device__ static int lookup_kmer_set_edge_zero_binary (shakehands_t * mssg, kmer_t * vs, ull * edges, uint size)
{
	edge_type edge;
	int stride;
	edge = (mssg->code >> (8*2)) & 0xff;
	if (edge > 3)
	{
		printf ("Encoded edge error!!!!!!!!!! %u\n", edge);
	}

	int index = binary_search_kmers (&mssg->dst, vs, size);
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
		atomicAnd (&edges[index], zerotable[edge+stride]);
	}

	return 0;
}

__global__ static void pull_mssg_respond (uint num_mssgs, int pid, voff_t psize, voff_t index_offset, void * local_receive, bool intra_inter)
{
	shakehands_t * buf;
	int pindex = id2index[pid];

	if (intra_inter)
	{
		buf = (shakehands_t *)local_receive + extra_send_offsets[pindex];
	}
	else
	{
		buf = (shakehands_t *)local_receive + receive_offsets[pindex];
	}

	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int r, w;
	w = (num_mssgs + TOTAL_THREADS_NODES - 1) / TOTAL_THREADS_NODES;
	kmer_t * local_kmer = sorted_kmers + index_offset;
	ull * local_edge = edges + index_offset;

	for (r=0; r<w; r++)
	{
		int index = gid + r * TOTAL_THREADS_NODES;
		if (index >= num_mssgs)
			break;

		if (lookup_kmer_set_edge_zero_binary (&buf[index], local_kmer, local_edge, psize) == -1)
		{
			printf ("GPU:: RESPOND ERROR WITH SOURCE KMER: %u, %u, pindex=%d, pid=%d\n", buf[index].dst.x, buf[index].dst.y, pindex, pid);
		}
	}
}

__device__ static int binary_search (ull akmer, ull * kmers, uint size)
{
	int begin = 0;
	int end = size - 1;
	int index = (begin + end) / 2;
	while (begin <= end)
	{
		if (akmer == kmers[index])
			return index; // RETURN INDEX of the kmer!!! SOMETIMES WE MAY NEED TO RETURN (INDEX+1) TO SET IT A POSSITIVE NUMBER!!!!!!!!!!!!
		else if (akmer < kmers[index])
		{
			end = index - 1;
		}
		else if (akmer > kmers[index])
		{
			begin = index + 1;
		}
		index = (begin + end) / 2;
	}
//	printf ("!!!!!!!!!!!!! Error occurs here: %u, %u\n", akmer->x, akmer->y);
	return -1;
}


//__device__ static int lookup_kmer_assign_source_id_binary (assid_t mssg, ull * kmers, uint size, int pindex, voff_t index_offset)
__device__ static int lookup_kmer_assign_source_id_binary (assid_t * mssg, kmer_t * kmers, uint size, int pindex, voff_t index_offset)
{
	int edge;
	int stride;
	edge = (mssg->code >> (8*2)) & 0xff;

//	ull dst = ((ull) mssg.dst.x) << 32;
//	dst |= (ull) mssg.dst.y;

//	int index = binary_search (dst, kmers, size);
	int index = binary_search_kmers (&mssg->dst, kmers, size);
	if (mssg->code >> (8*3))
	{
		stride = EDGE_DIC_SIZE / 2;
	}
	else
	{
		stride = 0;
	}
	if (index==-1)
	{
//		printf ("PINDEX=%d, self_id=%d, CANNOT FIND THE KMER %u\t%u\n", pindex, self_id, mssg.dst.x, mssg.dst.y);
		return -1;
	}
	else
	{
		nbs[edge + stride][index_offset + index] = mssg->srcid;
	}

	return 0;
}

__global__ static void pull_mssg_assign_id_binary (uint num_mssgs, int pid, uint psize, voff_t index_offset, int num_of_partitions, voff_t receive_start, bool intra_inter, int did)
{
	assid_t * buf;
	int pindex = id2index[pid];
	if (intra_inter)
		buf = (assid_t *)send + receive_start + send_offsets[pindex];
	else buf = (assid_t *)send + receive_start + receive_offsets[pindex];

//	ull * local_kmer = sorted_kmers + index_offset;
	kmer_t * local_kmer = sorted_kmers + index_offset;

	const int gid = blockIdx.x*blockDim.x+threadIdx.x;
	int r, w;
	w = (num_mssgs + TOTAL_THREADS_NODES - 1)/ TOTAL_THREADS_NODES;
	for (r=0; r<w; r++)
	{
		int index = gid + r * TOTAL_THREADS_NODES;
		if (index >= num_mssgs)
			break;
		assid_t tmp = buf[index];
		msp_id_t id = tmp.code & ID_BITS;
		// CHECK
		if (id != pid)
			printf ("ERROR IN ID ENCODING!\n");
		if (lookup_kmer_assign_source_id_binary (&buf[index], local_kmer, psize, pindex, index_offset) == -1)
		{
			printf ("KMER NOT FOUND: %u, %u, pindex=%d, pid=%d\n", tmp.dst.x, tmp.dst.y, pindex, pid); // assign the neigbhor id for kmer tmp.dst
		}
	}
}

__global__ static void gather_vertex_binary (uint size, int pid, voff_t index_offset, int cutoff)
{
	const uint gid = blockIdx.x * blockDim.x + threadIdx.x;
	uint r, w;
	w = (size + TOTAL_THREADS_NODES - 1) / TOTAL_THREADS_NODES;

	ull * local_edge = edges + index_offset;
	vid_t * local_vid = vids + index_offset;
	vid_t * local_nbs[EDGE_DIC_SIZE];
//	ull * local_kmers = sorted_kmers + index_offset;
	kmer_t * local_kmers = sorted_kmers + index_offset;
	int i;
	for (i=0; i<EDGE_DIC_SIZE; i++)
	{
		local_nbs[i] = nbs[i] + index_offset;
	}

	for (r = 0; r < w; r++)
	{
		if (gid + r * TOTAL_THREADS_NODES >= size)
			break;

		int index = gid + r * TOTAL_THREADS_NODES;
		if (local_vid[index] == 0)
			continue;

		voff_t off = local_vid[index] - id_offsets[pid] - 1;
		if (local_vid[index] - id_offsets[pid] <= jid_offset[pid]) // a junction here
		{
			for (i=0; i<EDGE_DIC_SIZE; i++)
			{
				adj_nbs[i][off] = local_nbs[i][index];
			}
			jkmers[off] = local_kmers[index];
			junct_edges[off] = local_edge[index];
		}
		else // a linear vertex
		{
			for (i = 0; i < EDGE_DIC_SIZE / 2; i++)
			{
				if ((*(local_edge + index) >> (i*8) & 0xff) >= cutoff)
				{
					posts[off - jid_offset[pid]] = local_nbs[i][index];
					post_edges[off - jid_offset[pid]] = i;
					break;
				}
			}
			for (i = EDGE_DIC_SIZE / 2; i < EDGE_DIC_SIZE; i++)
			{
				if ((*(local_edge + index) >> (i*8) & 0xff) >= cutoff)
				{
					pres[off - jid_offset[pid]] = local_nbs[i][index];
					pre_edges[off - jid_offset[pid]] = i - EDGE_DIC_SIZE/2;
					break;
				}
			}
		}
	}
}

__global__ static void gather_vertex_partitioned (uint size, int pid, voff_t index_offset, int cutoff, int k, int p, int num_of_partitions)
{
	const uint gid = blockIdx.x * blockDim.x + threadIdx.x;
	uint r, w;
	w = (size + TOTAL_THREADS_NODES - 1) / TOTAL_THREADS_NODES;

	ull * local_edge = edges + index_offset;
	vid_t * local_vid = vids + index_offset;
	vid_t * local_nbs[EDGE_DIC_SIZE];
//	ull * local_kmers = sorted_kmers + index_offset;
	kmer_t * local_kmers = sorted_kmers + index_offset;
	int i;
	for (i=0; i<EDGE_DIC_SIZE; i++)
	{
		local_nbs[i] = nbs[i] + index_offset;
	}

	for (r = 0; r < w; r++)
	{
		if (gid + r * TOTAL_THREADS_NODES >= size)
			break;

		int index = gid + r * TOTAL_THREADS_NODES;
		if (local_vid[index] == 0)
			continue;

		voff_t off = local_vid[index] - 1;
		if (local_vid[index] <= jid_offset[pid]) // a junction here
		{
			for (i=0; i<EDGE_DIC_SIZE/2; i++)
			{
				if ((local_edge[index] >> (i*8) & 0xff) >= cutoff)
				{
					spids[off] |= ((ull)(get_adj_id_from_post (&local_kmers[index], (edge_type)i, k, p, size, num_of_partitions))) << (i*SPID_BITS);
				}
				adj_nbs[i][off] = local_nbs[i][index];
			}
			for (i=EDGE_DIC_SIZE/2; i<EDGE_DIC_SIZE; i++)
			{
				if ((local_edge[index] >> (i*8) & 0xff) >= cutoff)
				{
					spidsr[off] |= ((ull)(get_adj_id_from_pre (&local_kmers[index], (edge_type) (i-EDGE_DIC_SIZE/2), k, p, size, num_of_partitions))) << ((i-EDGE_DIC_SIZE/2)*SPID_BITS);
				}
				adj_nbs[i][off] = local_nbs[i][index];
			}
			jkmers[off] = local_kmers[index];
			junct_edges[off] = local_edge[index];
		}
		else // a linear vertex
		{
			for (i = 0; i < EDGE_DIC_SIZE / 2; i++)
			{
				if ((*(local_edge + index) >> (i*8) & 0xff) >= cutoff)
				{
					spidlv[off - jid_offset[pid]] |= (get_adj_id_from_post (&local_kmers[index], (edge_type) i, k, p, size, num_of_partitions));
					posts[off - jid_offset[pid]] = local_nbs[i][index];
					post_edges[off - jid_offset[pid]] = i;
					break;
				}
			}
			for (i = EDGE_DIC_SIZE / 2; i < EDGE_DIC_SIZE; i++)
			{
				if ((*(local_edge + index) >> (i*8) & 0xff) >= cutoff)
				{
					spidlv[off - jid_offset[pid]] |= get_adj_id_from_pre (&local_kmers[index], (edge_type) (i-EDGE_DIC_SIZE/2), k, p, size, num_of_partitions) << SPID_BITS;
					pres[off - jid_offset[pid]] = local_nbs[i][index];
					pre_edges[off - jid_offset[pid]] = i - EDGE_DIC_SIZE/2;
					break;
				}
			}
		}
	}
}

#endif /* PREPROCESS_CUH_ */
