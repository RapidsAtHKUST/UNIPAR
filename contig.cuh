/*
 * contig.cuh
 *
 *  Created on: 2018-5-10
 *      Author: qiushuang
 *
 *  Device kernels and functions for contigs
 */

#ifndef CONTIG_CUH_
#define CONTIG_CUH_

#include "include/dbgraph.h"
#include "include/comm.h"
#include "bitkmer.cuh"

#define MAX_NUM_BLOCKS 1024 // for performance, setting it to be 1024; the maximum number of blocks in GPU can be at least 4096
#define THREADS_PER_BLOCK_NODES 1024
#define TOTAL_THREADS_NODES (THREADS_PER_BLOCK_NODES * MAX_NUM_BLOCKS)
#define THREADS_PER_BLOCK_NODES_TEST THREADS_PER_BLOCK_NODES
#define MAX_NUM_BLOCKS_TEST MAX_NUM_BLOCKS
#define TOTAL_THREADS_NODES_TEST (THREADS_PER_BLOCK_NODES_TEST * MAX_NUM_BLOCKS_TEST)

__constant__ static voff_t * id_offsets; // used to assign global id for each node in subgraphs
__constant__ static voff_t * jid_offset; // junction vertex id offsets, used to calculate id of each vertex from its index

/*CSR for junctions: */
__constant__ static voff_t * jnb_offsets;
__constant__ static vid_t * jnbs;

__constant__ static vid_t * pres; // pres for linear vertices (post neighbor of the reverse)
__constant__ static vid_t * posts; // posts for linear vertices
__constant__ static voff_t * fwd; // forward distance
__constant__ static voff_t * bwd; // backward distance
__constant__ static vid_t * fjid; // junction id for forward path
__constant__ static vid_t * bjid; // junction id for backward path
__constant__ static kmer_t * jkmers; // kmer values for junctions
__constant__ static edge_type * post_edges; // post edges of linear vertices
__constant__ static edge_type * pre_edges; // pre edges of linear vertices
__constant__ static size_t * ulens; // for each junction, store the length of the unitig starting from this end point
__constant__ static size_t * unitig_offsets; // record output offsets of unitigs for partitions in a processor
__constant__ static char * unitigs; // for each pair of junctions, output a unitig
__constant__ static ull * junct_edges; // for gathering contigs

__constant__ static voff_t * send_offsets; // used to locate the write position of messages for each partition in send buffer
__constant__ static voff_t * receive_offsets;
__constant__ static voff_t * extra_send_offsets;
__constant__ static voff_t * tmp_send_offsets;
__constant__ static int * id2index;

__constant__ static void * send;
__constant__ static void * receive;

__constant__ static ull * gpu_not_found;
__constant__ static char rev_table[4] = {'A', 'C', 'G', 'T'};

extern "C"
{
static void set_globals_graph_compute_gpu (meta_t * dm, master_t * mst)
{
	int num_of_devices = mst->num_of_devices;
#ifdef SINGLE_NODE
	int world_rank = mst->world_rank;
	int world_size = mst->world_size;
#endif
	int i;
	for (i = 0; i < num_of_devices; i++)
	{
#ifdef SINGLE_NODE
		cudaSetDevice(world_rank * num_of_devices + i);
#else
		cudaSetDevice(i + DEVICE_SHIFT);
#endif
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (posts, &(dm[i].edge.post), sizeof(vid_t *)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (pres, &dm[i].edge.pre, sizeof(vid_t *)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (fwd, &dm[i].edge.fwd, sizeof(voff_t *)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (bwd, &dm[i].edge.bwd, sizeof(voff_t *)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (fjid, &dm[i].edge.fjid, sizeof(vid_t *)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (bjid, &dm[i].edge.bjid, sizeof(vid_t *)));

		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (id2index, &dm[i].comm.id2index, sizeof(int *)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (receive_offsets, &dm[i].comm.receive_offsets, sizeof(voff_t *)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (send_offsets, &dm[i].comm.send_offsets, sizeof(voff_t *)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (tmp_send_offsets, &dm[i].comm.tmp_send_offsets, sizeof(voff_t*)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (send, &dm[i].comm.send, sizeof(void *)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (id_offsets, &dm[i].id_offsets, sizeof(vid_t *)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (jid_offset, &dm[i].jid_offset, sizeof(vid_t *)));
	}
}

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

static void set_receive_buffer_gpu (void ** recv, int did, int world_rank, int num_of_devices)
{
#ifdef SINGLE_NODE
	CUDA_CHECK_RETURN (cudaSetDevice(world_rank * num_of_devices + did));
#else
	CUDA_CHECK_RETURN (cudaSetDevice(did + DEVICE_SHIFT));
#endif
	CUDA_CHECK_RETURN (cudaMemcpyToSymbol(receive, recv, sizeof(void*)));
}

static void set_edges_gpu (meta_t * dm, master_t * mst)
{
	int num_of_devices = mst->num_of_devices;
#ifdef SINGLE_NODE
	int world_rank = mst->world_rank;
	int world_size = mst->world_size;
#endif
	int i;
	for (i = 0; i < num_of_devices; i++)
	{
#ifdef SINGLE_NODE
		cudaSetDevice(world_rank * num_of_devices + i);
#else
		cudaSetDevice(i + DEVICE_SHIFT);
#endif
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol(pre_edges, &dm[i].edge.pre_edges, sizeof(edge_type*)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol(post_edges, &dm[i].edge.post_edges, sizeof(edge_type*)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol(junct_edges, &dm[i].junct.edges, sizeof(ull*)));
	}
}

static void set_junctions_gpu (meta_t * dm, master_t * mst)
{
	int num_of_devices = mst->num_of_devices;
#ifdef SINGLE_NODE
	int world_rank = mst->world_rank;
	int world_size = mst->world_size;
#endif
	int i;
	for (i = 0; i < num_of_devices; i++)
	{
#ifdef SINGLE_NODE
		cudaSetDevice(world_rank * num_of_devices + i);
#else
		cudaSetDevice(i + DEVICE_SHIFT);
#endif
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol(jnb_offsets, &dm[i].junct.offs, sizeof(voff_t *)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol(jnbs, &dm[i].junct.nbs, sizeof(vid_t *)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol(jkmers, &dm[i].junct.kmers, sizeof(kmer_t *)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol(ulens, &dm[i].junct.ulens, sizeof(size_t *)));
	}
}

static void set_unitig_pointer_gpu (meta_t * dm, int did, master_t * mst)
{
#ifdef SINGLE_NODE
	int world_rank = mst->world_rank;
	int num_of_devices = mst->num_of_devices;
	cudaSetDevice(world_rank * num_of_devices + did);
#else
	cudaSetDevice(did + DEVICE_SHIFT);
#endif
	CUDA_CHECK_RETURN (cudaMemcpyToSymbol(unitigs, &dm->junct.unitigs, sizeof(char*)));
	CUDA_CHECK_RETURN (cudaMemcpyToSymbol(unitig_offsets, &dm->junct.unitig_offs, sizeof(size_t*)));
}

}
__device__ static int query_partition_id (vid_t id, int num_of_partitions)
{
	int begin = 0;
	int end = num_of_partitions;
	int index;
	while (begin <= end)
	{
		index = (begin + end) / 2;
		if (id < id_offsets[index])
		{
			if (id >= id_offsets[index - 1])
				return index - 1;
			else
			{
				end = index - 1;
			}
		}
		else
		{
			if (id < id_offsets[index + 1])
				return index;
			else
			{
				begin = index + 1;
			}
		}
	}
	return -1; // error: vertex id is out of range!!!

}

// ********* this is compute among junctions *************
__global__ static void push_mssg_offset_compact (uint size, int num_of_partitions, int curr_pid, voff_t jindex_offset, voff_t jnb_index_offset)
{
	int pindex = id2index[curr_pid];
	voff_t * local_joffs = jnb_offsets + jindex_offset + pindex;
	voff_t * local_nbs = jnbs + jnb_index_offset;

	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int r, w;
	w = (size + TOTAL_THREADS_NODES - 1) / TOTAL_THREADS_NODES;

	for (r=0; r<w; r++)
	{
		int index = gid + r * TOTAL_THREADS_NODES;
		if (index >= size)
			break;
		int pindex;
		int pid;
		int num = local_joffs[index+1] - local_joffs[index];
		int j;
		for (j=0; j<num; j++)
		{
			vid_t vertex = local_nbs[local_joffs[index] + j];
			pid = query_partition_id(vertex, num_of_partitions);
			if (vertex < id_offsets[pid] + jid_offset[pid]) // a junction neighbor
				continue;
			pindex = id2index[pid];
			atomicAdd(&send_offsets[pindex+1], 1);
		}
	}
}

// ********* this is compute among junctions *************
__global__ static void push_mssg_compact (uint size, int num_of_partitions, int curr_pid, voff_t jindex_offset, voff_t jnb_index_offset)
{
	query_t * buf = (query_t *)send;
	int pindex = id2index[curr_pid];
	voff_t * local_joffs = jnb_offsets + jindex_offset + pindex;
	voff_t * local_nbs = jnbs + jnb_index_offset;

	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int r, w;
	w = (size + TOTAL_THREADS_NODES - 1) / TOTAL_THREADS_NODES;

	for (r=0; r<w; r++)
	{
		int index = gid + r * TOTAL_THREADS_NODES;
		if (index >= size)
			break;
		int pindex;
		int pid;
		int num = local_joffs[index+1] - local_joffs[index];
		voff_t local_offset;
		voff_t off;
		int j;
		for (j=0; j<num; j++)
		{
			vid_t vertex = local_nbs[local_joffs[index] + j];
			pid = query_partition_id(vertex, num_of_partitions);
			if (vertex < id_offsets[pid] + jid_offset[pid]) // a junction neighbor
				continue;
			pindex = id2index[pid];
			local_offset = atomicAdd(&tmp_send_offsets[pindex+1], 1); // temporary send offsets indicate an exact index in the partition
			off = local_offset + send_offsets[pindex]; // send_offsets recorded in push_mssg_offsets determine the global offset of each partition
			buf[off].jid = id_offsets[curr_pid] + index;
			buf[off].nid = vertex;
		}
	}
}

// ********* this is compute among linear vertices *************
__global__ static void push_update_offset (uint num_mssgs, int pid, voff_t index_offset, int num_of_partitions, voff_t receive_start, bool intra_inter)
{
	query_t * buf;
	if (intra_inter)
	{
		int pindex = id2index[pid];
		buf = (query_t *)send + receive_start + send_offsets[pindex];
	}
	else
	{
		int pindex = id2index[pid];
		buf = (query_t *)send + receive_start + receive_offsets[pindex];
	}

	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int r, w;
	w = (num_mssgs + TOTAL_THREADS_NODES - 1) / TOTAL_THREADS_NODES;
	vid_t * local_fjid = fjid + index_offset;
	vid_t * local_bjid = bjid + index_offset;

	for (r=0; r<w; r++)
	{
		int index = gid + r * TOTAL_THREADS_NODES;
		if (index >= num_mssgs)
			break;
		query_t tmp = buf[index];
		vid_t vindex = tmp.nid - id_offsets[pid] - jid_offset[pid];
		if (local_fjid[vindex] != tmp.jid && local_bjid[vindex] != tmp.jid)
		{
			printf ("error!!!: cannot find destination!\n");
		}
		else
		{
			int jpid = query_partition_id(tmp.jid, num_of_partitions);
			int pindex = id2index[jpid];
			atomicAdd(&extra_send_offsets[pindex + 1], 1);
			// extra_send_offsets used in push-pull_and_push-pull mode, since send_offsets is occupied at this moment; it acts the same as send_offsets
		}
	}
}

// ********* this is compute among linear vertices *************
__global__ static void push_update (uint num_mssgs, int pid, voff_t index_offset, int num_of_partitions, voff_t receive_start, bool intra_inter)
{
	query_t * buf;
	compact_t * vs = (compact_t *)receive;
	if (intra_inter)
	{
		int pindex = id2index[pid];
		buf = (query_t *)send + receive_start + send_offsets[pindex];
	}
	else
	{
		int pindex = id2index[pid];
		buf = (query_t *)send + receive_start + receive_offsets[pindex];
	}

	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int r, w;
	w = (num_mssgs + TOTAL_THREADS_NODES - 1) / TOTAL_THREADS_NODES;

	vid_t * local_fjid = fjid + index_offset;
	vid_t * local_bjid = bjid + index_offset;
	voff_t * local_fwd = fwd + index_offset;
	voff_t * local_bwd = bwd + index_offset;

	for (r=0; r<w; r++)
	{
		int index = gid + r * TOTAL_THREADS_NODES;
		if (index >= num_mssgs)
			break;
		query_t tmp = buf[index];
		vid_t vindex = tmp.nid - id_offsets[pid] - jid_offset[pid];
		voff_t local_offset;
		if (local_fjid[vindex] == tmp.jid || local_bjid[vindex] == tmp.jid)
		{
			int jpid = query_partition_id(tmp.jid, num_of_partitions);
			int pindex = id2index[jpid];
			local_offset = atomicAdd(&tmp_send_offsets[pindex+1], 1);
			voff_t off = local_offset + extra_send_offsets[pindex];
			vs[off].nid = tmp.nid;
			vs[off].jid = tmp.jid;
			if (local_fjid[vindex] == tmp.jid)
			{
				vs[off].ojid = local_bjid[vindex];
#ifdef CONTIG_CHECK
				if (local_fwd[vindex] != 1)
					printf ("record length error!!!\n");
#endif
			}
			else
			{
				vs[off].ojid = local_fjid[vindex];
#ifdef CONTIG_CHECK
				if (local_bwd[vindex] != 1)
					printf ("record length error!!!\n");
#endif
			}
			vs[off].plen = local_fwd[vindex] + local_bwd[vindex];
		}
	}
}

// ********* this is compute among junctions *************
__global__ static void pull_update (uint num_mssgs, int pid, voff_t jindex_offset, voff_t jnb_index_offset, void * local_receive, bool intra_inter)
{
	compact_t * vs;
	int pindex = id2index[pid];
	voff_t * local_joffs = jnb_offsets + jindex_offset + pindex;
	voff_t * local_nbs = jnbs + jnb_index_offset;
	size_t * local_ulens = ulens + jnb_index_offset;

	if (intra_inter)
	{
		vs = (compact_t *)local_receive + extra_send_offsets[pindex];
	}
	else
	{
		vs = (compact_t *)local_receive + receive_offsets[pindex];
	}

	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int r, w;
	w = (num_mssgs + TOTAL_THREADS_NODES - 1) / TOTAL_THREADS_NODES;

	for (r=0; r<w; r++)
	{
		int index = gid + r * TOTAL_THREADS_NODES;
		if (index >= num_mssgs)
			break;

		vid_t nb = vs[index].nid;
		vid_t jindex = vs[index].jid - id_offsets[pid];
		int num = local_joffs[jindex+1] - local_joffs[jindex];
		int j;
		for (j=0; j<num; j++)
		{
			if(local_nbs[local_joffs[jindex] + j] == nb)
			{
				local_nbs[local_joffs[jindex] + j] = vs[index].ojid;
				local_ulens[local_joffs[jindex] + j] = vs[index].plen - 1;
				break;
			}
		}
#ifdef CONTIG_CHECK
		if (j==num) // not found junction!
		{
			printf ("Error!!!!!!! target junction not found!\n");
		}
#endif
	}
}

//**************** this is update with ulens with junctions *************
__global__ static void update_ulens_with_kplus1 (uint size, int pid, voff_t jindex_offset, voff_t jnb_index_offset, int k, int total_num_partitions)
{
	int pindex = id2index[pid];
	voff_t * local_joffs = jnb_offsets + jindex_offset + pindex;
	voff_t * local_nbs = jnbs + jnb_index_offset;
	size_t * local_ulens = ulens + jnb_index_offset;

	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int r, w;
	w = (size + TOTAL_THREADS_NODES - 1) / TOTAL_THREADS_NODES;

	for (r=0; r<w; r++)
	{
		int index = gid + r * TOTAL_THREADS_NODES;
		if (index >= size)
			break;
		voff_t nb_off = local_joffs[index];
		voff_t num_nbs = local_joffs[index+1] - local_joffs[index];
		vid_t jid = id_offsets[pid] + index;
		voff_t j;
		for (j=0; j<num_nbs; j++)
		{
			vid_t nbid = local_nbs[nb_off + j];
			if (nbid == jid)
			{
				if (local_ulens[nb_off + j] > 0)
				{
					local_ulens[nb_off + j] += k+1;
					continue;
				}
			}
			int nb_pid = query_partition_id (nbid, total_num_partitions);
			if (nbid < id_offsets[nb_pid] + jid_offset[nb_pid] && nbid < jid)
			{
				local_ulens[nb_off + j] = 0;
			}
			else
			{
				local_ulens[nb_off + j] += k+1;
			}

		}
	}
}

// ********* this is compute among linear vertices *************
__global__ static void push_mssg_offset_contig (uint size, int pid, voff_t index_offset, int total_num_partitions)
{
	vid_t * local_fjid = fjid + index_offset;
	vid_t * local_bjid = bjid + index_offset;

	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int r, w;
	w = (size + TOTAL_THREADS_NODES - 1) / TOTAL_THREADS_NODES;

	for (r=0; r<w; r++)
	{
		int index = gid + r * TOTAL_THREADS_NODES;
		if (index >= size)
			break;
		int fpid = query_partition_id (local_fjid[index], total_num_partitions);
		int bpid = query_partition_id (local_bjid[index], total_num_partitions);
		int jpid;
		if (local_fjid[index] >= id_offsets[fpid] + jid_offset[fpid] && local_bjid[index] >= id_offsets[bpid] + jid_offset[bpid])
		{
			printf ("GPU: NONE JUNCTION END POINT FOUND:: pid=%d, index=%u, fid=%u, bid=%u!\n", pid, index, local_fjid[index], local_bjid[index]);
			continue;
		}
		else if (local_fjid[index] < id_offsets[fpid] + jid_offset[fpid] && local_bjid[index] < id_offsets[bpid] + jid_offset[bpid])
		{
			jpid = local_fjid[index] <= local_bjid[index] ? fpid : bpid;
		}
		else
		{
			jpid = local_fjid[index] < id_offsets[fpid] + jid_offset[fpid]? fpid : bpid;
		}

		int pindex = id2index[jpid];
		atomicAdd (&send_offsets[pindex+1], 1);
	}
}

// ********* this is compute among linear vertices *************
__global__ static void push_mssg_contig (uint size, int pid, voff_t index_offset, int total_num_partitions)
{
	vid_t * local_fjid = fjid + index_offset;
	vid_t * local_bjid = bjid + index_offset;
	voff_t * local_fwd = fwd + index_offset;
	voff_t * local_bwd = bwd + index_offset;
	edge_type * local_post_edge = post_edges + index_offset;
	edge_type * local_pre_edge = pre_edges + index_offset;
	unitig_t * buf = (unitig_t *)send;

	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int r, w;
	w = (size + TOTAL_THREADS_NODES - 1) / TOTAL_THREADS_NODES;

	for (r=0; r<w; r++)
	{
		int index = gid + r * TOTAL_THREADS_NODES;
		if (index >= size)
			break;
		int fpid = query_partition_id (local_fjid[index], total_num_partitions);
		int bpid = query_partition_id (local_bjid[index], total_num_partitions);
		vid_t jid;
		int jpid;
		if (local_fjid[index] >= id_offsets[fpid] + jid_offset[fpid] && local_bjid[index] >= id_offsets[bpid] + jid_offset[bpid])
			continue;
		else if (local_fjid[index] < id_offsets[fpid] + jid_offset[fpid] && local_bjid[index] < id_offsets[bpid] + jid_offset[bpid])
		{
			jid = local_fjid[index] <= local_bjid[index] ? local_fjid[index] : local_bjid[index];
			jpid = local_fjid[index] <= local_bjid[index] ? fpid : bpid;
		}
		else
		{
			jid = local_fjid[index] < id_offsets[fpid] + jid_offset[fpid]? local_fjid[index] : local_bjid[index];
			jpid = local_fjid[index] < id_offsets[fpid] + jid_offset[fpid]? fpid: bpid;
		}

		int pindex = id2index[jpid];
		voff_t local_offset = atomicAdd(&tmp_send_offsets[pindex+1], 1); // temporary send offsets indicate an exact index in the partition
		voff_t off = local_offset + send_offsets[pindex]; // send_offsets recorded in push_mssg_offsets determine the global offset of each partition
		if (jid == local_fjid[index])
		{
			buf[off].jid = local_fjid[index];
			buf[off].ojid = local_bjid[index];
			buf[off].rank = local_fwd[index] | (local_pre_edge[index] << (sizeof(voff_t)*CHAR_BITS) - 2);
		}
		else if (jid == local_bjid[index])
		{
			buf[off].jid = local_bjid[index];
			buf[off].ojid = local_fjid[index];
			buf[off].rank = local_bwd[index] | (local_post_edge[index] << (sizeof(voff_t)*CHAR_BITS) - 2);
		}
		else
		{
			printf ("Please find the error!!!!!!!!!!!\n");
		}
		buf[off].len = local_fwd[index] + local_bwd[index];
	}
}

// ********* this is compute among junctions *************
__global__ static void pull_mssg_contig (uint num_mssgs, int pid, voff_t jindex_offset, voff_t jnb_index_offset, voff_t receive_start, bool intra_inter, int k)
{
	int pindex = id2index[pid];
	voff_t * local_joffs = jnb_offsets + jindex_offset + pindex;
	voff_t * local_nbs = jnbs + jnb_index_offset;
	size_t * local_ulens = ulens + jnb_index_offset;
	char * local_unitigs = unitigs + unitig_offsets[pindex];

	unitig_t * buf;
	if (intra_inter)
		buf = (unitig_t *)send + receive_start + send_offsets[pindex];
	else buf = (unitig_t *)send + receive_start + receive_offsets[pindex];

	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int r, w;
	w = (num_mssgs + TOTAL_THREADS_NODES - 1) / TOTAL_THREADS_NODES;

	for (r=0; r<w; r++)
	{
		int index = gid + r * TOTAL_THREADS_NODES;
		if (index >= num_mssgs)
			break;
		vid_t jindex = buf[index].jid - id_offsets[pid];
		if (jindex > jid_offset[pid])
		{
			printf ("EEEEEEerror on GPU !!!!!!\n");
		}
		vid_t nb = buf[index].ojid;
		voff_t rank = buf[index].rank & 0x3fffffff;
		char edge = rev_table[buf[index].rank >> (sizeof(voff_t) * CHAR_BITS - 2)];
		voff_t len = buf[index].len;
		int num = local_joffs[jindex+1] - local_joffs[jindex];
		int j;
		for (j=0; j<num; j++)
		{
			voff_t nb_off = local_joffs[jindex] + j;
			size_t unitig_off = nb_off == 0 ? 0 : local_ulens[nb_off-1]; // local_ulens[0] >= 0
			size_t length = nb_off == 0 ? local_ulens[0] : (local_ulens[nb_off] - local_ulens[nb_off-1]);
			if(local_nbs[nb_off] == nb && length-k == len)
			{
				if (buf[index].jid <= nb && length-k-1 >= rank) // be careful: there may be parallel edges for the same (jid, ojid), handle it later
					local_unitigs[unitig_off + k + rank] = edge;
				break;
			}
		}
#ifdef CONTIG_CHECK
		if (j==num) // not found junction!
		{
			printf ("Error!!!!!!! target junction not found!\n");
		}
#endif
	}
}

__device__ static void get_kmer_string (kmer_t * kmer, int k, char * str)
{
	unit_kmer_t * ptr = (unit_kmer_t *)kmer;
	int i;
	int j = 0;
	int num = 0;
	int kmer_unit_bits = sizeof(unit_kmer_t) * CHAR_BITS;
	for (i=0; i<k; i++)
	{
		str[i] = rev_table[((*ptr) >> (kmer_unit_bits - i*2 - num*kmer_unit_bits - 2)) & 0x3];
		j += 2;
		if (j / kmer_unit_bits == 1)
		{
			ptr++;
			num++;
			j -= kmer_unit_bits;
		}
	}

}
__device__ static int complete_first_2bps (kmer_t * kmer, int k, ull * edge, char * unitig, int nb_num, int cutoff)
{
	int i;
	int num_nb=0;
	for (i=0; i<EDGE_DIC_SIZE/2; i++)
	{
		if ((((*edge) >> (i*8)) & 0xff) >= cutoff)
		{
			if (num_nb++ == nb_num) // a post edge corresponds to the neighbor
			{
				get_kmer_string (kmer, k, unitig);
				unitig[k] = rev_table[i];
				return 1;
			}
		}
	}
	for (i=EDGE_DIC_SIZE/2; i<EDGE_DIC_SIZE; i++)
	{
		if ((((*edge) >> (i*8)) & 0xff) >= cutoff)
		{
			if (num_nb++ == nb_num)
			{
				kmer_t reverse;
				get_reverse_kmer (kmer, &reverse, k);
				get_kmer_string (&reverse, k, unitig);
				unitig[k] = rev_table[i-EDGE_DIC_SIZE/2];
				return 1;
			}
		}
	}
	return 0;
}

__global__ static void complete_contig_with_junction_gpu (uint size, int pid, voff_t jindex_offset, voff_t jnb_index_offset, int k, int cutoff)
{
	int pindex = id2index[pid];
	voff_t * local_joffs = jnb_offsets + jindex_offset + pindex;
	ull * local_edges = junct_edges + jindex_offset;
	kmer_t * local_kmer = jkmers + jindex_offset;
	voff_t * local_nbs = jnbs + jnb_index_offset;
	size_t * local_ulens = ulens + jnb_index_offset;
	char * local_unitigs = unitigs + unitig_offsets[pindex];

	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int r, w;
	w = (size + TOTAL_THREADS_NODES - 1) / TOTAL_THREADS_NODES;

	for (r=0; r<w; r++)
	{
		int index = gid + r * TOTAL_THREADS_NODES;
		if (index >= size)
			break;

		voff_t num = local_joffs[index+1] - local_joffs[index];
		vid_t jid = id_offsets[pid] + index;
		int j;
		for (j=0; j<num; j++)
		{
			voff_t nb_off = local_joffs[index] + j;
			vid_t ojid = local_nbs[nb_off];
			size_t unitig_off = nb_off<1 ? 0 : local_ulens[nb_off-1];
			if (jid <= ojid)
			{
				if (jid == ojid)
				{
					size_t length = nb_off==0? local_ulens[nb_off] : (local_ulens[nb_off] - local_ulens[nb_off-1]);
					if (length == 0)
						continue;
				}
				if (complete_first_2bps (local_kmer+index, k, local_edges+index, local_unitigs+unitig_off, j, cutoff) == 0)
				{
#ifdef CONTIG_CHECK
					printf ("complete contig with junction kmer error!\n");
#endif
				}
			}
		}
	}

}

#endif /* CONTIG_CUH_ */
