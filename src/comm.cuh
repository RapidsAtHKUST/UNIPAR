/*
 * comm.cuh
 *
 *  Created on: 2017-7-14
 *      Author: qiushuang
 */


#ifndef COMM_CUH_
#define COMM_CUH_

#include "../include/dbgraph.h"
#include "../include/comm.h"

#define MAX_NUM_BLOCKS 1024 // for performance, setting it to be 1024; the maximum number of blocks in GPU can be at least 4096
#define THREADS_PER_BLOCK_NODES 512 // no greater than 512, otherwise resources for gpu threads (probably the register size) is not enough for to launch the kernel!!!
#define TOTAL_THREADS_NODES (THREADS_PER_BLOCK_NODES * MAX_NUM_BLOCKS)
#define THREADS_PER_BLOCK_NODES_TEST THREADS_PER_BLOCK_NODES
#define MAX_NUM_BLOCKS_TEST MAX_NUM_BLOCKS
#define TOTAL_THREADS_NODES_TEST (THREADS_PER_BLOCK_NODES_TEST * MAX_NUM_BLOCKS_TEST)


__constant__ static goffset_t * id_offsets; // used to assign global id for each node in subgraphs
__constant__ static goffset_t * jid_offset; // junction vertex id offsets, used to calculate id of each vertex from its index

__constant__ static uint * spid_nbs; // partition id shared by posts and pres
__constant__ static vid_t * pres; // pres for linear vertices
__constant__ static vid_t * posts; // posts for linear vertices
__constant__ static voff_t * fwd; // forward distance
__constant__ static voff_t * bwd; // backward distance
__constant__ static uint * spid_js; // partition id shared by fjid and bjid
__constant__ static vid_t * fjid; // junction id for forward path
__constant__ static vid_t * bjid; // junction id for backward path
__constant__ static voff_t * mssg_offset; // mssg writing offset buffer, not needed if we do not store the message offsets in push_mssg_offset

__constant__ static int * id2index; // partition id to the index of partition list

__constant__ static voff_t * send_offsets; // used to locate the write position of messages for each partition in send buffer
__constant__ static voff_t * receive_offsets;
__constant__ static voff_t * extra_send_offsets; // used when pull message and push new messages happens simultaneously.
__constant__ static voff_t * tmp_send_offsets;

__constant__ static void * send;
__constant__ static void * receive;

__constant__ static kmer_t * lkmers; // kmer values of linear vertices

extern __shared__ voff_t sidoff[]; //shared memory to store id_offsets for fast query partition id; in size of (number of partitions + 1)
__constant__ ull * gpu_not_found;

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
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (spid_nbs, &dm[i].edge.spid_nb, sizeof(uint *)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (spid_js, &dm[i].edge.spid_js, sizeof(uint *)));

		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (id2index, &dm[i].comm.id2index, sizeof(int *)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (receive_offsets, &dm[i].comm.receive_offsets, sizeof(voff_t *)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (send_offsets, &dm[i].comm.send_offsets, sizeof(voff_t *)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (tmp_send_offsets, &dm[i].comm.tmp_send_offsets, sizeof(voff_t*)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (send, &dm[i].comm.send, sizeof(void *)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (id_offsets, &dm[i].id_offsets, sizeof(goffset_t *)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol (jid_offset, &dm[i].jid_offset, sizeof(goffset_t *)));
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

__device__ static int query_partition_id_shrd (vid_t id, int num_of_partitions)
{
	int begin = 0;
	int end = num_of_partitions;
	int index;
	while (begin <= end)
	{
		index = (begin + end) / 2;
		if (id < sidoff[index])
		{
			if (id >= sidoff[index - 1])
				return index - 1;
			else
			{
				end = index - 1;
			}
		}
		else
		{
			if (id < sidoff[index + 1])
				return index;
			else
			{
				begin = index + 1;
			}
		}
	}
	return -1; // error: vertex id is out of range!!!

}

static void __device__ update_pid_for_post (uint pid, uint *spid)
{
	atomicAnd (spid, 0xffff0000);
	atomicOr (spid, pid);
}

static void __device__ update_pid_for_pre (uint pid, uint *spid)
{
	atomicAnd (spid, 0x0000ffff);
	atomicOr (spid, pid<<SPID_BITS);
}


__global__ static void init_lr (uint size, int num_of_partitions, int cur_id, voff_t index_offset)
{
	vid_t * local_post = posts + index_offset;
	vid_t * local_pre = pres + index_offset;
	voff_t * local_fwd = fwd + index_offset;
	voff_t * local_bwd = bwd + index_offset;
	vid_t * local_fjid = fjid + index_offset;
	vid_t * local_bjid = bjid + index_offset;
	uint * local_spid_nbs = spid_nbs + index_offset;
	uint * local_spid_js = spid_js + index_offset;

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

		vid_t self_id;
		get_selfid(self_id, index, jid_offset[cur_id], id_offsets[cur_id]);
		int preid;
		int postid;
		get_pid_for_pre(preid, local_spid_nbs[index]);
		get_pid_for_post(postid, local_spid_nbs[index]);
		if (local_pre[index] == local_post[index] && preid == postid)
		{
			local_bjid[index] = self_id;
			update_pid_for_pre(preid, &local_spid_js[index]);
			local_pre[index] = MAX_VID;
//			int pid = query_partition_id(local_post[index], num_of_partitions);
			if (preid<0 || preid>=num_of_partitions)
			{
				printf ("query id error!\n");
				// exit(0);
			}
			if (is_junction(local_post[index], jid_offset[postid], id_offsets[postid])) // post is a junction
			{
				local_fjid[index] = local_post[index];
				update_pid_for_post(postid, &local_spid_js[index]);
				local_post[index] = MAX_VID;
			}
		}
		else
		{
			if (local_pre[index] == self_id && preid == cur_id)
			{
				local_bjid[index] = self_id;
				update_pid_for_pre(preid, &local_spid_js[index]);
				local_pre[index] = MAX_VID;
			}
			else
			{
//				int pid = query_partition_id(local_pre[index], num_of_partitions);
				if (preid<0 || preid>=num_of_partitions)
				{
					printf ("query id error!\n");
				}
				if (is_junction(local_pre[index], jid_offset[preid], id_offsets[preid])) // pre is a junction
				{
					local_bjid[index] = local_pre[index];
					update_pid_for_pre(preid, &local_spid_js[index]);
					local_pre[index] = MAX_VID;
				}
			}
			if (local_post[index] == self_id && postid == cur_id)
			{
				local_fjid[index] = self_id;
				update_pid_for_post(postid, &local_spid_js[index]);
				local_post[index] = MAX_VID;
			}
			else
			{
//				int pid = query_partition_id(local_post[index], num_of_partitions);
				if (postid<0 || postid>=num_of_partitions)
				{
					printf ("query id error!\n");
				}
				if (is_junction(local_post[index], jid_offset[postid], id_offsets[postid])) // post is a junction
				{
					local_fjid[index] = local_post[index];
					update_pid_for_post(postid, &local_spid_js[index]);
					local_post[index] = MAX_VID;
				}
			}
		}

		local_fwd[index] = 1;
		local_bwd[index] = 1;
	}
}


__global__ static void push_mssg_offset_lr (uint size, int num_of_partitions, int curr_pid, voff_t index_offset)
{
	vid_t * local_post = posts + index_offset;
	vid_t * local_pre = pres + index_offset;
	uint * local_spid_nbs = spid_nbs + index_offset;

	const int gid = blockIdx.x*blockDim.x+threadIdx.x;
//	const int tid = threadIdx.x;
	int r,w;
/*	w = (num_of_partitions + THREADS_PER_BLOCK_NODES) / THREADS_PER_BLOCK_NODES;
	for (r=0; r<w; r++)
	{
		if (tid + r * THREADS_PER_BLOCK_NODES > num_of_partitions)
			break;
		sidoff[tid + r*THREADS_PER_BLOCK_NODES] = id_offsets[tid + r*THREADS_PER_BLOCK_NODES];
	}
	__syncthreads();*/

	w = (size + TOTAL_THREADS_NODES - 1) / TOTAL_THREADS_NODES;
	int index;
	for(r = 0; r < w; r++)
	{
		index = gid + r * TOTAL_THREADS_NODES;
		if(index >= size)
			break;
		int pindex;
//		int pid;
		vid_t self_id;
		get_selfid(self_id, index, jid_offset[curr_pid], id_offsets[curr_pid]);
		int postid;
		int preid;
		get_pid_for_post(postid, local_spid_nbs[index]);
		get_pid_for_pre(preid, local_spid_nbs[index]);
		if(local_post[index] != MAX_VID && !(local_post[index] == self_id && postid == curr_pid))
		{
//			pid = query_partition_id_shrd(local_post[index], num_of_partitions);
			pindex = id2index[postid];
			atomicAdd(&send_offsets[pindex+1], 1);
		}
		if(local_pre[index] != MAX_VID && !(local_pre[index] == self_id && preid == curr_pid))
		{
//			pid = query_partition_id_shrd(local_pre[index], num_of_partitions);
			pindex = id2index[preid];
			atomicAdd(&send_offsets[pindex+1], 1);
		}
	}
}

static void (*push_mssg_offset_lr_async) (uint, int, int, voff_t) = push_mssg_offset_lr;

__global__ static void push_mssg_lr (uint size, int num_of_partitions, int curr_pid, voff_t index_offset)
{
	path_t * buf = (path_t *)send;
	vid_t * local_post = posts + index_offset;
	vid_t * local_pre = pres + index_offset;
	voff_t * local_fwd = fwd + index_offset;
	voff_t * local_bwd = bwd + index_offset;
	vid_t * local_fjid = fjid + index_offset;
	vid_t * local_bjid = bjid + index_offset;
	uint * local_spid_nbs = spid_nbs + index_offset;
	uint * local_spid_js = spid_js + index_offset;

	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
//	const int tid = threadIdx.x;
	int r, w;
/*	w = (num_of_partitions + THREADS_PER_BLOCK_NODES) / THREADS_PER_BLOCK_NODES;
	for (r=0; r<w; r++)
	{
		if (tid + r * THREADS_PER_BLOCK_NODES > num_of_partitions)
			break;
		sidoff[tid + r*THREADS_PER_BLOCK_NODES] = id_offsets[tid + r*THREADS_PER_BLOCK_NODES];
	}
	__syncthreads();*/

	w = (size + TOTAL_THREADS_NODES - 1) / TOTAL_THREADS_NODES;
	int index;
	for (r=0; r<w; r++)
	{
		index = gid + r * TOTAL_THREADS_NODES;
		if (index >= size)
			break;
		int pindex;
//		int pid;
		voff_t off;
		path_t tmp;
		voff_t local_offset;
		vid_t self_id;
		get_selfid(self_id, index, jid_offset[curr_pid], id_offsets[curr_pid]);
		int postid;
		int preid;
		get_pid_for_post(postid, local_spid_nbs[index]);
		get_pid_for_pre(preid, local_spid_nbs[index]);
		if(local_post[index] != MAX_VID && !(local_post[index] == self_id && postid == curr_pid))
		{
//			pid = query_partition_id_shrd(local_post[index], num_of_partitions);
			pindex = id2index[postid];
			local_offset = atomicAdd(&tmp_send_offsets[pindex+1], 1);
			off = local_offset + send_offsets[pindex];
			tmp.dst = local_post[index] - jid_offset[postid];
			tmp.dist = local_bwd[index];
			tmp.opps = local_pre[index];
			int jpid = 0;
			if (tmp.opps == MAX_VID)
			{
				tmp.jid = local_bjid[index];
				get_pid_for_pre(jpid, local_spid_js[index]);
			}
			tmp.cid = self_id;
			encode_path_pids(tmp.spids, postid, preid, jpid, curr_pid);
			buf[off] = tmp;
		}

		if(local_pre[index] != MAX_VID && !(local_pre[index] == self_id && preid == curr_pid))
		{
//			pid = query_partition_id_shrd(local_pre[index], num_of_partitions);
			pindex = id2index[preid];
			local_offset = atomicAdd(&tmp_send_offsets[pindex+1], 1);
			off = local_offset + send_offsets[pindex];
			tmp.dst = local_pre[index] - jid_offset[preid];
			tmp.dist = local_fwd[index];
			tmp.opps = local_post[index];
			int jpid = 0;
			if (tmp.opps == MAX_VID)
			{
				tmp.jid = local_fjid[index];
				get_pid_for_post(jpid, local_spid_js[index]);
			}
			tmp.cid = self_id;
			encode_path_pids(tmp.spids, preid, postid, jpid, curr_pid);
			buf[off] = tmp;
		}
	}
}

static void (*push_mssg_lr_async) (uint, int, int, voff_t) = push_mssg_lr;

__global__ static void pull_mssg_lr (uint num_mssgs, int pid, voff_t index_offset, voff_t receive_start, bool intra_inter)
{
	path_t * buf;
	int pindex = id2index[pid];
	if (intra_inter)
		buf = (path_t *)send + receive_start + send_offsets[pindex];
	else buf = (path_t *)send + receive_start + receive_offsets[pindex];

	vid_t * local_post = posts + index_offset;
	vid_t * local_pre = pres + index_offset;
	voff_t * local_fwd = fwd + index_offset;
	voff_t * local_bwd = bwd + index_offset;
	vid_t * local_fjid = fjid + index_offset;
	vid_t * local_bjid = bjid + index_offset;
	uint * local_spid_nbs = spid_nbs + index_offset;
	uint * local_spid_js = spid_js + index_offset;

	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int r, w;
	w = (num_mssgs + TOTAL_THREADS_NODES - 1) / TOTAL_THREADS_NODES;
	int index;
	for (r=0; r<w; r++)
	{
		index = gid + r * TOTAL_THREADS_NODES;
		if (index >= num_mssgs)
			break;
		path_t tmp = buf[index];
		vid_t vindex = tmp.dst;
		int postid;
		int preid;
		get_pid_for_post(postid, local_spid_nbs[vindex]);
		get_pid_for_pre(preid, local_spid_nbs[vindex]);
		int fjpid;
		int bjpid;
		get_pid_for_post(fjpid, local_spid_js[vindex]);
		get_pid_for_pre(bjpid, local_spid_js[vindex]);
		int oppsid;
		int dstid;
		int jpid;
		int cpid;
		decode_path_pids(tmp.spids, dstid, oppsid, jpid, cpid);
		if (dstid != pid)
		{
			printf ("error in encoding dstid!\n");
		}
		if ((tmp.opps == local_fjid[vindex] && oppsid == fjpid) || (tmp.opps == local_bjid[vindex] && oppsid == bjpid)) // next neighbor is the same as previous, a circle???
		{
			local_post[vindex] = MAX_VID;
			local_pre[vindex] = MAX_VID;
			local_fjid[vindex] = tmp.cid;
			update_pid_for_post(cpid, &local_spid_js[vindex]);
			local_bjid[vindex] = tmp.cid;
			update_pid_for_pre(cpid, &local_spid_js[vindex]);
			local_fwd[vindex] += tmp.dist;
			local_bwd[vindex] = 0;
		}
		else if(local_post[vindex] == tmp.cid && postid == cpid)
		{
			if (tmp.opps == MAX_VID)
			{
				local_fjid[vindex] = tmp.jid;
				update_pid_for_post(jpid, &local_spid_js[vindex]);
			}
			else
			{
				local_fjid[vindex] = local_post[vindex];
				update_pid_for_post(postid, &local_spid_js[vindex]);
			}
			local_post[vindex] = tmp.opps;
			update_pid_for_post(oppsid, &local_spid_nbs[vindex]);
			local_fwd[vindex] += tmp.dist;
		}
		else if(local_pre[vindex] == tmp.cid && preid == cpid)
		{
			if (tmp.opps == MAX_VID)
			{
				local_bjid[vindex] = tmp.jid;
				update_pid_for_pre(jpid, &local_spid_js[vindex]);
			}
			else
			{
				local_bjid[vindex] = local_pre[vindex];
				update_pid_for_pre(preid, &local_spid_js[vindex]);
			}
			local_pre[vindex] = tmp.opps;
			update_pid_for_pre(oppsid, &local_spid_nbs[vindex]);
			local_bwd[vindex] += tmp.dist;
		}
		else
		{
			printf ("GPU::::::::: partition %d: error!!!: %d : cannot find destination!\n", pid, (int)intra_inter);
		}
	}
}

static void (*pull_mssg_lr_async) (uint, int, voff_t, voff_t, bool) = pull_mssg_lr;

__global__ static void push_selfloop_offset (uint num_mssgs, int pid, voff_t index_offset, int num_of_partitions, voff_t receive_start, bool intra_inter)
{
	path_t * buf;
	voff_t * local_offsets = extra_send_offsets;
	if (intra_inter)
	{
		int pindex = id2index[pid];
		buf = (path_t *)send + receive_start + send_offsets[pindex];
	}
	else
	{
		int pindex = id2index[pid];
		buf = (path_t *)send + receive_start + receive_offsets[pindex];
	}

	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int r, w;
	w = (num_mssgs + TOTAL_THREADS_NODES - 1) / TOTAL_THREADS_NODES;
	int index;
	vid_t * local_post = posts + index_offset;
	vid_t * local_pre = pres + index_offset;

	for (r=0; r<w; r++)
	{
		index = gid + r * TOTAL_THREADS_NODES;
		if (index >= num_mssgs)
			break;
		path_t tmp = buf[index];
		vid_t vindex = tmp.dst;
		if (local_post[vindex] != tmp.cid && local_pre[vindex] != tmp.cid)
		{
			int cpid = query_partition_id(tmp.cid, num_of_partitions);
			int pindex = id2index[cpid];
			atomicAdd(&local_offsets[pindex + 1], 1);
		}
	}
}


__global__ static void push_selfloop (uint num_mssgs, int pid, voff_t index_offset, int num_of_partitions, voff_t receive_start, bool intra_inter)
{
	path_t * buf;
	selfloop_t * vs = (selfloop_t *)receive;
	voff_t * local_offsets = extra_send_offsets;
	if (intra_inter)
	{
		int pindex = id2index[pid];
		buf = (path_t *)send + receive_start + send_offsets[pindex];
	}
	else
	{
		int pindex = id2index[pid];
		buf = (path_t *)send + receive_start + receive_offsets[pindex];
	}

	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int r, w;
	w = (num_mssgs + TOTAL_THREADS_NODES - 1) / TOTAL_THREADS_NODES;
	int index;

	vid_t * local_post = posts + index_offset;
	vid_t * local_pre = pres + index_offset;

	for (r=0; r<w; r++)
	{
		index = gid + r * TOTAL_THREADS_NODES;
		if (index >= num_mssgs)
			break;
		path_t tmp = buf[index];
		vid_t vindex = tmp.dst;
		voff_t local_offset;
		if (local_post[vindex] != tmp.cid && local_pre[vindex] != tmp.cid)
		{
			int cpid = query_partition_id(tmp.cid, num_of_partitions);
			int pindex = id2index[cpid];
			local_offset = atomicAdd(&tmp_send_offsets[pindex+1], 1);
			vs[local_offset + local_offsets[pindex]].v = tmp.cid - jid_offset[cpid] - id_offsets[cpid];
			vs[local_offset + local_offsets[pindex]].dst = tmp.dst + jid_offset[pid] + id_offsets[pid];
//			vs[local_mssg_offset[index] + local_offsets[pindex]].v = tmp.cid - jid_offset[cpid] - id_offsets[cpid];
//			vs[local_mssg_offset[index] + local_offsets[pindex]].dst = tmp.dst + jid_offset[pid] + id_offsets[pid];
		}
	}
}

__global__ static void pull_selfloop (uint num_mssgs, int pid, voff_t index_offset, selfloop_t * local_receive, bool intra_inter)
{
	selfloop_t * vs;
	vid_t * local_post = posts + index_offset;
	vid_t * local_pre = pres + index_offset;
	vid_t * local_fjid = fjid + index_offset;
	vid_t * local_bjid = bjid + index_offset;

	if (intra_inter)
	{
		int pindex = id2index[pid];
		vs = (selfloop_t *)local_receive + extra_send_offsets[pindex];
	}
	else
	{
		int pindex = id2index[pid];
		vs = (selfloop_t *)local_receive + receive_offsets[pindex];
	}

	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int r, w;
	w = (num_mssgs + TOTAL_THREADS_NODES - 1) / TOTAL_THREADS_NODES;
	int index;

	for (r=0; r <w; r++)
	{
		index = gid + r * TOTAL_THREADS_NODES;
		if (index >= num_mssgs)
			break;
		int i = vs[index].v;
		if (local_post[i] == vs[index].dst)
		{
			atomicMax(&local_post[i], MAX_VID);
			local_fjid[i] = i + jid_offset[pid] + id_offsets[pid]; // Meanwhile, local_fjid[i] = self_id to denote a selfloop
		}
		else if (local_pre[i] == vs[index].dst)
		{
			atomicMax(&local_pre[i], MAX_VID);
			local_bjid[i] = i + jid_offset[pid] + id_offsets[pid]; // Meanwhile, local_bjid[i] = self_id to denote a selfloop
		}
		else
		{
			if (local_pre[i] != MAX_VID && local_post[i] != MAX_VID)
			{
				printf ("GPUGPUGPUGPU:: ERROR:: %d:: selfloop not made!\n", int(intra_inter));
//				atomicAdd (gpu_not_found, 1);
			}
		}
	}
}


#endif /* COMM_CUH_ */
