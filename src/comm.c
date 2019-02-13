/*
 * comm.c
 *
 *  Created on: 2017-7-14
 *      Author: qiushuang
 */

/* This file implement the primitives for different processors to communicate with each other */

#include <omp.h>
#include "../include/dbgraph.h"
#include "../include/comm.h"
#include "../include/malloc.h"
#include "../include/share.h"
#include "../include/hash.h"

uint dst_not_found = 0;
uint selfloop_made = 0;

#define DEPTH 11
#define INIT_CPU_THREADS 24
#define CPU_THREADS 24
#define PULL_CPU_THREADS 24

extern long cpu_threads;
extern float push_offset_time[NUM_OF_PROCS];
extern float push_time[NUM_OF_PROCS];
extern float pull_intra_time[NUM_OF_PROCS];
extern float pull_inter_time[NUM_OF_PROCS];
float over_time[NUM_OF_PROCS];

extern int lock_flag[NUM_OF_PROCS];

static goffset_t * id_offsets; // used to assign global id for each node in subgraphs
static goffset_t * jid_offset; // junction vertex id offsets, used to calculate id of each vertex from its index

static uint * spid_nbs; // partition id shared by posts and pres
static vid_t * pres; // pres for linear vertices
static vid_t * posts; // posts for linear vertices
static voff_t * fwd; // forward distance
static voff_t * bwd; // backward distance
static uint * spid_js; // partition id shared by fjid and bjid
static vid_t * fjid; // junction id for forward path
static vid_t * bjid; // junction id for backward path
static voff_t * mssg_offset; // mssg writing offset buffer, not needed if we do not store the message offsets in push_mssg_offset

static voff_t * send_offsets; // used to locate the write position of messages for each partition in send buffer
static voff_t * receive_offsets;
static voff_t * extra_send_offsets;
static int * id2index;
static voff_t * send_offsets_th;
static voff_t * tmp_send_offsets_th;

static void * send;
static void * receive;

static voff_t * noffs; // offset array for the neighbors of junctions
static vid_t * nids; // neighbor ids of junctions
static kmer_t * jkmers; // kmer values of junctions

extern int debug;

static void update_pid_for_post (uint pid, uint *spid)
{
	atomic_and_int (spid, 0xffff0000);
	atomic_or_int (spid, pid);
}

static void update_pid_for_pre (uint pid, uint *spid)
{
	atomic_and_int (spid, 0x0000ffff);
	atomic_or_int (spid, pid<<SPID_BITS);
}

static void set_globals_cpu (meta_t * dm, int num_of_partitions)
{
	pres = dm->edge.pre;
	posts = dm->edge.post;
	fwd = dm->edge.fwd;
	bwd = dm->edge.bwd;
	fjid = dm->edge.fjid;
	bjid = dm->edge.bjid;
	spid_nbs = dm->edge.spid_nb;
	spid_js = dm->edge.spid_js;

	send_offsets = dm->comm.send_offsets;
	receive_offsets = dm->comm.receive_offsets;
	send = dm->comm.send;
//	receive = dm->comm.receive;
	id2index = dm->comm.id2index;
	id_offsets = dm->id_offsets;
	jid_offset = dm->jid_offset;

	send_offsets_th = (voff_t*) malloc (sizeof(voff_t) * (num_of_partitions+1) * (cpu_threads+1));
	CHECK_PTR_RETURN (send_offsets_th, "malloc local send offsets for multi-threads in push mssg offset lr error!\n");
	memset (send_offsets_th, 0, sizeof(voff_t) * (num_of_partitions+1) * (cpu_threads+1));
	tmp_send_offsets_th = (voff_t*) malloc (sizeof(voff_t) * (num_of_partitions+1) * (cpu_threads+1));
	CHECK_PTR_RETURN (send_offsets_th, "malloc tmp send offsets for multi-threads in push mssg offset lr error!\n");
	memset (tmp_send_offsets_th, 0, sizeof(voff_t) * (num_of_partitions+1) * (cpu_threads+1));
}

static void release_globals_cpu (void)
{
	free(send_offsets_th);
	free(tmp_send_offsets_th);
}

static void set_mssg_offset_buffer (meta_t * dm)
{
	mssg_offset = dm->comm.dtemp;
	extra_send_offsets = dm->comm.extra_send_offsets;
}

static void set_pull_push_receive (comm_t * cm)
{
	receive = cm->receive;
}

void free_partition_list (master_t * mst)
{
	int num_of_procs = mst->num_of_cpus + mst->num_of_devices;
	int i;
	for (i = 0; i < num_of_procs; i++)
	{
		free (mst->not_reside[i]);
		free (mst->id2index[i]);
		free (mst->pfrom[i]);
	}
	free (mst->partition_list);
	free (mst->r2s);
}


static void init_lr (uint size, int num_of_partitions, int cur_id, voff_t index_offset)
{
//	printf ("current id: %d\n", cur_id);
	vid_t * local_post = posts + index_offset;
	vid_t * local_pre = pres + index_offset;
	voff_t * local_fwd = fwd + index_offset;
	voff_t * local_bwd = bwd + index_offset;
	vid_t * local_fjid = fjid + index_offset;
	vid_t * local_bjid = bjid + index_offset;
	uint * local_spid_nbs = spid_nbs + index_offset;
	uint * local_spid_js = spid_js + index_offset;
	uint r;
	uint count=0;
	uint loop = 0;
#pragma omp parallel for num_threads(cpu_threads)
	for (r = 0; r < size; r++)
	{
		int index = r;
		vid_t self_id;
		get_selfid(self_id, index, jid_offset[cur_id], id_offsets[cur_id]);
		int preid;
		int postid;
		get_pid_for_pre(preid, local_spid_nbs[index]);
		get_pid_for_post(postid, local_spid_nbs[index]);
		if (local_pre[index] == local_post[index] && preid == postid)
		{
			local_bjid[index] = self_id; // remove one edge
			update_pid_for_pre(preid, &local_spid_js[index]);
			local_pre[index] = MAX_VID; // selfloop
			loop++;
//			int pid = query_partition_id_from_idoffsets (local_post[index], num_of_partitions, id_offsets);
			if (preid<0 || preid>=num_of_partitions)
			{
				printf ("query id error! local_post[index]=%u, local_pre[index]=%u\n", local_post[index], local_pre[index]);
				exit(0);
			}
			if (is_junction(local_post[index], jid_offset[postid], id_offsets[postid])) // post is a junction
			{
				local_fjid[index] = local_post[index];
				update_pid_for_post(postid, &local_spid_js[index]);
				local_post[index] = MAX_VID; // junction is reached
				count++;
			}
		}
		else
		{
			if (local_pre[index] == self_id && preid == cur_id) // pre is a selfloop
			{
				local_bjid[index] = self_id; // selfloop
				update_pid_for_pre(preid, &local_spid_js[index]);
				local_pre[index] = MAX_VID;
				loop++;
			}
			else
			{

//				int pid = query_partition_id_from_idoffsets (local_pre[index], num_of_partitions, id_offsets);
				if (preid<0 || preid>=num_of_partitions)
				{
					printf ("query id error! local_post[index]=%u, local_pre[index]=%u\n", local_post[index], local_pre[index]);
					exit(0);
				}
				if (is_junction(local_pre[index], jid_offset[preid], id_offsets[preid])) // pre is a junction
				{
					local_bjid[index] = local_pre[index];
					update_pid_for_pre(preid, &local_spid_js[index]);
					local_pre[index] = MAX_VID; // junction is reached
					count++;
				}
			}
			if (local_post[index] == self_id && postid == cur_id) // post is a selfloop
			{
				local_fjid[index] = self_id;
				update_pid_for_post(postid, &local_spid_js[index]);
				local_post[index] = MAX_VID;
				loop++;
			}
			else
			{
//				int pid = query_partition_id_from_idoffsets (local_post[index], num_of_partitions, id_offsets);
				if (postid<0 || postid>=num_of_partitions)
				{
					printf ("query id error! local_post[index]=%u, local_pre[index]=%u\n", local_post[index], local_pre[index]);
					exit(0);
				}
				if (is_junction(local_post[index], jid_offset[postid], id_offsets[postid])) // post is a junction
				{
					local_fjid[index] = local_post[index];
					update_pid_for_post(postid, &local_spid_js[index]);
					local_post[index] = MAX_VID; // junction is reached
					count++;
				}
			}
		}
		local_fwd[index] = 1;
		local_bwd[index] = 1;
	}
//	printf ("JUNCTION %u, LOOP %u, SIZE %u\n", count, loop, size);
}


static void push_mssg_offset_lr_cpu (uint size, int num_of_partitions, int curr_pid, voff_t index_offset)
{
	omp_set_num_threads(cpu_threads);
#pragma omp parallel
	{
		int nth = omp_get_num_threads();
		if (nth != cpu_threads)
		{
			printf ("warning: setting cpu thread number failed! real number of threads: %d!\n", nth);
//			exit(0);
		}
		uint size_per_th = (size + nth - 1)/nth;
		if (size_per_th <= nth)
		{
			printf ("CCCCCCCCCCCcareful about size!\n");
			size_per_th = size/nth;
		}
		uint size_th;
		int thid = omp_get_thread_num();
		if (thid == nth-1)
		{
			size_th = size - size_per_th * (nth - 1);
		}
		else
			size_th = size_per_th;
		int r;
		vid_t * local_post = posts + index_offset + thid * size_per_th;
		vid_t * local_pre = pres + index_offset + thid * size_per_th;
		uint * local_spid_nbs = spid_nbs + index_offset + thid * size_per_th;
		voff_t * local_send_offsets = send_offsets_th + (thid+1) * (num_of_partitions+1);

		for(r = 0; r < size_th; r++)
		{
			int index = r;
			int pindex;
//			int pid;
			vid_t self_id;
			get_selfid(self_id, thid*size_per_th+index, jid_offset[curr_pid], id_offsets[curr_pid]);
			int postid;
			int preid;
			get_pid_for_post(postid, local_spid_nbs[index]);
			get_pid_for_pre(preid, local_spid_nbs[index]);
			if(local_post[index] != MAX_VID && !(local_post[index] == self_id && postid == curr_pid))
			{
//				pid = query_partition_id_from_idoffsets (local_post[index], num_of_partitions, id_offsets);
				pindex = id2index[postid];
				local_send_offsets[pindex+1]++;
			}
			if(local_pre[index] != MAX_VID && !(local_pre[index] == self_id && preid == curr_pid))
			{
//				pid = query_partition_id_from_idoffsets (local_pre[index], num_of_partitions, id_offsets);
				pindex = id2index[preid];
				local_send_offsets[pindex+1]++;
			}
		}
	}

}

static void push_mssg_lr_cpu (uint size, int num_of_partitions, int curr_pid, voff_t index_offset)
{
	omp_set_num_threads(cpu_threads);
#pragma omp parallel
	{
		int nth = omp_get_num_threads();
		if (nth != cpu_threads)
		{
			printf ("warning: setting cpu thread number failed! real number of threads: %d!\n", nth);
//			exit(0);
		}
		uint size_per_th = (size + nth - 1)/nth;
		if (size_per_th <= nth)
		{
			printf ("CCCCCCCCCCCcareful about size!\n");
			size_per_th = size/nth;
		}
		uint size_th;
		int thid = omp_get_thread_num();
		if (thid == nth-1)
		{
			size_th = size - size_per_th * (nth - 1);
		}
		else
			size_th = size_per_th;

		int r;
		vid_t * local_post = posts + index_offset + thid * size_per_th;
		vid_t * local_pre = pres + index_offset + thid * size_per_th;
		voff_t * local_fwd = fwd + index_offset + thid * size_per_th;
		voff_t * local_bwd = bwd + index_offset + thid * size_per_th;
		vid_t * local_fjid = fjid + index_offset + thid * size_per_th;
		vid_t * local_bjid = bjid + index_offset + thid * size_per_th;
		uint * local_spid_nbs = spid_nbs + index_offset + thid * size_per_th;
		uint * local_spid_js = spid_js + index_offset + thid * size_per_th;
		voff_t * local_send_offsets = tmp_send_offsets_th + (thid+1) * (num_of_partitions+1);
		path_t * buf = (path_t *)send;

		for (r=0; r<size_th; r++)
		{
			int index = r;
//			int pid;
			int pindex;
			voff_t off;
			path_t tmp;
			voff_t local_offset;
			vid_t self_id;
			get_selfid(self_id, thid*size_per_th+index, jid_offset[curr_pid], id_offsets[curr_pid]);
			int postid;
			int preid;
			get_pid_for_post(postid, local_spid_nbs[index]);
			get_pid_for_pre(preid, local_spid_nbs[index]);
			if(local_post[index] != MAX_VID && !(local_post[index] == self_id && postid == curr_pid)) // post neighbor is a linear vertex
			{
//				pid = query_partition_id_from_idoffsets (local_post[index], num_of_partitions, id_offsets);
				pindex = id2index[postid];
				local_offset = local_send_offsets[pindex+1]++;
				off = local_offset + send_offsets_th[thid*(num_of_partitions+1)+(pindex+1)] + send_offsets[pindex];
				tmp.dst = local_post[index] - jid_offset[postid]; // index of dst vertex
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

			if(local_pre[index] != MAX_VID && !(local_pre[index] == self_id && preid == curr_pid)) // pre neighbor is a linear vertex
			{
//				pid = query_partition_id_from_idoffsets (local_pre[index], num_of_partitions, id_offsets);
				pindex = id2index[preid];
				local_offset = local_send_offsets[pindex+1]++;
				off = local_offset + send_offsets_th[thid*(num_of_partitions+1)+(pindex+1)] + send_offsets[pindex];
				tmp.dst = local_pre[index] - jid_offset[preid]; // index of dst vertex
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
}

static void pull_mssg_lr_cpu (uint num_mssgs, int pid, voff_t index_offset, voff_t receive_start, bool intra_inter)
{
	path_t * buf;
	int pindex = id2index[pid];
	if (intra_inter)
		buf = (path_t *)send + receive_start + send_offsets[pindex];
	else buf = (path_t *)send + receive_start + receive_offsets[pindex];
	int r;
	vid_t * local_post = posts + index_offset;
	vid_t * local_pre = pres + index_offset;
	voff_t * local_fwd = fwd + index_offset;
	voff_t * local_bwd = bwd + index_offset;
	vid_t * local_fjid = fjid + index_offset;
	vid_t * local_bjid = bjid + index_offset;
	uint * local_spid_nbs = spid_nbs + index_offset;
	uint * local_spid_js = spid_js + index_offset;

#pragma omp parallel for num_threads(cpu_threads)
	for (r=0; r<num_mssgs; r++)
	{
		int index = r;
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
			exit(0);
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
				local_fjid[vindex] = local_post[vindex]; // temporarily record current to junction
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
				local_bjid[vindex] = local_pre[vindex]; // temporarily record current to junction
				update_pid_for_pre(preid, &local_spid_js[vindex]);
			}
			local_pre[vindex] = tmp.opps;
			update_pid_for_pre(oppsid, &local_spid_nbs[vindex]);
			local_bwd[vindex] += tmp.dist;
		}
		else
		{
			dst_not_found++;
			printf ("CPUCPUCPUCPUCPU:: %d :: pid=%d, index=%u error!!!: cannot find destination!\n", (int)intra_inter, pid, index);
		}
	}
}

static void push_selfloop_offset_cpu (uint num_mssgs, int pid, voff_t index_offset, int num_of_partitions, voff_t receive_start, bool intra_inter)
{
	omp_set_num_threads(cpu_threads);
#pragma omp parallel
	{
		int nth = omp_get_num_threads();
		if (nth != cpu_threads)
		{
			printf ("warning: setting cpu thread number failed! real number of threads: %d!\n", nth);
			// exit(0);
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

		path_t * buf;
		vid_t * local_mssg_offset;
		voff_t * local_send_offsets = send_offsets_th + (thid+1) * (num_of_partitions+1);

		if (intra_inter)
		{
			int pindex = id2index[pid];
			buf = (path_t *)send + receive_start + send_offsets[pindex] + thid * size_per_th;
			local_mssg_offset = mssg_offset + receive_start + send_offsets[pindex] + thid * size_per_th;
		}
		else
		{
			int pindex = id2index[pid];
			buf = (path_t *)send + receive_start + receive_offsets[pindex] + thid * size_per_th;
			local_mssg_offset = mssg_offset + receive_start + receive_offsets[pindex] + thid * size_per_th;
		}
		vid_t * local_post = posts + index_offset;
		vid_t * local_pre = pres + index_offset;

		int r;
		for (r=0; r<size_th; r++)
		{
			int index = r;
			path_t tmp = buf[index];
			vid_t vindex = tmp.dst;
			if (local_post[vindex] != tmp.cid && local_pre[vindex] != tmp.cid)
			{
				dst_not_found++;
				int cpid = query_partition_id_from_idoffsets (tmp.cid, num_of_partitions, id_offsets);
				int pindex = id2index[cpid];
				local_send_offsets[pindex+1]++;
	//			local_mssg_offset[r] = atomic_increase(&local_offsets[pindex + 1], 1);
	//			printf ("error!!!: cannot find destination!\n");
			}
		}
	}
}


static void push_selfloop_cpu (uint num_mssgs, int pid, voff_t index_offset, int num_of_partitions, voff_t receive_start, bool intra_inter)
{
	omp_set_num_threads(cpu_threads);
#pragma omp parallel
	{
		int nth = omp_get_num_threads();
		if (nth != cpu_threads)
		{
			printf ("warning: setting cpu thread number failed! real number of threads: %d!\n", nth);
			// exit(0);
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

		path_t * buf;
		selfloop_t * vs = (selfloop_t *)receive;
		vid_t * local_mssg_offset;
		voff_t * local_offsets = extra_send_offsets;
		voff_t * local_send_offsets = tmp_send_offsets_th + (thid+1) * (num_of_partitions+1);
		if (intra_inter)
		{
			int pindex = id2index[pid];
			buf = (path_t *)send + receive_start + send_offsets[pindex] + thid * size_per_th;
			local_mssg_offset = mssg_offset + receive_start + send_offsets[pindex] + thid * size_per_th;
		}
		else
		{
			int pindex = id2index[pid];
			buf = (path_t *)send + receive_start + receive_offsets[pindex] + thid * size_per_th;
			local_mssg_offset = mssg_offset + receive_start + receive_offsets[pindex] + thid * size_per_th;
		}
		vid_t * local_post = posts + index_offset;
		vid_t * local_pre = pres + index_offset;
		int r;
		for (r=0; r<size_th; r++)
		{
			int index = r;
			path_t tmp = buf[index];
			vid_t vindex = tmp.dst;
			voff_t local_send_offset;
			if (local_post[vindex] != tmp.cid && local_pre[vindex] != tmp.cid)
			{
				int cpid = query_partition_id_from_idoffsets (tmp.cid, num_of_partitions, id_offsets);
				int pindex = id2index[cpid];
				local_send_offset = local_send_offsets[pindex+1]++;
				off_t off = local_send_offset + send_offsets_th[thid*(num_of_partitions+1)+(pindex+1)] + local_offsets[pindex];
				vs[off].v = tmp.cid - jid_offset[cpid] - id_offsets[cpid];
				vs[off].dst = tmp.dst + jid_offset[pid] + id_offsets[pid];
//				vs[local_mssg_offset[r] + local_offsets[pindex]].v = tmp.cid - jid_offset[cpid] - id_offsets[cpid];
//				vs[local_mssg_offset[r] + local_offsets[pindex]].dst = tmp.dst + jid_offset[pid] + id_offsets[pid];
			}
		}
	}
}
static void pull_selfloop_cpu (uint num_mssgs, int pid, voff_t index_offset, selfloop_t * local_receive, bool intra_inter)
{
	selfloop_t * vs;
	vid_t * local_post = posts + index_offset;
	vid_t * local_pre = pres + index_offset;
	vid_t * local_fjid = fjid + index_offset;
	vid_t * local_bjid = bjid + index_offset;

	if (intra_inter) // true if intra partitions
	{
		int pindex = id2index[pid];
		vs = (selfloop_t *)local_receive + extra_send_offsets[pindex];
	}
	else
	{
		int pindex = id2index[pid];
		vs = (selfloop_t *)local_receive + receive_offsets[pindex];
	}

	int r;
#pragma omp parallel for num_threads(cpu_threads)
	for (r=0; r <num_mssgs; r++)
	{
		int index = r;
		int i = vs[index].v;
		if (local_post[i] == vs[index].dst)
		{
			local_post[i] = MAX_VID;
			local_fjid[i] = i + jid_offset[pid] + id_offsets[pid]; // Meanwhile, local_fjid[i] = self_id to denote a selfloop
			selfloop_made++;
		}
		else if (local_pre[i] == vs[index].dst)
		{
			local_pre[i] = MAX_VID;
			local_bjid[i] = i + jid_offset[pid] + id_offsets[pid]; // Meanwhile, local_bjid[i] = self_id to denote a selfloop
			selfloop_made++;
		}
		else
		{
			printf ("CPUCPUCPUCPUCPU:: %d :: ERROR: selfloop not made!\n", (int)intra_inter);
		}
	}
}


void * listrank_push_cpu (void * arg)
{
	evaltime_t start, end;
	lr_arg * carg = (lr_arg *) arg;
	comm_t * cm = carg->cm;
	master_t * mst = carg->mst;
	int num_of_cpus = mst->num_of_cpus;
	int num_of_devices = mst->num_of_devices;
	int did = carg->did;
	int total_num_partitions = mst->total_num_partitions;
	int num_of_partitions = mst->num_partitions[did + 1] - mst->num_partitions[did];
	int poffset = mst->num_partitions[did];
	voff_t * index_offset = mst->index_offset[did];

	if (mst->world_rank == 0)
		printf ("WORLD RANK %d: listrank push cpu:\n", mst->world_rank);

	memset(cm->send_offsets, 0, sizeof(voff_t) * (total_num_partitions + 1));
	memset (send_offsets_th, 0, sizeof(voff_t) * (total_num_partitions+1) * (cpu_threads+1));

#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	int i;
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset+i];
		voff_t size = mst->id_offsets[pid+1] - mst->id_offsets[pid] - mst->jid_offset[pid];
		push_mssg_offset_lr_cpu (size, total_num_partitions, pid, index_offset[i]);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	print_exec_time (start, end, "***************** PUSH MSSG OFFSET FOR CPU LISTRANKING INTRA PROCESSOR TIME: ");
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
		voff_t size = mst->id_offsets[pid+1] - mst->id_offsets[pid] - mst->jid_offset[pid];
		push_mssg_lr_cpu (size, total_num_partitions, pid, index_offset[i]);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	print_exec_time (start, end, "***************** PUSH MSSG FOR CPU LISTRANKING INTRA PROCESSOR TIME: ");
#endif

	memcpy(mst->roff[did], cm->send_offsets, sizeof(voff_t)*(total_num_partitions + 1));
	voff_t inter_start = mst->roff[did][num_of_partitions];
	voff_t inter_end = mst->roff[did][total_num_partitions];
//	memcpy(mst->receive[did], (path_t *)cm->send+inter_start, (inter_end-inter_start)*sizeof(path_t));
//	pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
//	pthread_mutex_lock(&mutex);
	mst->receive[did] = (path_t *)cm->send+inter_start;
//	pthread_mutex_unlock(&mutex);

	return ((void*) 0);
}

void * listrank_pull_cpu (void * arg)
{
	evaltime_t start, end;
	lr_arg * carg = (lr_arg *) arg;
	comm_t * cm = carg->cm;
	master_t * mst = carg->mst;
	int num_of_cpus = mst->num_of_cpus;
	int num_of_devices = mst->num_of_devices;
	int did = carg->did;
	int total_num_partitions = mst->total_num_partitions;
	int num_of_partitions = mst->num_partitions[did + 1] - mst->num_partitions[did];
	int poffset = mst->num_partitions[did];
	voff_t * index_offset = mst->index_offset[did];
	if (mst->world_rank == 0)
		printf ("WORLD RANK %d: listrank pull cpu:\n", mst->world_rank);

#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	int i;
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset + i];
		voff_t num_mssgs = cm->send_offsets[i+1] - cm->send_offsets[i];
		pull_mssg_lr_cpu (num_mssgs, pid, index_offset[i], 0, 1);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	print_exec_time (start, end, "***************** PULL MSSG FOR CPU LISTRANKING INTRA PROCESSOR TIME: ");
#endif

	voff_t receive_start = mst->roff[did][num_of_partitions];
	memcpy(cm->receive_offsets, mst->soff[did], sizeof(voff_t) * (num_of_partitions + 1));
	voff_t inter_size = mst->soff[did][num_of_partitions];
	memcpy((path_t*)cm->send + receive_start, mst->send[did], sizeof(path_t) * inter_size);

#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset + i];
		voff_t num_mssgs = cm->receive_offsets[i+1] - cm->receive_offsets[i];
		pull_mssg_lr_cpu (num_mssgs, pid, index_offset[i], receive_start, 0);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	print_exec_time (start, end, "***************** PULL MSSG FOR CPU LISTRANKING INTER PROCESSORS TIME: ");
#endif

	return ((void *) 0);
}

void * listrank_push_intra_pull_cpu (void * arg)
{
	evaltime_t overs, overe;
	evaltime_t start, end;
	lr_arg * carg = (lr_arg *) arg;
	comm_t * cm = carg->cm;
	master_t * mst = carg->mst;
	int num_of_cpus = mst->num_of_cpus;
	int num_of_devices = mst->num_of_devices;
	int did = carg->did;
	int total_num_partitions = mst->total_num_partitions;
	int num_of_partitions = mst->num_partitions[did + 1] - mst->num_partitions[did];
	int poffset = mst->num_partitions[did];
	voff_t * index_offset = mst->index_offset[did];
	if (mst->world_rank == 0)
		printf ("WORLD RANK %d: listrank push intra pull cpu:\n", mst->world_rank);

	gettimeofday (&overs, NULL);
	memset(cm->send_offsets, 0, sizeof(voff_t) * (total_num_partitions + 1));
	memset (send_offsets_th, 0, sizeof(voff_t) * (total_num_partitions+1) * (cpu_threads+1));

#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	int i;
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset+i];
		voff_t size = mst->id_offsets[pid+1] - mst->id_offsets[pid] - mst->jid_offset[pid];
		push_mssg_offset_lr_cpu (size, total_num_partitions, pid, index_offset[i]);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	push_offset_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "***************** PUSH MSSG OFFSET FOR CPU LISTRANKING INTRA PROCESSOR TIME: ");
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
		voff_t size = mst->id_offsets[pid+1] - mst->id_offsets[pid] - mst->jid_offset[pid];
		push_mssg_lr_cpu (size, total_num_partitions, pid, index_offset[i]);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	push_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "***************** PUSH MSSG FOR CPU LISTRANKING INTRA PROCESSOR TIME: ");
#endif

	memcpy(mst->roff[did], cm->send_offsets, sizeof(voff_t)*(total_num_partitions + 1));
	voff_t inter_start = mst->roff[did][num_of_partitions];
	voff_t inter_end = mst->roff[did][total_num_partitions];
//	memcpy(mst->receive[did], (path_t *)cm->send+inter_start, (inter_end-inter_start)*sizeof(path_t));
//	pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
//	pthread_mutex_lock(&mutex);
	mst->receive[did] = (path_t *)cm->send+inter_start;
//	lock_flag[did] = 1;
//	pthread_mutex_unlock(&mutex);

#ifndef SYNC_ALL2ALL_
	if (atomic_set_value (&lock_flag[did], 1, 0) == false)
		printf("!!!!!!!!!!!!!!!!!! CAREFUL, SETTING VALUE DOES NOT WORK FINE!\n");
#endif
/*	printf ("WORLD RANK %d::::::: RRRRRRRRRRRECHECK INDEX_OFFSETS::::::::", did);
	print_offsets (index_offset, num_of_partitions+1);

	printf ("WORLD RANK %d::::: TTTTTTTTTTTTTSET RECEIVE OFFSETS::::::::::::::", mst->world_rank);
	print_offsets (mst->roff[did], total_num_partitions+1);*/

#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset + i];
		voff_t num_mssgs = cm->send_offsets[i+1] - cm->send_offsets[i];
		pull_mssg_lr_cpu (num_mssgs, pid, index_offset[i], 0, 1);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	pull_intra_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "***************** PULL MSSG FOR CPU LISTRANKING INTRA PROCESSOR TIME: ");
#endif

	gettimeofday (&overe, NULL);
	over_time[did] += (float)((overe.tv_sec * 1000000 + overe.tv_usec) - (overs.tv_sec * 1000000 + overs.tv_usec)) / 1000;

	return ((void*) 0);
}

void * listrank_inter_pull_cpu (void * arg)
{
	evaltime_t overs, overe;
	evaltime_t start, end;
	lr_arg * carg = (lr_arg *) arg;
	comm_t * cm = carg->cm;
	master_t * mst = carg->mst;
	int num_of_cpus = mst->num_of_cpus;
	int num_of_devices = mst->num_of_devices;
	int did = carg->did;
	int total_num_partitions = mst->total_num_partitions;
	int num_of_partitions = mst->num_partitions[did + 1] - mst->num_partitions[did];
	int poffset = mst->num_partitions[did];
	voff_t * index_offset = mst->index_offset[did];
	if (mst->world_rank == 0)
		printf ("WORLD RANK %d: listrank inter pull cpu:\n", mst->world_rank);

	gettimeofday (&overs, NULL);
	voff_t receive_start = mst->roff[did][num_of_partitions];
	memcpy(cm->receive_offsets, mst->soff[did], sizeof(voff_t) * (num_of_partitions + 1));
	voff_t inter_size = mst->soff[did][num_of_partitions];
	if(inter_size == 0)
		return ((void*) 0);
	memcpy((path_t*)cm->send + receive_start, mst->send[did], sizeof(path_t) * inter_size);

	int i;
#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset + i];
		voff_t num_mssgs = cm->receive_offsets[i+1] - cm->receive_offsets[i];
		pull_mssg_lr_cpu (num_mssgs, pid, index_offset[i], receive_start, 0);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	pull_inter_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "***************** PULL MSSG FOR CPU LISTRANKING INTER PROCESSORS TIME: ");
#endif

	gettimeofday (&overe, NULL);
	over_time[did] += (float)((overe.tv_sec * 1000000 + overe.tv_usec) - (overs.tv_sec * 1000000 + overs.tv_usec)) / 1000;

	return ((void *) 0);
}
void init_graph_data_cpu (int did, master_t * mst, meta_t * dm, d_lvs_t * lvs)
{
	int * num_partitions = mst->num_partitions;
	int * partition_list = mst->partition_list;
	voff_t * index_offset = mst->index_offset[did];
	int num_of_partitions = num_partitions[did+1]-num_partitions[did];
	int total_num_partitions = mst->total_num_partitions;
	set_globals_cpu (dm, total_num_partitions);
	set_mssg_offset_buffer(dm);

	int world_size = mst->world_size;
	int world_rank = mst->world_rank;
	int np_per_node = (total_num_partitions + world_size - 1)/world_size;
	int start_partition_id = np_per_node*world_rank;

	int i;
	voff_t offset = 0;
	for (i=0; i<num_of_partitions; i++)
	{
		int poffset = num_partitions[did];
		int pid = partition_list[poffset+i] - start_partition_id;
		voff_t size = lvs[pid].asize + lvs[pid].esize;
		memcpy(dm->edge.post+offset, lvs[pid].posts, sizeof(vid_t)*size);
		memcpy(dm->edge.pre+offset, lvs[pid].pres, sizeof(vid_t)*size);
		memcpy(dm->edge.spid_nb+offset, lvs[pid].spids, sizeof(uint)*size);
		index_offset[i] = offset;
		init_lr (size, total_num_partitions, pid + start_partition_id, offset);
		offset += size;
	}
	index_offset[i] = offset;
	goffset_t * id_offsets = mst->id_offsets;
//	printf ("id offsets: ++++++++++++++++++++++++++++++++++++\n");
//	print_offsets (id_offsets, total_num_partitions + 1);
//	print_offsets (jid_offset, total_num_partitions);

//	printf ("index offsets: ++++++++++++++++++++++++++++++++++++==\n");
//	print_offsets (mst->index_offset[did], num_of_partitions+1);
}

void finalize_graph_data_cpu (void)
{
	release_globals_cpu();
}

void * listrank_push_modifygraph_intra_push_cpu (void * arg)
{
	evaltime_t overs, overe;
	evaltime_t start, end;
	lr_arg * carg = (lr_arg *) arg;
	comm_t * cm = carg->cm;
	master_t * mst = carg->mst;
	int num_of_cpus = mst->num_of_cpus;
	int num_of_devices = mst->num_of_devices;
	int did = carg->did;
	if (mst->world_rank == 0)
		printf ("WORLD RANK %d:::::::: listrank push modifygraph intra push cpu:\n", mst->world_rank);

	gettimeofday (&overs, NULL);
	int total_num_partitions = mst->total_num_partitions;
	int num_of_partitions = mst->num_partitions[did + 1] - mst->num_partitions[did];
	int poffset = mst->num_partitions[did];
	voff_t * index_offset = mst->index_offset[did];

	memset(cm->send_offsets, 0, sizeof(voff_t) * (total_num_partitions + 1));
	memset (send_offsets_th, 0, sizeof(voff_t) * (total_num_partitions+1) * (cpu_threads+1));

#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	int i;
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset+i];
		voff_t size = mst->id_offsets[pid+1] - mst->id_offsets[pid] - mst->jid_offset[pid];
		push_mssg_offset_lr_cpu (size, total_num_partitions, pid, index_offset[i]);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	push_offset_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "***************** PUSH MSSG OFFSET FOR CPU LISTRANKING INTRA PROCESSOR TIME: ");
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
		voff_t size = mst->id_offsets[pid+1] - mst->id_offsets[pid] - mst->jid_offset[pid];
		push_mssg_lr_cpu (size, total_num_partitions, pid, index_offset[i]);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	push_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "***************** PUSH MSSG FOR CPU LISTRANKING INTRA PROCESSOR TIME: ");
#endif

	memcpy(mst->roff[did], cm->send_offsets, sizeof(voff_t)*(total_num_partitions + 1));
	voff_t inter_start = mst->roff[did][num_of_partitions];
	voff_t inter_end = mst->roff[did][total_num_partitions];
//	memcpy(mst->receive[did], (path_t *)cm->send+inter_start, (inter_end-inter_start)*sizeof(path_t));
	mst->receive[did] = (path_t *)cm->send+inter_start;

	memset (cm->extra_send_offsets, 0, sizeof(voff_t) * (total_num_partitions+1));
	memset (send_offsets_th, 0, sizeof(voff_t) * (total_num_partitions+1) * (cpu_threads+1));

#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset + i];
		voff_t num_mssgs = cm->send_offsets[i+1] - cm->send_offsets[i];
		push_selfloop_offset_cpu (num_mssgs, pid, index_offset[i], total_num_partitions, 0, 1);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	pull_intra_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "*********** PUSH SELFLOOP OFFSET CPU INTRA PROCESSOR TIME: ");
#endif

	gettimeofday (&overe, NULL);
	over_time[did] += (float)((overe.tv_sec * 1000000 + overe.tv_usec) - (overs.tv_sec * 1000000 + overs.tv_usec)) / 1000;

	return ((void*) 0);
}
/*
void * listrank_pull_modifygraph_push_cpu (void * arg)
{
	evaltime_t start, end;
	lr_arg * carg = (lr_arg *) arg;
	comm_t * cm = carg->cm;
	master_t * mst = carg->mst;
	int num_of_cpus = mst->num_of_cpus;
	int num_of_devices = mst->num_of_devices;
	int did = carg->did;
	int total_num_partitions = mst->total_num_partitions;
	int num_of_partitions = mst->num_partitions[did + 1] - mst->num_partitions[did];
	int poffset = mst->num_partitions[did];
	voff_t * index_offset = mst->index_offset[did];
	memset (cm->extra_send_offsets, 0, sizeof(voff_t) * (total_num_partitions+1));
	memset (send_offsets_th, 0, sizeof(voff_t) * (total_num_partitions+1) * (cpu_threads+1));
	if (mst->world_rank == 0)
		printf ("WORLD RANK %d::::::: listrank pull modifygraph push cpu:\n", mst->world_rank);

	voff_t receive_start = cm->send_offsets[num_of_partitions];
#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	int i;
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset + i];
		voff_t num_mssgs = cm->send_offsets[i+1] - cm->send_offsets[i];
		push_selfloop_offset_cpu (num_mssgs, pid, index_offset[i], total_num_partitions, 0, 1);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	print_exec_time (start, end, "*********** PUSH SELFLOOP OFFSET CPU INTRA PROCESSOR TIME: ");
#endif

	memcpy(cm->receive_offsets, mst->soff[did], sizeof(voff_t) * (num_of_partitions + 1));
	voff_t inter_size = cm->receive_offsets[num_of_partitions];
	memcpy((path_t*)(cm->send) + receive_start, mst->send[did], sizeof(path_t) * inter_size);

#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset + i];
		voff_t num_mssgs = cm->receive_offsets[i+1] - cm->receive_offsets[i];
		push_selfloop_offset_cpu (num_mssgs, pid, index_offset[i], total_num_partitions, receive_start, 0);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	print_exec_time (start, end, "*********** PUSH SELFLOOP OFFSET CPU INTER PROCESSORS TIME: ");
#endif

	get_global_offsets (cm->extra_send_offsets, send_offsets_th, total_num_partitions, (cpu_threads+1));
	inclusive_prefix_sum (cm->extra_send_offsets, total_num_partitions + 1);

	// *************** malloc (send and) receive buffer for pull and push mode
	voff_t rcv_size = cm->extra_send_offsets[num_of_partitions];
	if (rcv_size == 0)
	{
		printf ("CPU::::::::::CCCCCCCCCcccareful:::::::::: receive size from intra selfloop push is 0!!!!!!!!\n");
		rcv_size = 200;
	}
	malloc_pull_push_receive_cpu (cm, sizeof(selfloop_t), did, rcv_size, 100*(total_num_partitions+num_of_partitions-1)/num_of_partitions);
	set_pull_push_receive (cm);

	memset (tmp_send_offsets_th, 0, sizeof(voff_t) * (total_num_partitions+1) * (cpu_threads+1));

#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset+i];
		voff_t num_mssgs = cm->send_offsets[i+1] - cm->send_offsets[i];
		push_selfloop_cpu (num_mssgs, pid, index_offset[i], total_num_partitions, 0, 1);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	print_exec_time (start, end, "************** PUSH SELFOOLP CPU INTRA PROCESSOR TIME: ");
#endif

#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset+i];
		voff_t num_mssgs = cm->receive_offsets[i+1] - cm->receive_offsets[i];
		push_selfloop_cpu (num_mssgs, pid, index_offset[i], total_num_partitions, receive_start, 0);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	print_exec_time (start, end, "************** PUSH SELFOOLP CPU INTER PROCESSORS TIME: ");
#endif

	memcpy(mst->roff[did], cm->extra_send_offsets, sizeof(voff_t)*(total_num_partitions + 1));
	voff_t inter_start = cm->extra_send_offsets[num_of_partitions];
	voff_t inter_end = cm->extra_send_offsets[total_num_partitions];

	mst->receive[did] = (selfloop_t*)cm->receive + inter_start;
//	memcpy(mst->receive[did], (selfloop_t *)cm->receive+inter_start, (inter_end-inter_start)*sizeof(selfloop_t));

	return ((void *) 0);
}
*/
void * listrank_pull_modifygraph_inter_push_intra_pull_cpu (void * arg)
{
	evaltime_t overs, overe;
	evaltime_t start, end;
	lr_arg * carg = (lr_arg *) arg;
	comm_t * cm = carg->cm;
	master_t * mst = carg->mst;
	int num_of_cpus = mst->num_of_cpus;
	int num_of_devices = mst->num_of_devices;
	int did = carg->did;
	int total_num_partitions = mst->total_num_partitions;
	int num_of_partitions = mst->num_partitions[did + 1] - mst->num_partitions[did];
	int poffset = mst->num_partitions[did];
	voff_t * index_offset = mst->index_offset[did];
//	memset (cm->extra_send_offsets, 0, sizeof(voff_t) * (total_num_partitions+1));
//	memset (send_offsets_th, 0, sizeof(voff_t) * (total_num_partitions+1) * (cpu_threads+1));
	if (mst->world_rank == 0)
		printf ("WORLD RANK %d::::::: listrank pull modifygraph inter push intra pull cpu:\n", mst->world_rank);

	gettimeofday (&overs, NULL);
	voff_t receive_start = cm->send_offsets[num_of_partitions];
	memcpy(cm->receive_offsets, mst->soff[did], sizeof(voff_t) * (num_of_partitions + 1));
	voff_t inter_size = cm->receive_offsets[num_of_partitions];
	memcpy((path_t*)(cm->send) + receive_start, mst->send[did], sizeof(path_t) * inter_size);

#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	int i;
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset + i];
		voff_t num_mssgs = cm->receive_offsets[i+1] - cm->receive_offsets[i];
		push_selfloop_offset_cpu (num_mssgs, pid, index_offset[i], total_num_partitions, receive_start, 0);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	pull_inter_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "*********** PUSH SELFLOOP OFFSET CPU INTER PROCESSORS TIME: ");
#endif

	get_global_offsets (cm->extra_send_offsets, send_offsets_th, total_num_partitions, (cpu_threads+1));
	inclusive_prefix_sum (cm->extra_send_offsets, total_num_partitions + 1);

	// *************** malloc (send and) receive buffer for pull and push mode
	voff_t rcv_size = cm->extra_send_offsets[num_of_partitions];
	if (rcv_size == 0)
	{
		printf ("WORLD_RANK %d:CPU:::::::::: CCCCCCCCCcccareful:::::::::: receive size from intra selfloop push is 0!!!!!!!!\n", mst->world_rank);
		printf ("WORLD_RANK %d:CPU::::::::::CCCCCCCCCCcheck number of messages pushed to inter selfloop: %u\n", mst->world_rank, cm->extra_send_offsets[total_num_partitions]);
		rcv_size = 200;
	}
	malloc_pull_push_receive_cpu (cm, sizeof(selfloop_t), did, rcv_size, 100*(total_num_partitions+num_of_partitions-1)/num_of_partitions);
	set_pull_push_receive (cm);

	memset (tmp_send_offsets_th, 0, sizeof(voff_t) * (total_num_partitions+1) * (cpu_threads+1));

#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset+i];
		voff_t num_mssgs = cm->send_offsets[i+1] - cm->send_offsets[i];
		push_selfloop_cpu (num_mssgs, pid, index_offset[i], total_num_partitions, 0, 1);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	push_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "************** PUSH SELFOOLP CPU INTRA PROCESSOR TIME: ");
#endif

#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset+i];
		voff_t num_mssgs = cm->receive_offsets[i+1] - cm->receive_offsets[i];
		push_selfloop_cpu (num_mssgs, pid, index_offset[i], total_num_partitions, receive_start, 0);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	push_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "************** PUSH SELFOOLP CPU INTER PROCESSORS TIME: ");
#endif

	memcpy(mst->roff[did], cm->extra_send_offsets, sizeof(voff_t)*(total_num_partitions + 1));
	voff_t inter_start = cm->extra_send_offsets[num_of_partitions];
	voff_t inter_end = cm->extra_send_offsets[total_num_partitions];

	mst->receive[did] = (selfloop_t*)cm->receive + inter_start;
//	memcpy(mst->receive[did], (selfloop_t *)cm->receive+inter_start, (inter_end-inter_start)*sizeof(selfloop_t));

#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset + i];
		voff_t num_mssgs = cm->extra_send_offsets[i+1] - cm->extra_send_offsets[i];
		pull_selfloop_cpu (num_mssgs, pid, index_offset[i], (selfloop_t*)cm->receive, 1);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	pull_intra_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "***************** PULL SELFLOOP CPU INTRA PROCESSOR TIME: ");
#endif

	gettimeofday (&overe, NULL);
	over_time[did] += (float)((overe.tv_sec * 1000000 + overe.tv_usec) - (overs.tv_sec * 1000000 + overs.tv_usec)) / 1000;

	return ((void *) 0);
}

void * modifygraph_inter_pull_cpu (void * arg)
{
	evaltime_t overs, overe;
	evaltime_t start, end;
	lr_arg * carg = (lr_arg *) arg;
	comm_t * cm = carg->cm;
	master_t * mst = carg->mst;
	int did = carg->did;
	int num_of_partitions = mst->num_partitions[did + 1] - mst->num_partitions[did];
	int poffset = mst->num_partitions[did];
	voff_t * index_offset = mst->index_offset[did];
	if (mst->world_rank == 0)
		printf ("WORLD RANK %d:::::::::: modifygraph inter pull cpu:\n", mst->world_rank);

	gettimeofday (&overs, NULL);
	voff_t receive_start = cm->extra_send_offsets[num_of_partitions];
	memcpy(cm->receive_offsets, mst->soff[did], sizeof(voff_t) * (num_of_partitions+1));
	voff_t inter_size = cm->receive_offsets[num_of_partitions];
	memcpy((selfloop_t *)(cm->receive) + receive_start, mst->send[did], sizeof(selfloop_t) * inter_size);
	if (cm->temp_size < (inter_size+receive_start)*sizeof(selfloop_t))
	{
		printf("WORLD RANK %d:::Error:::::::: malloced receive buffer size smaller than actual receive buffer size!\n", mst->world_rank);
		exit(0);
	}

#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	int i;
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset + i];
		voff_t num_mssgs = cm->receive_offsets[i+1] - cm->receive_offsets[i];
		pull_selfloop_cpu (num_mssgs, pid, index_offset[i], (selfloop_t*)(cm->receive) + receive_start, 0);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	pull_inter_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "***************** PULL SELFLOOP CPU INTER PROCESSORS TIME: ");
#endif

	printf ("WORLD_RANK %d +++++++++ DST NOT FOUND IN FIRST ITERATION: %u\n", mst->world_rank, dst_not_found);
	gettimeofday (&overe, NULL);
	over_time[did] += (float)((overe.tv_sec * 1000000 + overe.tv_usec) - (overs.tv_sec * 1000000 + overs.tv_usec)) / 1000;

	// *************** free (send and) receive buffer for pull and push mode
	free_pull_push_receive_cpu(cm);

	return ((void *) 0);
}
/*
void * modifygraph_pull_cpu (void * arg)
{
	evaltime_t start, end;
	printf ("modifygraph pull cpu:\n");
	lr_arg * carg = (lr_arg *) arg;
	comm_t * cm = carg->cm;
	master_t * mst = carg->mst;
	int did = carg->did;
	int num_of_partitions = mst->num_partitions[did + 1] - mst->num_partitions[did];
	int poffset = mst->num_partitions[did];
	voff_t * index_offset = mst->index_offset[did];

#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	int i;
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset + i];
		voff_t num_mssgs = cm->extra_send_offsets[i+1] - cm->extra_send_offsets[i];
		pull_selfloop_cpu (num_mssgs, pid, index_offset[i], (selfloop_t*)cm->receive, 1);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	print_exec_time (start, end, "***************** PULL SELFLOOP CPU INTRA PROCESSOR TIME: ");
#endif
	voff_t receive_start = cm->extra_send_offsets[num_of_partitions];
	memcpy(cm->receive_offsets, mst->soff[did], sizeof(voff_t) * (num_of_partitions+1));
	voff_t inter_size = cm->receive_offsets[num_of_partitions];
	memcpy((selfloop_t *)(cm->receive) + receive_start, mst->send[did], sizeof(selfloop_t) * inter_size);

#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset + i];
		voff_t num_mssgs = cm->receive_offsets[i+1] - cm->receive_offsets[i];
		pull_selfloop_cpu (num_mssgs, pid, index_offset[i], (selfloop_t*)(cm->receive) + receive_start, 0);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	print_exec_time (start, end, "***************** PULL SELFLOOP CPU INTER PROCESSORS TIME: ");
#endif
	printf ("+++++++++ DST NOT FOUND IN FIRST ITERATION: %u\n", dst_not_found);

	// *************** free (send and) receive buffer for pull and push mode
	free_pull_push_receive_cpu(cm);

	return ((void *) 0);
}
*/
