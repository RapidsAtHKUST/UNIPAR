/*
 * contig.c
 *
 *  Created on: 2018-4-21
 *      Author: qiushuang
 *
 *  This file updates neighbors of junctions, overpassing the linear vertices; gathers contigs
 */

#include <omp.h>
#include <math.h>
#include "include/dbgraph.h"
#include "include/comm.h"
#include "include/preprocess.h"
#include "include/bitkmer.h"
#include "include/share.h"
#include "include/malloc.h"

#define CPU_THREADS 24
//#define CONTIG_CHECK

extern long cpu_threads;
static char rev_table[4] = {'A', 'C', 'G', 'T'};
static uint count = 0;

static voff_t * id_offsets; // used to assign global id for each node in subgraphs
static voff_t * jid_offset; // junction vertex id offsets, used to calculate id of each vertex from its index

/*CSR for junctions: */
static voff_t * jnb_offsets;
static vid_t * jnbs;

static voff_t * posts; // forward neighbor ids of linear vertices
static voff_t * pres; // backward neighbor ids of linear vertices, i.e., forward neighbor ids of reverses
static voff_t * fwd; // forward distance
static voff_t * bwd; // backward distance
static vid_t * fjid; // junction id for forward path
static vid_t * bjid; // junction id for backward path
static kmer_t * jkmers; // kmer values for junctions
static edge_type * post_edges; // post edge character for linear vertices
static edge_type * pre_edges; // pre edge character for linear vertices
static size_t * ulens; // for each junction, store the length of the unitig starting from this end point
static size_t * unitig_offsets; // record output offsets of unitigs for partitions in a processor
static char * unitigs; // for each pair of junctions, output a unitig
static ull * junct_edges; // for gathering contigs

static voff_t * send_offsets; // used to locate the write position of messages for each partition in send buffer
static voff_t * receive_offsets;
static voff_t * extra_send_offsets;
static int * id2index;
static voff_t * send_offsets_th;
static voff_t * tmp_send_offsets_th;

static void * send;
static void * receive;

voff_t * joffsets; // junction offsets indicating junction location offset in each partition in a processor
voff_t * jnboffsets; // location offsets of neighbors of junctions in each partition in a processor

extern int cutoff;
extern uint dst_not_found;
extern float push_offset_time[NUM_OF_PROCS];
extern float push_time[NUM_OF_PROCS];
extern float pull_intra_time[NUM_OF_PROCS];
extern float pull_inter_time[NUM_OF_PROCS];
extern float over_time[NUM_OF_PROCS];

static void set_globals_cpu (meta_t * dm, int num_of_partitions)
{
	posts = dm->edge.post;
	pres = dm->edge.pre;
	fwd = dm->edge.fwd;
	bwd = dm->edge.bwd;
	fjid = dm->edge.fjid;
	bjid = dm->edge.bjid;
	post_edges = dm->edge.post_edges;
	pre_edges = dm->edge.pre_edges;

	send_offsets = dm->comm.send_offsets;
	receive_offsets = dm->comm.receive_offsets;
	send = dm->comm.send;
	id2index = dm->comm.id2index;
	id_offsets = dm->id_offsets;
	jid_offset = dm->jid_offset;

	jkmers = dm->junct.kmers;
	junct_edges = dm->junct.edges;
	jnb_offsets = dm->junct.offs;
	jnbs = dm->junct.nbs;
	ulens = dm->junct.ulens;

	send_offsets_th = (voff_t*) malloc (sizeof(voff_t) * (num_of_partitions+1) * (cpu_threads+1));
	CHECK_PTR_RETURN (send_offsets_th, "malloc local send offsets for multi-threads in push mssg offset lr error!\n");
	memset (send_offsets_th, 0, sizeof(voff_t) * (num_of_partitions+1) * (cpu_threads+1));
	tmp_send_offsets_th = (voff_t*) malloc (sizeof(voff_t) * (num_of_partitions+1) * (cpu_threads+1));
	CHECK_PTR_RETURN (send_offsets_th, "malloc tmp send offsets for multi-threads in push mssg offset lr error!\n");
	memset (tmp_send_offsets_th, 0, sizeof(voff_t) * (num_of_partitions+1) * (cpu_threads+1));
}

static void set_pull_push_receive (comm_t * cm)
{
	receive = cm->receive;
}

static void set_mssg_offset_buffer (meta_t * dm)
{
	extra_send_offsets = dm->comm.extra_send_offsets;
}

static void release_globals_cpu (void)
{
	free(send_offsets_th);
	free(tmp_send_offsets_th);
}

void set_unitig_pointer_cpu (meta_t * dm)
{
	unitigs = dm->junct.unitigs;
	unitig_offsets = dm->junct.unitig_offs;
}

void finalize_contig_data_cpu (void)
{
	release_globals_cpu();
}

void init_contig_data_cpu (int did, d_jvs_t * jvs, d_lvs_t * lvs, meta_t * dm, master_t * mst)
{
	int * num_partitions = mst->num_partitions;
	int * partition_list = mst->partition_list;
	int num_of_partitions = num_partitions[did+1]-num_partitions[did];
	int total_num_partitions = mst->total_num_partitions;
	// index_offset, jindex_offset and jnb_index_offset are used for index of partition located in a processor
	voff_t * index_offset = mst->index_offset[did];
	voff_t * jindex_offset = mst->jindex_offset[did];
	voff_t * jnb_index_offset = mst->jnb_index_offset[did];
	// joffsets and jnboffsets are used for input of partitions
	voff_t * joffsets = jvs->csr_offs_offs; // the difference of joffsets and jindex_offset is that the order of partitions may be different!!!
	vid_t * jnboffsets = jvs->csr_nbs_offs; // the difference of jnboffsets and jnb_index_offset is that the order of partitions may be different!!!
	voff_t * joffs = jvs->csr_offs;
	vid_t * jnbs = jvs->csr_nbs;

	int world_size = mst->world_size;
	int world_rank = mst->world_rank;
	int np_per_node = (total_num_partitions + world_size - 1)/world_size;
	int start_partition_id = np_per_node*world_rank;

	set_globals_cpu (dm, total_num_partitions);
	set_mssg_offset_buffer(dm);

	int i;
	voff_t loffset = 0;
	voff_t joffset = 0;
	voff_t jnb_offset = 0;
	for (i=0; i<num_of_partitions; i++)
	{
		int poffset = num_partitions[did];
		int pid = partition_list[poffset+i] - start_partition_id;
		if (lvs[pid].asize != 0)
		{
			printf ("lsize error!!!\n");
//			exit(0);
		}
		voff_t lsize = lvs[pid].esize + lvs[pid].asize;
		voff_t jsize = jvs[pid].size;
		voff_t jnbsize = jnboffsets[pid+1] - jnboffsets[pid];
//		printf ("partition %d: pindex = %d, joffset = %u, loffset = %u\n", pid, i, joffset, loffset);
		memcpy(dm->edge.pre_edges+loffset, lvs[pid].pre_edges, sizeof(edge_type)*lsize);
		memcpy(dm->edge.post_edges+loffset, lvs[pid].post_edges, sizeof(edge_type)*lsize);
		memcpy(dm->junct.kmers+joffset, jvs[pid].kmers, sizeof(kmer_t)*jsize);
		memcpy(dm->junct.edges+joffset, jvs[pid].edges, sizeof(ull)*jsize);
		memcpy(dm->junct.offs+joffset+i, joffs + joffsets[pid] + pid, sizeof(voff_t)*(jsize+1));
		memcpy(dm->junct.nbs+jnb_offset, jnbs + jnboffsets[pid], sizeof(vid_t)*jnbsize);
		jindex_offset[i] = joffset;
		jnb_index_offset[i] = jnb_offset;
		loffset += lsize;
		joffset += jsize;
		jnb_offset += jnbsize;
	}

	jindex_offset[i] = joffset;
	jnb_index_offset[i] = jnb_offset;
//	printf ("index offset on CPU: \n", did);
//	print_offsets (index_offset, num_of_partitions+1);
//	print_offsets (jindex_offset, num_of_partitions+1);
//	print_offsets (jnb_index_offset, num_of_partitions+1);
}


// ********* this is compute among junctions *************
static void push_mssg_offset_compact_cpu (uint size, int num_of_partitions, int curr_pid, voff_t jindex_offset, voff_t jnb_index_offset)
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
		int r;

		int pindex = id2index[curr_pid];
		voff_t * local_send_offsets = send_offsets_th + (thid+1) * (num_of_partitions+1);
		voff_t * local_joffs = jnb_offsets + jindex_offset + pindex + thid * size_per_th;
		voff_t * local_nbs = jnbs + jnb_index_offset;

		for(r = 0; r < size_th; r++)
		{
			int pindex;
			int pid;
			int num = local_joffs[r+1] - local_joffs[r];
			int j;
			for (j=0; j<num; j++)
			{
				vid_t vertex = local_nbs[local_joffs[r] + j];
				pid = query_partition_id_from_idoffsets (vertex, num_of_partitions, id_offsets);
				if (vertex < id_offsets[pid] + jid_offset[pid]) // a junction neighbor
					continue;
				pindex = id2index[pid];
				local_send_offsets[pindex+1]++;
			}
		}
	}

}

// ********* this is compute among junctions *************
static void push_mssg_compact_cpu (uint size, int num_of_partitions, int curr_pid, voff_t jindex_offset, voff_t jnb_index_offset)
{
	omp_set_num_threads(cpu_threads);
#pragma omp parallel
	{
		int nth = omp_get_num_threads();
//		printf ("PUSH MSSG THREADS: %d!!!!!!!!!\n", nth);
		if (nth != cpu_threads)
		{
			printf ("warning: setting cpu thread number failed!\n");
//			exit(0);
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

		int r;

		int pindex = id2index[curr_pid];
		voff_t * local_send_offsets = tmp_send_offsets_th + (thid+1) * (num_of_partitions+1);
		voff_t * local_joffs = jnb_offsets + jindex_offset + pindex + thid * size_per_th;
		voff_t * local_nbs = jnbs + jnb_index_offset;
		query_t * buf = (query_t *)send;

		voff_t local_offset;
		for(r = 0; r < size_th; r++)
		{
			int pindex;
			int pid;
			int num = local_joffs[r+1] - local_joffs[r];

			voff_t off;
			int j;
			for (j=0; j<num; j++)
			{
				vid_t vertex = local_nbs[local_joffs[r] + j];
				pid = query_partition_id_from_idoffsets (vertex, num_of_partitions, id_offsets);
				if (vertex < id_offsets[pid] + jid_offset[pid]) // a junction neighbor
					continue;
				pindex = id2index[pid];
				local_offset = local_send_offsets[pindex+1]++;
				off = local_offset + send_offsets_th[thid*(num_of_partitions+1)+(pindex+1)] + send_offsets[pindex];
				buf[off].jid = id_offsets[curr_pid] + thid * size_per_th + r;
				buf[off].nid = vertex;
			}
		}
	}
}

// ********* this is compute among linear vertices *************
static void push_update_offset_cpu (uint num_mssgs, int pid, voff_t index_offset, int num_of_partitions, voff_t receive_start, bool intra_inter)
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

		query_t * buf;
		voff_t * local_send_offsets = send_offsets_th + (thid+1) * (num_of_partitions+1);

		if (intra_inter)
		{
			int pindex = id2index[pid];
			buf = (query_t *)send + receive_start + send_offsets[pindex] + thid * size_per_th;
		}
		else
		{
			int pindex = id2index[pid];
			buf = (query_t *)send + receive_start + receive_offsets[pindex] + thid * size_per_th;
		}
		vid_t * local_fjid = fjid + index_offset;
		vid_t * local_bjid = bjid + index_offset;

		voff_t r;
		for (r=0; r<size_th; r++)
		{
			int index = r;
			query_t tmp = buf[index];
			vid_t vindex = tmp.nid - id_offsets[pid] - jid_offset[pid];
			if (local_fjid[vindex] != tmp.jid && local_bjid[vindex] != tmp.jid)
			{
				dst_not_found++;
//				printf ("error!!!: cannot find destination!tmp.jid=%lu, local_fjid=%u, local_bjid=%u\n", \
						tmp.jid, local_fjid[vindex], local_bjid[vindex]);
			}
			else
			{
				int jpid = query_partition_id_from_idoffsets (tmp.jid, num_of_partitions, id_offsets);
				int pindex = id2index[jpid];
				local_send_offsets[pindex+1]++;
			}
		}
	}
}

// ********* this is compute among linear vertices *************
static void push_update_cpu (uint num_mssgs, int pid, voff_t index_offset, int num_of_partitions, voff_t receive_start, bool intra_inter)
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

		query_t * buf;
		compact_t * vs = (compact_t *)receive;
		voff_t * local_offsets = extra_send_offsets;
		voff_t * local_send_offsets = tmp_send_offsets_th + (thid+1) * (num_of_partitions+1);
		if (intra_inter)
		{
			int pindex = id2index[pid];
			buf = (query_t *)send + receive_start + send_offsets[pindex] + thid * size_per_th;
		}
		else
		{
			int pindex = id2index[pid];
			buf = (query_t *)send + receive_start + receive_offsets[pindex] + thid * size_per_th;
		}
		vid_t * local_fjid = fjid + index_offset;
		vid_t * local_bjid = bjid + index_offset;
		voff_t * local_fwd = fwd + index_offset;
		voff_t * local_bwd = bwd + index_offset;

		voff_t r;
		for (r=0; r<size_th; r++)
		{
			query_t tmp = buf[r];
			vid_t vindex = tmp.nid  - id_offsets[pid] - jid_offset[pid];
			voff_t local_send_offset;
			if (local_fjid[vindex] != tmp.jid && local_bjid[vindex] != tmp.jid)
			{
				dst_not_found++;
				printf ("error!!!: pid = %d: cannot find destination!tmp.jid=%lu, local_fjid=%u, local_bjid=%u\n", \
				pid, tmp.jid, local_fjid[vindex], local_bjid[vindex]);
			}
			if (local_fjid[vindex] == tmp.jid || local_bjid[vindex] == tmp.jid)
			{
				int jpid = query_partition_id_from_idoffsets (tmp.jid, num_of_partitions, id_offsets);
				int pindex = id2index[jpid];
				local_send_offset = local_send_offsets[pindex+1]++;
				off_t off = local_send_offset + send_offsets_th[thid*(num_of_partitions+1)+(pindex+1)] + local_offsets[pindex];
				vs[off].nid = tmp.nid;
				vs[off].jid = tmp.jid;
				if (local_fjid[vindex] == tmp.jid)
				{
					vs[off].ojid = local_bjid[vindex];
#ifdef CONTIG_CHECK
					if (local_fwd[vindex] != 1)
						printf ("record length error: there must be a loop with junction fjid=%lu, bjid=%lu!!!\n", local_fjid[vindex], local_bjid[vindex]);
#endif
				}
				else
				{
					vs[off].ojid = local_fjid[vindex];
#ifdef CONTIG_CHECK
					if (local_bwd[vindex] != 1)
						printf ("record length error: there must be a loop with junction fjid=%lu, bjid=%lu!!!\n", local_fjid[vindex], local_bjid[vindex]);
#endif
				}
				vs[off].plen = local_fwd[vindex] + local_bwd[vindex];
			}
		}
	}
}

// ********* this is compute among junctions *************
static void pull_update_cpu (uint num_mssgs, int pid, voff_t jindex_offset, voff_t jnb_index_offset, void * local_receive, bool intra_inter, int k)
{
	int pindex = id2index[pid];
	compact_t * vs;
	if (intra_inter) // true if intra partitions
	{
		vs = (compact_t *)local_receive + extra_send_offsets[pindex];
	}
	else
	{
		vs = (compact_t *)local_receive + receive_offsets[pindex];
	}
	voff_t * local_joffs = jnb_offsets + jindex_offset + pindex;
	voff_t * local_nbs = jnbs + jnb_index_offset;
	size_t * local_ulens = ulens + jnb_index_offset;

	voff_t r;
#pragma omp parallel for num_threads(cpu_threads)
	for (r=0; r <num_mssgs; r++)
	{
		int index = r;
		vid_t nb = vs[index].nid;
		vid_t jindex = vs[index].jid - id_offsets[pid];
		int num = local_joffs[jindex+1] - local_joffs[jindex];
		if (vs[index].plen > 300)
		{
			count++;
//			printf ("LONG CONTIG LENGTH %d\n\n", vs[index].plen);
		}
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
#ifdef CONTIG_CHECK
	printf ("Unitig longer than 300: %u\n", count);
#endif
}

//**************** this is update with ulens with junctions *************
static void update_ulens_with_kplus1 (uint size, int pid, voff_t jindex_offset, voff_t jnb_index_offset, int k, int total_num_partitions)
{
	int pindex = id2index[pid];
	voff_t * local_joffs = jnb_offsets + jindex_offset + pindex;
	voff_t * local_nbs = jnbs + jnb_index_offset;
	size_t * local_ulens = ulens + jnb_index_offset;

	voff_t i;
#pragma parallel for
	for (i=0; i<size; i++)
	{
		voff_t nb_off = local_joffs[i];
		voff_t num_nbs = local_joffs[i+1] - local_joffs[i];
		vid_t jid = id_offsets[pid] + i;
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
			int nb_pid = query_partition_id_from_idoffsets (nbid, total_num_partitions, id_offsets);
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
static void push_mssg_offset_contig_cpu (uint size, int pid, voff_t index_offset, int num_of_partitions)
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

		voff_t * local_send_offsets = send_offsets_th + (thid+1) * (num_of_partitions+1);

		vid_t * local_fjid = fjid + index_offset + thid * size_per_th;
		vid_t * local_bjid = bjid + index_offset + thid * size_per_th;

		voff_t r;
		for (r=0; r<size_th; r++)
		{
			voff_t index = r;
			int fpid = query_partition_id_from_idoffsets (local_fjid[index], num_of_partitions, id_offsets);
			int bpid = query_partition_id_from_idoffsets (local_bjid[index], num_of_partitions, id_offsets);
			vid_t jid;
			int jpid;
			if (local_fjid[index] >= id_offsets[fpid] + jid_offset[fpid] && local_bjid[index] >= id_offsets[bpid] + jid_offset[bpid])
			{
				printf ("CPU: NONE JUNCTION END POINT FOUND:: pid=%d, index=%u, fid=%u, bid=%u!\n", pid, index, local_fjid[index], local_bjid[index]); // a cycle here, maybe
				continue;
			}
			else if (local_fjid[index] < id_offsets[fpid] + jid_offset[fpid] && local_bjid[index] < id_offsets[bpid] + jid_offset[bpid])
			{
				if (local_fjid[index] <= local_bjid[index])
				{
					jid = local_fjid[index];
					jpid = fpid;
				}
				else
				{
					jid = local_bjid[index];
					jpid = bpid;
				}
			}
			else if (local_fjid[index] < id_offsets[fpid] + jid_offset[fpid])
			{
				jid = local_fjid[index];
				jpid = fpid;
			}
			else
			{
				jid = local_bjid[index];
				jpid = bpid;
			}
			int pindex = id2index[jpid];
			local_send_offsets[pindex+1]++;
		}
	}
}

// ********* this is compute among linear vertices *************
static void push_mssg_contig_cpu (uint size, int pid, voff_t index_offset, int num_of_partitions)
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

		voff_t * local_send_offsets = tmp_send_offsets_th + (thid+1) * (num_of_partitions+1);

		vid_t * local_fjid = fjid + index_offset + thid * size_per_th;
		vid_t * local_bjid = bjid + index_offset + thid * size_per_th;
		voff_t * local_fwd = fwd + index_offset + thid * size_per_th;
		voff_t * local_bwd = bwd + index_offset + thid * size_per_th;
		edge_type * local_post_edge = post_edges + index_offset + thid * size_per_th;
		edge_type * local_pre_edge = pre_edges + index_offset + thid * size_per_th;
		unitig_t * buf = (unitig_t *)send;

		voff_t r;
		for (r=0; r<size_th; r++)
		{
			voff_t index = r;
			int fpid = query_partition_id_from_idoffsets (local_fjid[index], num_of_partitions, id_offsets);
			int bpid = query_partition_id_from_idoffsets (local_bjid[index], num_of_partitions, id_offsets);
			vid_t jid;
			int jpid;
			if (local_fjid[index] >= id_offsets[fpid] + jid_offset[fpid] && local_bjid[index] >= id_offsets[bpid] + jid_offset[bpid])
				continue;
			else if (local_fjid[index] < id_offsets[fpid] + jid_offset[fpid] && local_bjid[index] < id_offsets[bpid] + jid_offset[bpid])
			{
				if (local_fjid[index] <= local_bjid[index])
				{
					jid = local_fjid[index];
					jpid = fpid;
				}
				else
				{
					jid = local_bjid[index];
					jpid = bpid;
				}
			}
			else if (local_fjid[index] < id_offsets[fpid] + jid_offset[fpid])
			{
				jid = local_fjid[index];
				jpid = fpid;
			}
			else
			{
				jid = local_bjid[index];
				jpid = bpid;
			}
			int pindex = id2index[jpid];
			voff_t local_offset = local_send_offsets[pindex+1]++;
			voff_t off = local_offset + send_offsets_th[thid*(num_of_partitions+1)+(pindex+1)] + send_offsets[pindex];
			if (jid == local_fjid[index])
			{
				buf[off].jid = local_fjid[index];
				buf[off].ojid = local_bjid[index];
				if (buf[off].jid > id_offsets[jpid]+jid_offset[jpid])
				{
					printf ("EEEEEEEEror!\n");
				}
				buf[off].rank = local_fwd[index] | (local_pre_edge[index] << (sizeof(voff_t)*CHAR_BITS) - 2);
				buf[off].len = local_fwd[index] + local_bwd[index];
			}
			else if (jid == local_bjid[index])
			{
				buf[off].jid = local_bjid[index];
				buf[off].ojid = local_fjid[index];
				if (buf[off].jid > id_offsets[jpid]+jid_offset[jpid])
				{
					printf ("EEEEEEEEror!\n");
				}
				buf[off].rank = local_bwd[index] | (local_post_edge[index] << (sizeof(voff_t)*CHAR_BITS) - 2);
				buf[off].len = local_fwd[index] + local_bwd[index];
			}
			else
			{
				printf ("Please find the error!!!!!!!!!!!\n");
			}
		}
	}
}

// ********* this is compute among junctions *************
static void pull_mssg_contig_cpu (uint num_mssgs, int pid, voff_t jindex_offset, voff_t jnb_index_offset, voff_t receive_start, bool intra_inter, int k)
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

	int r;
#pragma omp parallel for num_threads(cpu_threads)
	for (r=0; r <num_mssgs; r++)
	{
		int index = r;
		vid_t jindex = buf[index].jid - id_offsets[pid];
		if (jindex > jid_offset[pid])
		{
			printf ("EEEEEEerror on CPU!\n");
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
			size_t unitig_off;
			size_t length;
			if (nb_off == 0)
			{
				unitig_off = 0;
				length = local_ulens[0];
			}
			else
			{
				unitig_off = local_ulens[nb_off-1];
				length = local_ulens[nb_off] - local_ulens[nb_off-1];
			}
			if(local_nbs[nb_off] == nb && length-k == len)
			{
				if (buf[index].jid <= nb && length-k-1 >= rank)
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

static void get_kmer_string (kmer_t * kmer, int k, char * str)
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
static int complete_first_2bps (kmer_t * kmer, int k, ull * edge, char * unitig, int nb_num)
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

static void complete_contig_with_junction_cpu (uint size, int pid, voff_t jindex_offset, voff_t jnb_index_offset, int k)
{
	int pindex = id2index[pid];
	voff_t * local_joffs = jnb_offsets + jindex_offset + pindex;
	ull * local_edges = junct_edges + jindex_offset;
	kmer_t * local_kmer = jkmers + jindex_offset;
	voff_t * local_nbs = jnbs + jnb_index_offset;
	size_t * local_ulens = ulens + jnb_index_offset;
	char * local_unitigs = unitigs + unitig_offsets[pindex];

//	printf ("total unitig size: %lu!!!\n", local_ulens[local_joffs[size] - 1]);
//	printf ("total number of unitigs: %u!!!\n", local_joffs[size]);

	int r;
#pragma omp parallel for num_threads(cpu_threads)
	for (r=0; r <size; r++)
	{
		int index = r;
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
				if (complete_first_2bps (local_kmer+index, k, local_edges+index, local_unitigs+unitig_off, j) == 0)
				{
#ifdef CONTIG_CHECK
					printf ("complete contig with junction kmer error!\n");
#endif
				}
			}
		}
	}
}

void * compact_push_update_intra_push_cpu (void * arg)
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
		printf ("WORLD RANK %d: compact dbgraph push cpu:\n", mst->world_rank);

	gettimeofday (&overs, NULL);
	int total_num_partitions = mst->total_num_partitions;
	int num_of_partitions = mst->num_partitions[did + 1] - mst->num_partitions[did];
	int poffset = mst->num_partitions[did];
	voff_t * lindex_offset = mst->index_offset[did];
	voff_t * jindex_offset = mst->jindex_offset[did];
	voff_t * jnb_index_offset = mst->jnb_index_offset[did];

	memset(cm->send_offsets, 0, sizeof(voff_t) * (total_num_partitions + 1));
	memset (send_offsets_th, 0, sizeof(voff_t) * (total_num_partitions+1) * (cpu_threads+1));

#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	int i;
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset+i];
		voff_t size = mst->jid_offset[pid];
		push_mssg_offset_compact_cpu (size, total_num_partitions, pid, jindex_offset[i], jnb_index_offset[i]);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	push_offset_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "***************** PUSH MSSG OFFSET FOR CPU *COMPACT* INTRA PROCESSOR TIME: ");
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
		voff_t size = mst->jid_offset[pid];
		push_mssg_compact_cpu (size, total_num_partitions, pid, jindex_offset[i], jnb_index_offset[i]);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	push_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "***************** PUSH MSSG FOR CPU *CAMPACT* INTRA PROCESSOR TIME: ");
#endif

	memcpy(mst->roff[did], cm->send_offsets, sizeof(voff_t)*(total_num_partitions + 1));
	voff_t inter_start = mst->roff[did][num_of_partitions];
	voff_t inter_end = mst->roff[did][total_num_partitions];
	mst->receive[did] = (query_t *)cm->send+inter_start;

	memset (cm->extra_send_offsets, 0, sizeof(voff_t) * (total_num_partitions+1));
	memset (send_offsets_th, 0, sizeof(voff_t) * (total_num_partitions+1) * (cpu_threads+1));

#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset + i];
		voff_t num_mssgs = cm->send_offsets[i+1] - cm->send_offsets[i];
		push_update_offset_cpu (num_mssgs, pid, lindex_offset[i], total_num_partitions, 0, 1);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	pull_intra_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "*********** PUSH *UPDATE* OFFSET CPU INTRA PROCESSOR TIME: ");
#endif

	gettimeofday (&overe, NULL);
	over_time[did] += (float)((overe.tv_sec * 1000000 + overe.tv_usec) - (overs.tv_sec * 1000000 + overs.tv_usec)) / 1000;

	return ((void*) 0);
}

void * compact_pull_update_inter_push_intra_pull_cpu (void * arg)
{
	evaltime_t overs, overe;
	evaltime_t start, end;
	lr_arg * carg = (lr_arg *) arg;
	comm_t * cm = carg->cm;
	master_t * mst = carg->mst;
	int num_of_cpus = mst->num_of_cpus;
	int num_of_devices = mst->num_of_devices;
	int did = carg->did;
	int k = carg->k;
	int total_num_partitions = mst->total_num_partitions;
	int num_of_partitions = mst->num_partitions[did + 1] - mst->num_partitions[did];
	int poffset = mst->num_partitions[did];
	voff_t * lindex_offset = mst->index_offset[did];
	voff_t * jindex_offset = mst->jindex_offset[did];
	voff_t * jnb_index_offset = mst->jnb_index_offset[did];
	if (mst->world_rank == 0)
		printf ("WORLD RANK %d: Compact pull junction update push cpu:\n", mst->world_rank);

	gettimeofday (&overs, NULL);
	voff_t receive_start = cm->send_offsets[num_of_partitions];

	memcpy(cm->receive_offsets, mst->soff[did], sizeof(voff_t) * (num_of_partitions + 1));
	voff_t inter_size = cm->receive_offsets[num_of_partitions];
	memcpy((query_t*)(cm->send) + receive_start, mst->send[did], sizeof(query_t) * inter_size);

#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	int i;
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset + i];
		voff_t num_mssgs = cm->receive_offsets[i+1] - cm->receive_offsets[i];
		push_update_offset_cpu (num_mssgs, pid, lindex_offset[i], total_num_partitions, receive_start, 0);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	pull_inter_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "*********** PUSH UPDATE OFFSET CPU INTER PROCESSORS TIME: ");
#endif

	get_global_offsets (cm->extra_send_offsets, send_offsets_th, total_num_partitions, (cpu_threads+1));
	inclusive_prefix_sum (cm->extra_send_offsets, total_num_partitions + 1);

	// *************** malloc (send and) receive buffer for pull and push mode, this is for junctions???????
	voff_t rcv_size = cm->extra_send_offsets[num_of_partitions];
	double remote_factor;
	voff_t init_mssg_size;
	if (rcv_size == 0)
	{
		printf ("WORLD RANK %d: CPU:: CCCCCCCCCcccareful:::::::::: receive size from intra junction update push is 0!!!!!!!!\n", mst->world_rank);
		remote_factor = 1;
	}
	else
	{
		remote_factor = ceil((float)(jnb_index_offset[num_of_partitions] - rcv_size)/rcv_size);
	}
	init_mssg_size = (jnb_index_offset[num_of_partitions] - rcv_size) > rcv_size ? (jnb_index_offset[num_of_partitions] - rcv_size) : rcv_size;
	malloc_pull_push_receive_cpu (cm, sizeof(compact_t), did, init_mssg_size, ((int)remote_factor) * 2);
	set_pull_push_receive (cm);
	printf ("############### WORLD RANK %d:: local number of messages: %u, remote_factor=%f, "
			"number of slots malloced for receive buffer of device %d: %u\n", mst->world_rank, rcv_size, remote_factor, did, cm->temp_size/sizeof(compact_t));

	memset (tmp_send_offsets_th, 0, sizeof(voff_t) * (total_num_partitions+1) * (cpu_threads+1));

#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset+i];
		voff_t num_mssgs = cm->send_offsets[i+1] - cm->send_offsets[i];
		push_update_cpu (num_mssgs, pid, lindex_offset[i], total_num_partitions, 0, 1);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	push_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "************** PUSH UPDATE CPU INTRA PROCESSOR TIME: ");
#endif

#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset+i];
		voff_t num_mssgs = cm->receive_offsets[i+1] - cm->receive_offsets[i];
		push_update_cpu (num_mssgs, pid, lindex_offset[i], total_num_partitions, receive_start, 0);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	push_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "************** PUSH UPDATE CPU INTER PROCESSORS TIME: ");
#endif

	memcpy(mst->roff[did], cm->extra_send_offsets, sizeof(voff_t)*(total_num_partitions + 1));
	voff_t inter_start = cm->extra_send_offsets[num_of_partitions];
	voff_t inter_end = cm->extra_send_offsets[total_num_partitions];

	mst->receive[did] = (compact_t*)(cm->receive) + inter_start;

#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset + i];
		voff_t num_mssgs = cm->extra_send_offsets[i+1] - cm->extra_send_offsets[i];
		pull_update_cpu (num_mssgs, pid, jindex_offset[i], jnb_index_offset[i], cm->receive, 1, k);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	pull_intra_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "***************** PULL UPDATE CPU INTRA PROCESSOR TIME: ");
#endif

	gettimeofday (&overe, NULL);
	over_time[did] += (float)((overe.tv_sec * 1000000 + overe.tv_usec) - (overs.tv_sec * 1000000 + overs.tv_usec)) / 1000;

	return ((void *) 0);
}

void * update_inter_pull_cpu (void * arg)
{
	evaltime_t overs, overe;
	evaltime_t start, end;
	lr_arg * carg = (lr_arg *) arg;
	comm_t * cm = carg->cm;
	master_t * mst = carg->mst;
	int did = carg->did;
	int k = carg->k;
	int num_of_partitions = mst->num_partitions[did + 1] - mst->num_partitions[did];
	int total_num_partitions = mst->total_num_partitions;
	int poffset = mst->num_partitions[did];
	voff_t * lindex_offset = mst->index_offset[did];
	voff_t * jindex_offset = mst->jindex_offset[did];
	voff_t * jnb_index_offset = mst->jnb_index_offset[did];
	if (mst->world_rank == 0)
		printf ("WORLD RANK %d: Junction update inter pull cpu:\n", mst->world_rank);


	gettimeofday (&overs, NULL);
	voff_t receive_start = cm->extra_send_offsets[num_of_partitions];
	memcpy(cm->receive_offsets, mst->soff[did], sizeof(voff_t) * (num_of_partitions+1));
	voff_t inter_size = cm->receive_offsets[num_of_partitions];
	memcpy((compact_t *)(cm->receive) + receive_start, mst->send[did], sizeof(compact_t) * inter_size);

	if (cm->temp_size < (inter_size+receive_start)*sizeof(compact_t))
	{
		printf("WORLD RANK %d: CPU:: Error:::::::: malloced receive buffer size smaller than actual receive buffer size!\n", mst->world_rank);
		exit(0);
	}

#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	int i;
	for (i=0; i<num_of_partitions; i++)
	{
		if (inter_size == 0)
			break;
		int pid = mst->partition_list[poffset + i];
		voff_t num_mssgs = cm->receive_offsets[i+1] - cm->receive_offsets[i];
		pull_update_cpu (num_mssgs, pid, jindex_offset[i], jnb_index_offset[i], cm->receive + sizeof(compact_t) * receive_start, 0, k);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	pull_inter_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "***************** PULL *UPDATE* CPU INTER PROCESSORS TIME: ");
#endif

//	printf ("+++++++++ DST NOT FOUND IN FIRST ITERATION: %u\n", dst_not_found);
	gettimeofday (&overe, NULL);
	over_time[did] += (float)((overe.tv_sec * 1000000 + overe.tv_usec) - (overs.tv_sec * 1000000 + overs.tv_usec)) / 1000;

	// *************** free (send and) receive buffer for pull and push mode
	free_pull_push_receive_cpu(cm);

	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset + i];
		voff_t size = mst->jid_offset[pid];
		update_ulens_with_kplus1 (size, pid, jindex_offset[i], jnb_index_offset[i], k, total_num_partitions);
	}

	return ((void *) 0);
}

void * gather_contig_push_intra_pull_cpu (void * arg)
{
	evaltime_t overs, overe;
	evaltime_t start, end;
	lr_arg * carg = (lr_arg *) arg;
	comm_t * cm = carg->cm;
	master_t * mst = carg->mst;
	int num_of_cpus = mst->num_of_cpus;
	int num_of_devices = mst->num_of_devices;
	int did = carg->did;
	int k = carg->k;
	if (mst->world_rank == 0)
		printf ("WORLD RANK %d: Gather contig push intra pull cpu:\n", mst->world_rank);

	gettimeofday (&overs, NULL);
	int total_num_partitions = mst->total_num_partitions;
	int num_of_partitions = mst->num_partitions[did + 1] - mst->num_partitions[did];
	int poffset = mst->num_partitions[did];
	voff_t * lindex_offset = mst->index_offset[did];
	voff_t * jindex_offset = mst->jindex_offset[did];
	voff_t * jnb_index_offset = mst->jnb_index_offset[did];

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
		push_mssg_offset_contig_cpu (size, pid, lindex_offset[i], total_num_partitions);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	push_offset_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "***************** PUSH MSSG OFFSET FOR CPU GATHERING CONTIGS TIME: ");
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
		push_mssg_contig_cpu (size, pid, lindex_offset[i], total_num_partitions);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	push_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "***************** PUSH MSSG FOR CPU GATHERING CONTIG TIME: ");
#endif

	memcpy(mst->roff[did], cm->send_offsets, sizeof(voff_t)*(total_num_partitions + 1));
	voff_t inter_start = mst->roff[did][num_of_partitions];
	voff_t inter_end = mst->roff[did][total_num_partitions];
	mst->receive[did] = (unitig_t *)cm->send+inter_start;

#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset + i];
		voff_t num_mssgs = cm->send_offsets[i+1] - cm->send_offsets[i];
		pull_mssg_contig_cpu (num_mssgs, pid, jindex_offset[i], jnb_index_offset[i], 0, 1, k);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	pull_intra_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "*********** CPU PULL MSSG FOR GATHERING CONTIGS INTRA PROCESSOR TIME: ");
#endif

	gettimeofday (&overe, NULL);
	over_time[did] += (float)((overe.tv_sec * 1000000 + overe.tv_usec) - (overs.tv_sec * 1000000 + overs.tv_usec)) / 1000;

	return ((void*) 0);
}

void * gather_contig_inter_pull_cpu (void * arg)
{
	evaltime_t overs, overe;
	evaltime_t start, end;
	lr_arg * carg = (lr_arg *) arg;
	comm_t * cm = carg->cm;
	master_t * mst = carg->mst;
	int k = carg->k;
	int did = carg->did;
	int num_of_partitions = mst->num_partitions[did + 1] - mst->num_partitions[did];
	int poffset = mst->num_partitions[did];
	voff_t * jindex_offset = mst->jindex_offset[did];
	voff_t * jnb_index_offset = mst->jnb_index_offset[did];
	if (mst->world_rank == 0)
		printf ("WORLD RANK %d: gather contig inter pull cpu:\n", mst->world_rank);

	gettimeofday (&overs, NULL);
	voff_t receive_start = cm->send_offsets[num_of_partitions];
	memcpy(cm->receive_offsets, mst->soff[did], sizeof(voff_t) * (num_of_partitions+1));
	voff_t inter_size = cm->receive_offsets[num_of_partitions];
	memcpy((unitig_t *)(cm->send) + receive_start, mst->send[did], sizeof(unitig_t) * inter_size);

#ifdef MEASURE_TIME_
	gettimeofday (&start, NULL);
#endif
	int i;
	for (i=0; i<num_of_partitions; i++)
	{
		if(inter_size == 0)
			break;
		int pid = mst->partition_list[poffset + i];
		voff_t num_mssgs = cm->receive_offsets[i+1] - cm->receive_offsets[i];
		pull_mssg_contig_cpu (num_mssgs, pid, jindex_offset[i], jnb_index_offset[i], receive_start, 0, k);
	}
#ifdef MEASURE_TIME_
	gettimeofday (&end, NULL);
	pull_inter_time[did] += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	print_exec_time (start, end, "***************** CPU PULL MSSG FOR GATHERING CONTIGS INTER PROCESSORS TIME: ");
#endif

	gettimeofday (&overe, NULL);
	over_time[did] += (float)((overe.tv_sec * 1000000 + overe.tv_usec) - (overs.tv_sec * 1000000 + overs.tv_usec)) / 1000;

	for (i=0; i<num_of_partitions; i++)
	{
		int pid = mst->partition_list[poffset + i];
		voff_t size = mst->jid_offset[pid];
		complete_contig_with_junction_cpu (size, pid, jindex_offset[i], jnb_index_offset[i], k);
	}
	return ((void *) 0);
}
