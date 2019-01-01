/*
 * malloc.cuh
 *
 *  Created on: 2018-9-24
 *      Author: qiushuang
 */

#ifndef MALLOC_CUH_
#define MALLOC_CUH_

#include "include/malloc.h"

static cudaStream_t streams[NUM_OF_DEVICES];
extern voff_t * tmp_counts[NUM_OF_PROCS];
extern double mssg_factor;
extern double junction_factor;

//#define BINARY_GPU_

#define DEVICE_SHIFT 0

extern "C"
{
static void init_device_filter1 (dbmeta_t * dbm, master_t * mst, vid_t max_subsize)
{
	int num_of_devices = mst->num_of_devices;
	int total_num_partitions = mst->total_num_partitions;
	int mssg_size = mst->mssg_size;  //******** SEE HERE TIMES 2 FOR BIDIRECTED EDGES *********
	if (mssg_factor == 0)
		mssg_factor = MSSG_FACTOR + 1;
	int i;
#ifdef SINGLE_NODE
	int world_rank = mst->world_rank;
#endif
	for (i = 0; i < num_of_devices; i++)
	{
		vid_t gsize = mst->num_vs[i] * BINARY_FACTOR;
		if (mst->world_size == 1)
			printf ("max_subsize = %u, send buffer malloced: %lu\n", max_subsize, (ull)(sizeof(entry_t)) * max_subsize);
#ifdef SINGLE_NODE
		cudaSetDevice(world_rank * num_of_devices + i);
#else
		cudaSetDevice(i + DEVICE_SHIFT);
#endif
//		CUDA_CHECK_RETURN (cudaMalloc (&dbm[i].sorted_kmers, sizeof(ull) * gsize));
		CUDA_CHECK_RETURN (cudaMalloc (&dbm[i].sorted_kmers, sizeof(kmer_t) * gsize));
		CUDA_CHECK_RETURN (cudaMalloc (&dbm[i].sorted_vids, sizeof(vid_t) * gsize));
		CUDA_CHECK_RETURN (cudaMalloc (&dbm[i].nds.edge, sizeof(ull) * gsize));
		dbm[i].before_sort = NULL; // when constructed dbgraph output sorted vertices
		dbm[i].before_vids = NULL; // when constructed dbgraph output sorted vertices
#ifdef USE_DISK_IO
		CUDA_CHECK_RETURN (cudaMalloc (&dbm[i].before_sort, sizeof(kmer_t) * max_subsize));//temporary space
		CUDA_CHECK_RETURN (cudaMalloc (&dbm[i].before_vids, sizeof(vid_t) * max_subsize));//temporary space
		CUDA_CHECK_RETURN (cudaMemset (dbm[i].before_sort, 0, sizeof(kmer_t) * max_subsize));
		CUDA_CHECK_RETURN (cudaMemset (dbm[i].before_vids, 0, sizeof(vid_t) * max_subsize));
#endif

		CUDA_CHECK_RETURN (cudaMemset (dbm[i].sorted_kmers, 0, sizeof(kmer_t) * gsize));
		CUDA_CHECK_RETURN (cudaMemset (dbm[i].sorted_vids, 0, sizeof(vid_t) * gsize));
		CUDA_CHECK_RETURN (cudaMemset (dbm[i].nds.edge, 0, sizeof(ull) * gsize));
		ull send_size = sizeof(entry_t) * max_subsize > (ull)mssg_size * gsize * mssg_factor? \
				sizeof(entry_t) * max_subsize : (ull)mssg_size * gsize * mssg_factor;
		// this is an estimation of the send buffer size: be careful:: if the number of junctions is large, mssg_size should times value larger than 2
		CUDA_CHECK_RETURN (cudaMalloc (&dbm[i].comm.send, send_size));

		CUDA_CHECK_RETURN (cudaMalloc (&dbm[i].lvld, sizeof(uint) * (max_subsize+1)));
		CUDA_CHECK_RETURN (cudaMalloc (&dbm[i].id_offsets, sizeof(vid_t) * (total_num_partitions + 1)));
		CUDA_CHECK_RETURN (cudaMalloc (&dbm[i].jid_offset, sizeof(vid_t) * total_num_partitions));

//		CUDA_CHECK_RETURN (cudaMemset (dbm[i].lvld, 0, sizeof(uint) * (max_subsize+1)));

		int j;
		for (j=0; j<EDGE_DIC_SIZE; j++)
		{
			CUDA_CHECK_RETURN (cudaMalloc (&dbm[i].nb[j], sizeof(vid_t) * gsize));
		}

		int num_partitions_device = mst->num_partitions[i+1] - mst->num_partitions[i];
		mst->index_offset[i] = (voff_t *) malloc (sizeof(voff_t) * (num_partitions_device+1));
		CHECK_PTR_RETURN (mst->index_offset[i], "malloc index_offset in master for device %d error!\n", i);
		memset(mst->index_offset[i], 0, sizeof(voff_t) * (num_partitions_device+1));
	}
}

static void init_device_filter2 (dbmeta_t * dbm, master_t * mst, vid_t max_subsize)
{
	int num_of_devices = mst->num_of_devices;

	int i;
#ifdef SINGLE_NODE
	int world_rank = mst->world_rank;
#endif
	for (i = 0; i < num_of_devices; i++)
	{
//		int num_partitions_device = mst->num_partitions[i+1] - mst->num_partitions[i];
//		vid_t gsize = mst->index_offset[i][num_partitions_device];
		if (mst->world_size == 1)
			printf ("max_subsize = %u\n", max_subsize);
#ifdef SINGLE_NODE
		cudaSetDevice(world_rank * num_of_devices + i);
#else
		cudaSetDevice(i + DEVICE_SHIFT);
#endif
		CUDA_CHECK_RETURN (cudaFree (dbm[i].before_sort));
		CUDA_CHECK_RETURN (cudaFree (dbm[i].before_vids));
//		CUDA_CHECK_RETURN (cudaFree (dbm[i].lvld));
//		CUDA_CHECK_RETURN (cudaFree (dbm[i].comm.send));

/*		int j;
		for (j=0; j<EDGE_DIC_SIZE; j++)
		{
			CUDA_CHECK_RETURN (cudaMalloc (&dbm[i].nb[j], sizeof(vid_t) * gsize));
		}*/

		CUDA_CHECK_RETURN (cudaMalloc (&dbm[i].jvld, sizeof(voff_t) * (max_subsize+1)));
//		CUDA_CHECK_RETURN (cudaMalloc (&dbm[i].lvld, sizeof(voff_t) * (gsize+1)));

//		CUDA_CHECK_RETURN (cudaMemset (dbm[i].jvld, 0, sizeof(voff_t) * (gsize+1)));
//		CUDA_CHECK_RETURN (cudaMemset (dbm[i].lvld, 0, sizeof(voff_t) * (gsize+1)));
	}
}

static void finalize_device_filter2 (dbmeta_t *dbm, master_t *mst)
{
	int num_of_devices = mst->num_of_devices;
#ifdef SINGLE_NODE
	int world_rank = mst->world_rank;
#endif
	int i;
	for (i=0; i<num_of_devices; i++)
	{
#ifdef SINGLE_NODE
		cudaSetDevice(world_rank * num_of_devices + i);
#else
		cudaSetDevice(i + DEVICE_SHIFT);
#endif
		CUDA_CHECK_RETURN (cudaFree (dbm[i].jvld));
		CUDA_CHECK_RETURN (cudaFree (dbm[i].lvld));
	}
}

static void init_device_preprocessing (dbmeta_t * dbm, master_t * mst)
{
		int num_of_devices = mst->num_of_devices;
		int total_num_partitions = mst->total_num_partitions;
		int mssg_size = mst->mssg_size;  //******** SEE HERE TIMES 2 FOR BIDIRECTED EDGES *********
		if (mssg_factor == 0)
			mssg_factor = MSSG_FACTOR + 1;
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
			CUDA_CHECK_RETURN (cudaStreamCreate(&streams[i]));
	//		int num_of_partitions = mst->num_partitions[i + 1] - mst->num_partitions[i];
			vid_t gsize = mst->num_vs[i] * BINARY_FACTOR;
			int num_partitions_device = mst->num_partitions[i+1] - mst->num_partitions[i];
//			vid_t gsize = mst->index_offset[i][num_partitions_device];
			if (gsize > MAX_NUM_NODES_DEVICE_HASH)
			{
				printf ("ERROR: graph size greater than maximum number of vertices in a device!\n");
				// exit(0);
			}
			if (mst->world_size == 1)
				printf ("++++++++++++++++++++++++\nnumber of vertices in device %d: %u\nmssg size: %d, message buffer size = %lu\n", \
					i, gsize, mst->mssg_size, (ull)mssg_size * gsize);

			CUDA_CHECK_RETURN (cudaMalloc (&dbm[i].comm.id2index, sizeof(int) * total_num_partitions));
			CUDA_CHECK_RETURN (cudaMalloc (&dbm[i].comm.receive_offsets, sizeof(voff_t) * (total_num_partitions + 1)));
			CUDA_CHECK_RETURN (cudaMalloc (&dbm[i].comm.send_offsets, sizeof(voff_t) * (total_num_partitions + 1)));
			CUDA_CHECK_RETURN (cudaMalloc (&dbm[i].comm.tmp_send_offsets, sizeof(voff_t) * (total_num_partitions + 1)));
			CUDA_CHECK_RETURN (cudaMemset (dbm[i].comm.receive_offsets, 0, sizeof(voff_t) * (total_num_partitions + 1)));
			CUDA_CHECK_RETURN (cudaMemset (dbm[i].comm.send_offsets, 0, sizeof(voff_t) * (total_num_partitions + 1)));
			CUDA_CHECK_RETURN (cudaMemset (dbm[i].comm.tmp_send_offsets, 0, sizeof(voff_t) * (total_num_partitions + 1)));
//			CUDA_CHECK_RETURN (cudaMalloc (&dbm[i].comm.send, (ull)mssg_size * gsize * mssg_factor));
			CUDA_CHECK_RETURN (cudaMemcpy (dbm[i].comm.id2index, mst->id2index[i], sizeof(int) * total_num_partitions, cudaMemcpyHostToDevice));

			CUDA_CHECK_RETURN(cudaMallocHost(&mst->roff[i], sizeof(voff_t) * (total_num_partitions+1)));
			CUDA_CHECK_RETURN(cudaMallocHost(&mst->soff[i], sizeof(voff_t) * (total_num_partitions+1)));
			CUDA_CHECK_RETURN(cudaMallocHost(&mst->receive[i], (ull)mssg_size * gsize * mssg_factor));
			CUDA_CHECK_RETURN(cudaMallocHost(&mst->send[i], (ull)mssg_size * gsize * mssg_factor));
			memset (mst->roff[i], 0, sizeof(voff_t) * (total_num_partitions+1));
			memset (mst->soff[i], 0, sizeof(voff_t) * (total_num_partitions+1));

			CUDA_CHECK_RETURN (cudaMalloc (&dbm[i].comm.index_offsets, sizeof(voff_t) * (num_partitions_device+1)));
			CUDA_CHECK_RETURN (cudaMemcpy (dbm[i].comm.index_offsets, mst->index_offset[i], sizeof(voff_t) * (num_partitions_device+1), cudaMemcpyHostToDevice));
			tmp_counts[i] = (voff_t *) malloc (sizeof(voff_t) * num_partitions_device);
		}
}

static void set_id_offsets_gpu (dbmeta_t * dbm, master_t * mst)
{
	int num_of_devices = mst->num_of_devices;
	int total_num_partitions = mst->total_num_partitions;
#ifdef SINGLE_NODE
	int world_rank = mst->world_rank;
#endif

	int i;
	for (i=0; i<num_of_devices; i++)
	{
#ifdef SINGLE_NODE
		cudaSetDevice(world_rank * num_of_devices + i);
#else
		cudaSetDevice(i + DEVICE_SHIFT);
#endif
		CUDA_CHECK_RETURN (cudaMemcpy (dbm[i].id_offsets, mst->id_offsets, sizeof(vid_t) * (total_num_partitions + 1), cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN (cudaMemcpy (dbm[i].jid_offset, mst->jid_offset, sizeof(vid_t) * total_num_partitions, cudaMemcpyHostToDevice));
	}
}

static void finalize_device_preprocessing (dbmeta * dbm, master_t * mst)
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
		CUDA_CHECK_RETURN (cudaStreamDestroy(streams[i]));

		cudaFree (dbm[i].comm.id2index);
		cudaFree (dbm[i].comm.receive_offsets);
		cudaFree (dbm[i].comm.send_offsets);
		cudaFree (dbm[i].comm.tmp_send_offsets);
		cudaFree (dbm[i].comm.send);
		cudaFree (dbm[i].comm.index_offsets);

		cudaFreeHost (mst->roff[i]);
		cudaFreeHost (mst->soff[i]);
		cudaFreeHost (mst->receive[i]);
		cudaFreeHost (mst->send[i]);

		free (tmp_counts[i]);
	}
}

static void init_device_gather (dbmeta_t * dbm, master_t * mst, vid_t max_subsize, vid_t max_jsize, vid_t max_lsize)
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
		CUDA_CHECK_RETURN (cudaMalloc (&dbm[i].jvld, sizeof(voff_t) * (max_subsize + 1)));
		CUDA_CHECK_RETURN (cudaMalloc (&dbm[i].lvld, sizeof(voff_t) * (max_subsize + 1)));
		int j;
		for (j=0; j<EDGE_DIC_SIZE; j++)
		{
			CUDA_CHECK_RETURN (cudaMalloc (&dbm[i].djs.nbs[j], sizeof(vid_t) * max_jsize));
		}
		CUDA_CHECK_RETURN (cudaMalloc (&dbm[i].djs.edges, sizeof(ull) * max_jsize));
		CUDA_CHECK_RETURN (cudaMalloc (&dbm[i].dls.posts, sizeof(vid_t) * max_lsize));
		CUDA_CHECK_RETURN (cudaMalloc (&dbm[i].dls.pres, sizeof(vid_t) * max_lsize));
		CUDA_CHECK_RETURN (cudaMalloc (&dbm[i].dls.post_edges, sizeof(edge_type) * max_lsize));
		CUDA_CHECK_RETURN (cudaMalloc (&dbm[i].dls.pre_edges, sizeof(edge_type) * max_lsize));

		CUDA_CHECK_RETURN (cudaMalloc (&dbm[i].djs.kmers, sizeof(kmer_t) * max_jsize));
//		CUDA_CHECK_RETURN (cudaMalloc (&dbm[i].dls.kmers, sizeof(kmer_t) * max_lsize));

		CUDA_CHECK_RETURN (cudaMemset (dbm[i].jvld, 0, sizeof(voff_t) * (max_subsize + 1)));
		CUDA_CHECK_RETURN (cudaMemset (dbm[i].lvld, 0, sizeof(voff_t) * (max_subsize + 1)));

		ull cpy_size = max_jsize * sizeof(vid_t) * EDGE_DIC_SIZE + max_lsize * sizeof(vid_t) * 2;
		mst->send[i] = (void *) malloc (cpy_size);
		CHECK_PTR_RETURN (mst->send[i], "malloc buffer for transfer adj graph for device %d error!\n", i);
	}
}

static void finalize_device_gather2 (dbmeta_t * dbm, master_t * mst)
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

		cudaFree (dbm[i].sorted_kmers);
		cudaFree (dbm[i].sorted_vids);
		cudaFree (dbm[i].nds.edge);
		cudaFree (dbm[i].djs.edges);
		cudaFree (dbm[i].djs.kmers);
		int j;
		for (j=0; j<EDGE_DIC_SIZE; j++)
		{
			cudaFree (dbm[i].nb[j]);
			cudaFree (dbm[i].djs.nbs[j]);
		}
		cudaFree (dbm[i].dls.posts);
		cudaFree (dbm[i].dls.pres);
		cudaFree (dbm[i].dls.post_edges);
		cudaFree (dbm[i].dls.pre_edges);
		cudaFree (dbm[i].jvld);
		cudaFree (dbm[i].lvld);
		cudaFree (dbm[i].id_offsets);
		cudaFree (dbm[i].jid_offset);

		free (mst->send[i]);
		free (mst->index_offset[i]);
	}
}

static void init_device_graph_compute (meta_t * dm, master_t * mst)
{
	int num_of_devices = mst->num_of_devices;
	int total_num_partitions = mst->total_num_partitions;
	int mssg_size = mst->mssg_size;
	if (mssg_factor == 0)
		mssg_factor = MSSG_FACTOR + 1;
	double contig_mssg_factor = mssg_factor;

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
		CUDA_CHECK_RETURN (cudaStreamCreate(&streams[i]));
		vid_t gsize = mst->num_vs[i];
		if (gsize > MAX_NUM_NODES_DEVICE)
		{
			printf ("ERROR: graph size greater than maximum number of vertices in a device!\n");
			// exit(0);
		}
		if (mst->world_size == 1)
			printf ("++++++++++++++++++++++++\nnumber of vertices in device %d: %u\nmssg size: %d\n", i, gsize, mst->mssg_size);
		CUDA_CHECK_RETURN (cudaMalloc (&dm[i].edge.post, sizeof(vid_t) * gsize));
		CUDA_CHECK_RETURN (cudaMalloc (&dm[i].edge.pre, sizeof(vid_t) * gsize));
		CUDA_CHECK_RETURN (cudaMalloc (&dm[i].edge.bwd, sizeof(voff_t) * gsize));
		CUDA_CHECK_RETURN (cudaMalloc (&dm[i].edge.fwd, sizeof(voff_t) * gsize));
		CUDA_CHECK_RETURN (cudaMalloc (&dm[i].edge.fjid, sizeof(vid_t) * gsize));
		CUDA_CHECK_RETURN (cudaMalloc (&dm[i].edge.bjid, sizeof(vid_t) * gsize));
		CUDA_CHECK_RETURN (cudaMemset (dm[i].edge.fjid, 0, sizeof(vid_t) * gsize));
		CUDA_CHECK_RETURN (cudaMemset (dm[i].edge.bjid, 0, sizeof(vid_t) * gsize));

		CUDA_CHECK_RETURN (cudaMalloc (&dm[i].comm.id2index, sizeof(int) * total_num_partitions));
		CUDA_CHECK_RETURN (cudaMalloc (&dm[i].comm.receive_offsets, sizeof(voff_t) * (total_num_partitions + 1)));
		CUDA_CHECK_RETURN (cudaMalloc (&dm[i].comm.send_offsets, sizeof(voff_t) * (total_num_partitions + 1)));
		CUDA_CHECK_RETURN (cudaMalloc (&dm[i].comm.tmp_send_offsets, sizeof(voff_t) * (total_num_partitions + 1)));
		CUDA_CHECK_RETURN (cudaMemset (dm[i].comm.receive_offsets, 0, sizeof(voff_t) * (total_num_partitions + 1)));
		CUDA_CHECK_RETURN (cudaMemset (dm[i].comm.send_offsets, 0, sizeof(voff_t) * (total_num_partitions + 1)));
		CUDA_CHECK_RETURN (cudaMemset (dm[i].comm.tmp_send_offsets, 0, sizeof(voff_t) * (total_num_partitions + 1)));
		CUDA_CHECK_RETURN (cudaMalloc (&dm[i].comm.send, (ull)mssg_size * gsize * contig_mssg_factor));

		CUDA_CHECK_RETURN (cudaMalloc (&dm[i].id_offsets, sizeof(vid_t) * (total_num_partitions + 1)));
		CUDA_CHECK_RETURN (cudaMalloc (&dm[i].jid_offset, sizeof(vid_t) * total_num_partitions));

		CUDA_CHECK_RETURN (cudaMemcpy (dm[i].comm.id2index, mst->id2index[i], sizeof(int) * total_num_partitions, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN (cudaMemcpy (dm[i].id_offsets, mst->id_offsets, sizeof(vid_t) * (total_num_partitions + 1), cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN (cudaMemcpy (dm[i].jid_offset, mst->jid_offset, sizeof(vid_t) * total_num_partitions, cudaMemcpyHostToDevice));

		CUDA_CHECK_RETURN(cudaMallocHost(&mst->roff[i], sizeof(voff_t) * (total_num_partitions+1)));
		CUDA_CHECK_RETURN(cudaMallocHost(&mst->soff[i], sizeof(voff_t) * (total_num_partitions+1)));
		CUDA_CHECK_RETURN(cudaMallocHost(&mst->receive[i], (ull)mssg_size * gsize * contig_mssg_factor));
		CUDA_CHECK_RETURN(cudaMallocHost(&mst->send[i], (ull)mssg_size * gsize * contig_mssg_factor));
		memset (mst->roff[i], 0, sizeof(voff_t) * (total_num_partitions+1));
		memset (mst->soff[i], 0, sizeof(voff_t) * (total_num_partitions+1));

		int num_partitions_device = mst->num_partitions[i+1] - mst->num_partitions[i];
		mst->index_offset[i] = (voff_t *) malloc (sizeof(voff_t) * (num_partitions_device+1));
		CHECK_PTR_RETURN (mst->index_offset[i], "malloc index_offset in master for device %d error!\n", i);
		memset(mst->index_offset[i], 0, sizeof(voff_t) * (num_partitions_device+1));
		tmp_counts[i] = (voff_t *) malloc (sizeof(voff_t) * num_partitions_device);
//		mst->flag[i] = 0;
	}
}

static void finalize_device_graph_compute (meta_t * dm, master_t * mst)
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
		CUDA_CHECK_RETURN (cudaStreamDestroy(streams[i]));

		cudaFree (dm[i].edge.fwd);
		cudaFree (dm[i].edge.bwd);
		cudaFree (dm[i].edge.fjid);
		cudaFree (dm[i].edge.bjid);

		cudaFree (dm[i].comm.id2index);
		cudaFree (dm[i].comm.receive_offsets);
		cudaFree (dm[i].comm.send_offsets);
		cudaFree (dm[i].comm.tmp_send_offsets);
		cudaFree (dm[i].comm.send);

		cudaFree (dm[i].id_offsets);
		cudaFree (dm[i].jid_offset);

		cudaFreeHost (mst->roff[i]);
		cudaFreeHost (mst->soff[i]);
		cudaFreeHost (mst->receive[i]);
		cudaFreeHost (mst->send[i]);

		free (mst->index_offset[i]);
		free (tmp_counts[i]);
	}
}

static void malloc_pull_push_offset_gpu (voff_t ** extra_send_offsets, master_t * mst, int i)
{
	int total_num_partitions = mst->total_num_partitions;
#ifdef SINGLE_NODE
	int world_rank = mst->world_rank;
	int num_of_devices = mst->num_of_devices;
	CUDA_CHECK_RETURN (cudaSetDevice (world_rank * num_of_devices + i));
#else
	CUDA_CHECK_RETURN (cudaSetDevice (i + DEVICE_SHIFT));
#endif
	CUDA_CHECK_RETURN (cudaMalloc(extra_send_offsets, sizeof(voff_t) * (total_num_partitions+1)));
	CUDA_CHECK_RETURN (cudaMemset (*extra_send_offsets, 0, sizeof(voff_t) * (total_num_partitions+1)));
//		CUDA_CHECK_RETURN (cudaMemcpyToSymbol(extra_send_offsets, &dm[i].comm.extra_send_offsets, sizeof(voff_t *)));
}

static void free_pull_push_offset_gpu (voff_t * extra_send_offsets)
{
	cudaFree(extra_send_offsets);
}

static size_t malloc_pull_push_receive_device (void ** recv, uint mssg_size, int did, voff_t num_of_mssgs, int expand, int world_rank, int num_of_devices)
{
#ifdef SINGLE_NODE
	CUDA_CHECK_RETURN (cudaSetDevice(world_rank * num_of_devices + did));
#else
	CUDA_CHECK_RETURN (cudaSetDevice(did + DEVICE_SHIFT));
#endif
	size_t malloc_size = (ull)mssg_size * num_of_mssgs * expand;
	CUDA_CHECK_RETURN (cudaMalloc(recv, (ull)mssg_size * num_of_mssgs * expand));
//	CUDA_CHECK_RETURN (cudaMemcpyToSymbol(receive, recv, sizeof(void*)));
	return malloc_size;
}

static void free_pull_push_receive_device (int did, comm_t * dm, int world_rank, int num_of_devices)
{
#ifdef SINGLE_NODE
	CUDA_CHECK_RETURN (cudaSetDevice(world_rank * num_of_devices + did));
#else
	CUDA_CHECK_RETURN (cudaSetDevice(did + DEVICE_SHIFT));
#endif
	CUDA_CHECK_RETURN (cudaFree(dm->receive));
}

static void realloc_device_edges (meta_t * dm, master_t * mst)
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
		vid_t lsize = mst->num_vs[i];
		vid_t jsize = mst->num_js[i];
		if (lsize > MAX_NUM_NODES_DEVICE || jsize > MAX_NUM_NODES_DEVICE)
		{
			printf ("ERROR: graph size greater than maximum number of vertices in a device!\n");
			// exit(0);
		}
		if (mst->world_size == 1)
			printf ("++++++++++++++++++++++++\nnumber of junctions in device %d: %u\n"
				"number of linear vertices: %u\nmssg size: %d\n", i, jsize, lsize, mst->mssg_size);
		CUDA_CHECK_RETURN (cudaMalloc(&dm[i].edge.pre_edges, sizeof(edge_type) * lsize));
		CUDA_CHECK_RETURN (cudaMalloc(&dm[i].edge.post_edges, sizeof(edge_type) * lsize));
		CUDA_CHECK_RETURN (cudaMalloc(&dm[i].junct.edges, sizeof(ull) * jsize));

/*		CUDA_CHECK_RETURN (cudaMemcpyToSymbol(pre_edges, &dm[i].edge.pre_edges, sizeof(edge_type*)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol(post_edges, &dm[i].edge.post_edges, sizeof(edge_type*)));
		CUDA_CHECK_RETURN (cudaMemcpyToSymbol(junct_edges, &dm[i].junct.edges, sizeof(ull*)));*/
	}
}

static void realloc_device_junctions (meta_t * dm, master_t * mst)
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
		vid_t gsize = mst->num_js[i];
		vid_t nbsize = mst->num_jnbs[i];
		if (gsize > MAX_NUM_NODES_DEVICE)
		{
			printf ("ERROR: graph size greater than maximum number of vertices in a device!\n");
			// exit(0);
		}
		if (mst->world_size == 1)
			printf ("++++++++++++++++++++++++\nnumber of vertices in device %d: %u\nmssg size: %d\n", i, gsize, mst->mssg_size);

		int num_partitions_device = mst->num_partitions[i+1] - mst->num_partitions[i];

		CUDA_CHECK_RETURN (cudaMalloc (&(dm[i].junct.offs), sizeof(voff_t) * (gsize + num_partitions_device)));
		CUDA_CHECK_RETURN (cudaMalloc (&(dm[i].junct.nbs), sizeof(vid_t) * nbsize));
		CUDA_CHECK_RETURN (cudaMalloc (&(dm[i].junct.kmers), sizeof(kmer_t) * gsize));
		CUDA_CHECK_RETURN (cudaMalloc (&(dm[i].junct.ulens), sizeof(size_t) * nbsize));

		CUDA_CHECK_RETURN (cudaMemset (dm[i].junct.ulens, 0, sizeof(size_t) * nbsize));

		mst->jindex_offset[i] = (voff_t *) malloc (sizeof(voff_t) * (num_partitions_device+1));
		CHECK_PTR_RETURN (mst->jindex_offset[i], "malloc jindex_offset in master for device %d error!\n", i);
		memset(mst->jindex_offset[i], 0, sizeof(voff_t) * (num_partitions_device+1));
		mst->jnb_index_offset[i] = (voff_t *) malloc (sizeof(voff_t) * (num_partitions_device+1));
		CHECK_PTR_RETURN (mst->jnb_index_offset[i], "malloc jnb_index_offset in master for device %d error!\n", i);
		memset(mst->jnb_index_offset[i], 0, sizeof(voff_t) * (num_partitions_device+1));
	}
}

static void free_device_realloc (meta_t * dm, master_t * mst)
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
		cudaFree(dm[i].edge.post);
		cudaFree(dm[i].edge.pre);
		cudaFree(dm[i].edge.pre_edges);
		cudaFree(dm[i].edge.post_edges);
		cudaFree(dm[i].junct.offs);
		cudaFree(dm[i].junct.nbs);
		cudaFree(dm[i].junct.kmers);
		cudaFree(dm[i].junct.ulens);
		cudaFree(dm[i].junct.edges);
		free (mst->jindex_offset[i]);
		free (mst->jnb_index_offset[i]);
	}
}

static void malloc_unitig_buffer_gpu (meta_t * dm, size_t size, int did, int num_of_devices, int num_of_partitions)
{
	cudaSetDevice(did + DEVICE_SHIFT);
#ifdef CONTIG_CHECK
	printf ("GPU %d: unitig buffer size %lu\n", size);
#endif
	CUDA_CHECK_RETURN (cudaMalloc(&dm->junct.unitigs, sizeof(char) * size));
	CUDA_CHECK_RETURN (cudaMalloc(&dm->junct.unitig_offs, sizeof(size_t) * (num_of_partitions+1)));
}

static void free_unitig_buffer_gpu (meta_t * dm, master_t * mst)
{
	int num_of_devices = mst->num_of_devices;
#ifdef SINGLE_NODE
	int world_rank = mst->world_rank;
	int world_size = mst->world_size;
#endif
	int i;
	for (i=0; i<num_of_devices; i++)
	{
#ifdef SINGLE_NODE
		cudaSetDevice(world_rank * num_of_devices + i);
#else
		cudaSetDevice(i + DEVICE_SHIFT);
#endif
		cudaFree (dm[i].junct.unitigs);
	}
}
}

#endif /* MALLOC_CUH_ */
