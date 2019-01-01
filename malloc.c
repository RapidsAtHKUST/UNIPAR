/*
 * malloc.cu
 *
 *  Created on: 2018-4-13
 *      Author: qiushuang
 */

#include "include/malloc.h"

voff_t * tmp_counts[NUM_OF_PROCS];
extern double mssg_factor;
extern double junction_factor;

void malloc_subgraph_subgraphs (subgraph_t * subgraph, int num_of_partitions)
{
	subgraph->subgraphs = (subsize_t *) malloc (sizeof(subsize_t) * num_of_partitions);
	CHECK_PTR_RETURN (subgraph, "malloc subgraph array error!\n");
	subgraph->num_of_subs = num_of_partitions;
	subgraph->total_graph_size = 0;
}

size_t get_total_size_subgraphs (subgraph_t * subgraph)
{
	int i;
	size_t sum=0;
	for (i=0; i<subgraph->num_of_subs; i++)
	{
		sum += (subgraph->subgraphs)[i].size;
	}
	return sum;
}

void free_subgraph_subgraphs (subgraph_t * subgraph)
{
	free (subgraph->subgraphs);
}

void free_subgraph_num_jnbs (subgraph_t * subgraph)
{
	free (subgraph->num_jnbs);
}

void init_host_filter2 (dbmeta_t * dbm, master_t * mst, vid_t max_subsize)
{
	int num_of_cpus = mst->num_of_cpus;
	int num_of_devices = mst->num_of_devices;
	int total_num_partitions = mst->total_num_partitions;
	int mssg_size = mst->mssg_size;  //******** SEE HERE TIMES 2 FOR BIDIRECTED EDGES *********
	if (mssg_factor == 0)
		mssg_factor = MSSG_FACTOR + 1;
	int i;
	for (i = num_of_devices; i < num_of_cpus+num_of_devices; i++)
	{
		vid_t gsize = mst->num_vs[i] * BINARY_FACTOR_CPU;

		dbm[i].vs = (vertex_t *) malloc (sizeof(vertex_t) * gsize);
		CHECK_PTR_RETURN (dbm[i].vs, "malloc vertex hash table error on cpu!\n");
		memset (dbm[i].vs, 0, sizeof(vertex_t) * gsize);
		ull send_size = sizeof(entry_t) * max_subsize > (ull)mssg_size * gsize * mssg_factor? \
				sizeof(entry_t) * max_subsize : (ull)mssg_size * gsize * mssg_factor;
		if (mst->world_size == 1)
			printf ("gsize = %u,, max_subsize = %u, send buffer malloced: %lu\n", gsize, max_subsize, send_size);

		dbm[i].comm.send = (void *) malloc (send_size);
		CHECK_PTR_RETURN (dbm[i].comm.send, "malloc send buffer for cpu %d error!\n", i);

		dbm[i].id_offsets = (vid_t *) malloc (sizeof(vid_t) * (total_num_partitions + 1));
		CHECK_PTR_RETURN(dbm[i].id_offsets, "malloc global id offsets error!\n");
		dbm[i].jid_offset = (vid_t *) malloc (sizeof(vid_t) * total_num_partitions);
		CHECK_PTR_RETURN(dbm[i].jid_offset, "malloc global junction id offset error!\n");
		dbm[i].jvld = (voff_t *) malloc (sizeof(voff_t) * (gsize+1));
		CHECK_PTR_RETURN(dbm[i].jvld, "malloc junction flag array error!\n");
		dbm[i].lvld = (voff_t *) malloc (sizeof(voff_t) * (gsize+1));
		CHECK_PTR_RETURN(dbm[i].lvld, "malloc linear flag array error!\n");

		memset (dbm[i].jvld, 0, sizeof(voff_t) * (gsize+1));
		memset (dbm[i].lvld, 0, sizeof(voff_t) * (gsize+1));

		int num_partitions_device = mst->num_partitions[i+1] - mst->num_partitions[i];
		mst->index_offset[i] = (voff_t *) malloc (sizeof(voff_t) * (num_partitions_device+1));
		CHECK_PTR_RETURN (mst->index_offset[i], "malloc index_offset in master for device %d error!\n", i);
		memset (mst->index_offset[i], 0, sizeof(voff_t) * (num_partitions_device+1));
	}
}

void finalize_host_filter2 (dbmeta_t *dbm, master_t *mst)
{
	int num_of_devices = mst->num_of_devices;
	int num_of_cpus = mst->num_of_cpus;
	int i;
	for (i=num_of_devices; i<num_of_devices+num_of_cpus; i++)
	{
		free(dbm[i].jvld);
		free(dbm[i].lvld);
//		free (dbm[i].comm.send);
	}
}

void set_id_offsets_cpu (dbmeta_t * dbm, master_t * mst)
{
	int num_of_devices = mst->num_of_devices;
	int num_of_cpus = mst->num_of_cpus;
	int total_num_partitions = mst->total_num_partitions;
#ifdef SINGLE_NODE
	int world_rank = mst->world_rank;
#endif

	int i;
	for (i=num_of_devices; i<num_of_devices+num_of_cpus; i++)
	{
		memcpy (dbm[i].id_offsets, mst->id_offsets, sizeof(vid_t) * (total_num_partitions + 1));
		memcpy (dbm[i].jid_offset, mst->jid_offset, sizeof(vid_t) * total_num_partitions);
	}
}

void init_host_preprocessing (dbmeta_t * dbm, master_t * mst)
{
	int num_of_cpus = mst->num_of_cpus;
	int num_of_devices = mst->num_of_devices;
	int total_num_partitions = mst->total_num_partitions;
	int mssg_size = mst->mssg_size;  //******** SEE HERE TIMES 2 FOR BIDIRECTED EDGES *********
	if (mssg_factor == 0)
		mssg_factor = MSSG_FACTOR + 1;
	vid_t mssg_from_gpus = 0;
	int i;
	for (i=0; i<num_of_devices; i++)
	{
		mssg_from_gpus += mst->num_vs[i];
	}
	for (i = num_of_devices; i < num_of_cpus+num_of_devices; i++)
	{
		vid_t gsize = mst->num_vs[i] * BINARY_FACTOR_CPU;
		if (mst->world_size == 1)
			printf ("++++++++++++++++++++++++\nNumber of vertices in CPU: %u\nmssg_size = %d\n", gsize, mst->mssg_size);

		dbm[i].comm.id2index = (int *) malloc (sizeof(int) * total_num_partitions);
		CHECK_PTR_RETURN (dbm[i].comm.id2index, "malloc id2index array for cpu %d error!\n", i);
		dbm[i].comm.receive_offsets = (voff_t *) malloc (sizeof(voff_t) * (total_num_partitions + 1));
		CHECK_PTR_RETURN (dbm[i].comm.receive_offsets, "malloc receive offsets for cpu %d error!\n", i);
		dbm[i].comm.send_offsets = (voff_t *) malloc (sizeof(voff_t) * (total_num_partitions + 1));
		CHECK_PTR_RETURN (dbm[i].comm.send_offsets, "malloc send offsets for cpu %d error!\n", i);
		memset (dbm[i].comm.receive_offsets, 0, sizeof(voff_t) * (total_num_partitions + 1));
		memset (dbm[i].comm.send_offsets, 0, sizeof(voff_t) * (total_num_partitions + 1));
//		dbm[i].comm.send = (void *) malloc ((ull)mssg_size * gsize * HASH_LOAD_FACTOR * INTER_BUF_FACTOR);
//		CHECK_PTR_RETURN (dbm[i].comm.send, "malloc send buffer for cpu %d error!\n", i);
		memcpy (dbm[i].comm.id2index, mst->id2index[i], sizeof(int) * total_num_partitions);

		mst->roff[i] = (voff_t *) malloc (sizeof(voff_t) * (total_num_partitions+1));
		CHECK_PTR_RETURN (mst->roff[i], "malloc receive offset in master for cpu %d error!\n", i);
		mst->soff[i] = (voff_t *) malloc (sizeof(voff_t) * (total_num_partitions+1));
		CHECK_PTR_RETURN (mst->soff[i], "malloc send offset in master for cpu %d error!\n", i);
		memset (mst->roff[i], 0, sizeof(voff_t) * (total_num_partitions+1));
		memset (mst->soff[i], 0, sizeof(voff_t) * (total_num_partitions+1));
		mst->receive[i] = dbm[i].comm.send; // actually set this on the run
		vid_t master_send_size = gsize;
		mst->send[i] = (void *) malloc ((ull)mssg_size * master_send_size * mssg_factor);
		CHECK_PTR_RETURN (mst->send[i], "malloc send buffer in master for device %d error!\n", i);

		int num_partitions_device = mst->num_partitions[i+1] - mst->num_partitions[i];
		dbm[i].comm.index_offsets = (voff_t *) malloc (sizeof(voff_t) * (num_partitions_device + 1));
		CHECK_PTR_RETURN (dbm[i].comm.index_offsets, "malloc index_offsets on cpu error!\n");
		memcpy (dbm[i].comm.index_offsets, mst->index_offset[i], sizeof(voff_t) * (num_partitions_device + 1));
		tmp_counts[i] = (voff_t *) malloc (sizeof(voff_t) * num_partitions_device);
	}
}

void finalize_host_preprocessing (dbmeta_t * dbm, master_t * mst)
{
	int num_of_devices = mst->num_of_devices;
	int num_of_cpus = mst->num_of_cpus;
	int i;
	for (i = num_of_devices; i < num_of_cpus+num_of_devices; i++)
	{
		free (dbm[i].comm.id2index);
		free (dbm[i].comm.receive_offsets);
		free (dbm[i].comm.send_offsets);
		free (dbm[i].comm.send);
		free (dbm[i].comm.index_offsets);

		free (mst->roff[i]);
		free (mst->soff[i]);
		free (mst->send[i]);
		free (tmp_counts[i]);
	}
}

void init_host_gather (dbmeta_t * dbm, master_t * mst, vid_t max_subsize, vid_t max_jsize, vid_t max_lsize)
{
	int num_of_cpus = mst->num_of_cpus;
	int num_of_devices = mst->num_of_devices;
	int i;
	for (i = num_of_devices; i < num_of_cpus+num_of_devices; i++)
	{
		dbm[i].jvld = (voff_t *) malloc (sizeof(voff_t) * (max_subsize + 1));
		CHECK_PTR_RETURN(dbm[i].jvld, "malloc junction flag array error!\n");
		dbm[i].lvld = (voff_t *) malloc (sizeof(voff_t) * (max_subsize + 1));
		CHECK_PTR_RETURN(dbm[i].lvld, "malloc linear flag array error!\n");
		int j;
		for (j=0; j<EDGE_DIC_SIZE; j++)
		{
			dbm[i].djs.nbs[j] = (vid_t *) malloc (sizeof(vid_t) * max_jsize);
			CHECK_PTR_RETURN(dbm[i].djs.nbs[j], "malloc junction neighbor array %d error!\n", j);
		}
		dbm[i].dls.posts = (vid_t *) malloc (sizeof(vid_t) * max_lsize);
		CHECK_PTR_RETURN(dbm[i].dls.posts, "malloc linear post array error!\n");
		dbm[i].dls.pres = (vid_t *) malloc (sizeof(vid_t) * max_lsize);
		CHECK_PTR_RETURN(dbm[i].dls.pres, "malloc linear pre array error!\n");

		dbm[i].djs.kmers = (kmer_t *) malloc (sizeof(kmer_t) * max_jsize);
		CHECK_PTR_RETURN (dbm[i].djs.kmers, "malloc host junction kmers for gather error!\n");
		dbm[i].djs.edges = (ull *) malloc (sizeof(ull) * max_jsize);
		CHECK_PTR_RETURN (dbm[i].djs.edges, "malloc host edges for gather error!\n");
//		dbm[i].dls.kmers = (kmer_t *) malloc (sizeof(kmer_t) * max_lsize);
//		CHECK_PTR_RETURN (dbm[i].dls.kmers, "malloc host linear vertex kmers for gather error!\n");
		dbm[i].dls.pre_edges = (edge_type *) malloc (sizeof(edge_type) * max_lsize);
		CHECK_PTR_RETURN (dbm[i].dls.pre_edges, "malloc host pre edges for gather error!\n");
		dbm[i].dls.post_edges = (edge_type *) malloc (sizeof(edge_type) * max_lsize);
		CHECK_PTR_RETURN (dbm[i].dls.post_edges, "malloc host post edges for gather error!\n");

		memset (dbm[i].jvld, 0, sizeof(voff_t) * (max_subsize + 1));
		memset (dbm[i].lvld, 0, sizeof(voff_t) * (max_subsize + 1));

		ull cpy_size = max_jsize * sizeof(vid_t) * EDGE_DIC_SIZE + max_lsize * sizeof(vid_t) * 2;
		mst->send[i] = (void *) malloc (cpy_size);
		CHECK_PTR_RETURN (mst->send[i], "malloc buffer for transfer adj graph for device %d error!\n", i);
	}
}

void finalize_host_gather2 (dbmeta_t * dbm, master_t * mst)
{
	int num_of_devices = mst->num_of_devices;
	int num_of_cpus = mst->num_of_cpus;
	int i;
	for (i = num_of_devices; i < num_of_cpus+num_of_devices; i++)
	{
		free(dbm[i].vs);
		int j;
		for (j=0; j<EDGE_DIC_SIZE; j++)
		{
			free (dbm[i].djs.nbs[j]);
		}
		free(dbm[i].id_offsets);
		free(dbm[i].jid_offset);
		free(dbm[i].djs.kmers);
		free(dbm[i].djs.edges);
//		free(dbm[i].dls.kmers);
		free(dbm[i].dls.posts);
		free(dbm[i].dls.pres);
		free(dbm[i].dls.post_edges);
		free(dbm[i].dls.pre_edges);
		free(dbm[i].jvld);
		free(dbm[i].lvld);

		free (mst->send[i]);
		free (mst->index_offset[i]);
	}
}

void init_host_graph_compute (meta_t * dm, master_t * mst)
{
	int num_of_cpus = mst->num_of_cpus;
	int num_of_devices = mst->num_of_devices;
	int total_num_partitions = mst->total_num_partitions;
	int mssg_size = mst->mssg_size;
	if (mssg_factor == 0)
		mssg_factor = MSSG_FACTOR + 1;
	double contig_mssg_factor = mssg_factor;
	vid_t mssg_from_gpus = 0;
	int i;
	for (i=0; i<num_of_devices; i++)
	{
		mssg_from_gpus += mst->num_vs[i];
	}
	for (i = num_of_devices; i < num_of_cpus+num_of_devices; i++)
	{
//		int num_of_partitions = mst->num_partitions[i + 1] - mst->num_partitions[i];
		vid_t gsize = mst->num_vs[i];
		if (mst->world_size == 1)
			printf ("++++++++++++++++++++++++\nNumber of vertices in CPU: %u\nmssg_size = %d\n", gsize, mst->mssg_size);
		dm[i].edge.post = (vid_t *) malloc (sizeof(vid_t) * gsize);
		CHECK_PTR_RETURN (dm[i].edge.post, "malloc edge post array for cpu %d error\n", i);
		dm[i].edge.pre = (vid_t *) malloc (sizeof(vid_t) * gsize);
		CHECK_PTR_RETURN (dm[i].edge.pre, "malloc edge pre array for cpu %d error!\n", i);
		dm[i].edge.fwd = (voff_t *) malloc (sizeof(voff_t) * gsize);
		CHECK_PTR_RETURN (dm[i].edge.fwd, "malloc edge fwd dist for cpu %d error!\n", i);
		dm[i].edge.bwd = (voff_t *) malloc (sizeof(voff_t) * gsize);
		CHECK_PTR_RETURN (dm[i].edge.bwd, "malloc edge bwd dist for cpu %d error!\n", i);
		dm[i].edge.fjid = (vid_t *) malloc (sizeof(vid_t) * gsize);
		CHECK_PTR_RETURN (dm[i].edge.fjid, "malloc edge fjid for cpu %d error!\n", i);
		dm[i].edge.bjid = (vid_t *) malloc (sizeof(vid_t) * gsize);
		CHECK_PTR_RETURN (dm[i].edge.bjid, "malloc edge bjid for cpu %d error!\n", i);
		memset (dm[i].edge.fjid, 0, sizeof(vid_t) * gsize);
		memset (dm[i].edge.bjid, 0, sizeof(vid_t) * gsize);

		dm[i].comm.id2index = (int *) malloc (sizeof(int) * total_num_partitions);
		CHECK_PTR_RETURN (dm[i].comm.id2index, "malloc id2index array for cpu %d error!\n", i);
		dm[i].comm.receive_offsets = (voff_t *) malloc (sizeof(voff_t) * (total_num_partitions + 1));
		CHECK_PTR_RETURN (dm[i].comm.receive_offsets, "malloc receive offsets for cpu %d error!\n", i);
		dm[i].comm.send_offsets = (voff_t *) malloc (sizeof(voff_t) * (total_num_partitions + 1));
		CHECK_PTR_RETURN (dm[i].comm.send_offsets, "malloc send offsets for cpu %d error!\n", i);
		memset (dm[i].comm.receive_offsets, 0, sizeof(voff_t) * (total_num_partitions + 1));
		memset (dm[i].comm.send_offsets, 0, sizeof(voff_t) * (total_num_partitions + 1));
		dm[i].comm.send = (void *) malloc ((ull)mssg_size * gsize * contig_mssg_factor);
		CHECK_PTR_RETURN (dm[i].comm.send, "malloc send buffer for cpu %d error!\n", i);

		dm[i].id_offsets = (vid_t *) malloc (sizeof(vid_t) * (total_num_partitions + 1));
		CHECK_PTR_RETURN (dm[i].id_offsets, "malloc id_offsets for cpu %d error!\n", i);
		dm[i].jid_offset = (vid_t *) malloc (sizeof(vid_t) * total_num_partitions);
		CHECK_PTR_RETURN (dm[i].jid_offset, "malloc jid_offsets for cpu %d error!\n", i);

		memcpy (dm[i].comm.id2index, mst->id2index[i], sizeof(int) * total_num_partitions);
		memcpy (dm[i].id_offsets, mst->id_offsets, sizeof(vid_t) * (total_num_partitions + 1));
		memcpy (dm[i].jid_offset, mst->jid_offset, sizeof(vid_t) * total_num_partitions);

		mst->roff[i] = (voff_t *) malloc (sizeof(voff_t) * (total_num_partitions+1));
		CHECK_PTR_RETURN (mst->roff[i], "malloc receive offset in master for cpu %d error!\n", i);
		mst->soff[i] = (voff_t *) malloc (sizeof(voff_t) * (total_num_partitions+1));
		CHECK_PTR_RETURN (mst->soff[i], "malloc send offset in master for cpu %d error!\n", i);
		memset (mst->roff[i], 0, sizeof(voff_t) * (total_num_partitions+1));
		memset (mst->soff[i], 0, sizeof(voff_t) * (total_num_partitions+1));
		mst->receive[i] = dm[i].comm.send; // actually set this on the run
		vid_t master_send_size = gsize;
//		if (num_of_devices > 0)
//			master_send_size = gsize < mssg_from_gpus? gsize : mssg_from_gpus;
		mst->send[i] = (void *) malloc ((ull)mssg_size * master_send_size * contig_mssg_factor);
		CHECK_PTR_RETURN (mst->send[i], "malloc send buffer in master for device %d error!\n", i);
		int num_partitions_device = mst->num_partitions[i+1] - mst->num_partitions[i];
		mst->index_offset[i] = (voff_t *) malloc (sizeof(voff_t) * (num_partitions_device+1));
		CHECK_PTR_RETURN (mst->index_offset[i], "malloc index_offset in master for device %d error!\n", i);
		memset (mst->index_offset[i], 0, sizeof(voff_t) * (num_partitions_device+1));
		tmp_counts[i] = (voff_t *) malloc (sizeof(voff_t) * num_partitions_device);
//		mst->flag[i] = 0;
	}
}


void finalize_host_graph_compute (meta_t * dm, master_t * mst)
{
	int num_of_devices = mst->num_of_devices;
	int num_of_cpus = mst->num_of_cpus;
	int i;
	for (i = num_of_devices; i < num_of_cpus+num_of_devices; i++)
	{
		free (dm[i].edge.fwd);
		free (dm[i].edge.bwd);
		free (dm[i].edge.fjid);
		free (dm[i].edge.bjid);

		free (dm[i].comm.id2index);
		free (dm[i].comm.receive_offsets);
		free (dm[i].comm.send_offsets);
		free (dm[i].comm.send);

		free (mst->roff[i]);
		free (mst->soff[i]);
		free (mst->send[i]);
		free (mst->index_offset[i]);
		free (tmp_counts[i]);
	}
}


void malloc_pull_push_offset_cpu (voff_t ** extra_send_offsets, master_t * mst)
{
	int total_num_partitions = mst->total_num_partitions;
#ifdef SINGLE_NODE
	int world_rank = mst->world_rank;
#endif

	*extra_send_offsets = (voff_t *) malloc(sizeof(voff_t) * (total_num_partitions + 1));
	CHECK_PTR_RETURN (*extra_send_offsets, "malloc extra send_offsets for extra message push error for device\n");
	memset (*extra_send_offsets, 0, sizeof(voff_t) * (total_num_partitions+1));
}

void free_pull_push_offset_cpu (voff_t * extra_send_offsets)
{
	free(extra_send_offsets);
}

void malloc_pull_push_receive_cpu (comm_t * dm, uint mssg_size, int did, voff_t num_of_mssgs, int expand)
{
	size_t malloc_size = (ull)mssg_size * num_of_mssgs * expand;
	dm->receive = (void *) malloc (malloc_size);
	CHECK_PTR_RETURN (dm->receive, "malloc receive buffer for cpu %d error!\n", did);
	dm->temp_size = malloc_size;
}

void free_pull_push_receive_cpu (comm_t * dm)
{
	free (dm->receive);
}

void realloc_host_edges (meta_t * dm, master_t * mst)
{
	int num_of_devices = mst->num_of_devices;
	int num_of_cpus = mst->num_of_cpus;
#ifdef SINGLE_NODE
	int world_rank = mst->world_rank;
	int world_size = mst->world_size;
#endif
	int i;
	for (i=num_of_devices; i<num_of_devices+num_of_cpus; i++)
	{
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
		dm[i].edge.pre_edges = (edge_type *) malloc (sizeof(edge_type) * lsize);
		CHECK_PTR_RETURN (dm[i].edge.pre_edges, "realloc linear pre edges error!!!\n");
		dm[i].edge.post_edges = (edge_type *) malloc (sizeof(edge_type) * lsize);
		CHECK_PTR_RETURN (dm[i].edge.post_edges, "realloc linear post edges error!!!\n");
		dm[i].junct.edges = (ull *) malloc (sizeof(ull) * jsize);
		CHECK_PTR_RETURN (dm[i].junct.edges, "realloc junction edges error!!!\n");
	}
}

void realloc_host_junctions (meta_t * dm, master_t * mst)
{
	int num_of_devices = mst->num_of_devices;
	int num_of_cpus = mst->num_of_cpus;
	int i;
	for (i=num_of_devices; i<num_of_devices+num_of_cpus; i++)
	{
		vid_t gsize = mst->num_js[i];
		vid_t nbsize = mst->num_jnbs[i];
		if (gsize > MAX_NUM_NODES_DEVICE)
		{
			printf ("ERROR: graph size greater than maximum number of vertices in a device!\n");
			// exit(0);
		}
		if (mst->world_size == 1)
			printf ("++++++++++++++++++++++++\nnumber of junctions in device %d: %u\n"
				"number of neighbors for junctions: %u\nmssg size: %d\n", i, gsize, nbsize, mst->mssg_size);

		int num_partitions_device = mst->num_partitions[i+1] - mst->num_partitions[i];

		dm[i].junct.offs = (voff_t *) malloc (sizeof(voff_t) * (gsize + num_partitions_device));
		dm[i].junct.nbs = (vid_t *) malloc (sizeof(vid_t) * nbsize);
		dm[i].junct.kmers = (kmer_t *) malloc (sizeof(kmer_t) * gsize);
		dm[i].junct.ulens = (size_t *) malloc (sizeof(size_t) * nbsize);
		CHECK_PTR_RETURN (dm->junct.offs, "realloc junct offsets on host error!\n");
		CHECK_PTR_RETURN (dm->junct.nbs, "realloc junct nbs on host error!\n");
		CHECK_PTR_RETURN (dm->junct.kmers, "realloc junct kmers on host error!\n");
		CHECK_PTR_RETURN (dm->junct.ulens, "malloc unitig length array for junctions error!\n");

		memset (dm[i].junct.ulens, 0, sizeof(size_t) * nbsize);

		mst->jindex_offset[i] = (voff_t *) malloc (sizeof(voff_t) * (num_partitions_device+1));
		CHECK_PTR_RETURN (mst->jindex_offset[i], "malloc jindex_offset in master for device %d error!\n", i);
		memset (mst->jindex_offset[i], 0, sizeof(voff_t) * (num_partitions_device+1));
		mst->jnb_index_offset[i] = (voff_t *) malloc (sizeof(voff_t) * (num_partitions_device+1));
		CHECK_PTR_RETURN (mst->jnb_index_offset[i], "malloc jnb_index_offset in master for device %d error!\n", i);
		memset (mst->jnb_index_offset[i], 0, sizeof(voff_t) * (num_partitions_device+1));
	}
}

void free_host_realloc (meta_t * dm, master_t * mst)
{
	int num_of_devices = mst->num_of_devices;
	int num_of_cpus = mst->num_of_cpus;
	int i;
	for (i=num_of_devices; i < num_of_devices + num_of_cpus; i++)
	{
		free (dm[i].edge.post);
		free (dm[i].edge.pre);
		free(dm[i].edge.pre_edges);
		free(dm[i].edge.post_edges);
		free(dm[i].junct.offs);
		free(dm[i].junct.nbs);
		free(dm[i].junct.kmers);
		free(dm[i].junct.ulens);
		free(dm[i].junct.edges);
		free (mst->jindex_offset[i]);
		free (mst->jnb_index_offset[i]);
	}
}

void collect_free_memory_cpu (meta_t * dm, master_t * mst)
{
	int num_of_devices = mst->num_of_devices;
	int num_of_cpus = mst->num_of_cpus;
	int i;
	for (i=num_of_devices; i<num_of_devices+num_of_cpus; i++)
	{
		free (dm[i].edge.post);
		free (dm[i].edge.pre);
	}
}

void malloc_unitig_buffer_cpu (meta_t * dm, size_t size, int num_of_partitions)
{
#ifdef CONTIG_CHECK
	printf ("CPU: unitig size malloced: %lu\n", size);
#endif
	dm->junct.unitigs = (char*) malloc (sizeof(char) * size);
	CHECK_PTR_RETURN (dm->junct.unitigs, "malloc unitig buffer error!\n");
	dm->junct.unitig_offs = (size_t *) malloc (sizeof(size_t) * (num_of_partitions+1));
	CHECK_PTR_RETURN (dm->junct.unitig_offs, "malloc unitig offsets to output unitigs error!\n");
}

void free_unitig_buffer_cpu (meta_t * dm, master_t * mst)
{
	int num_of_devices = mst->num_of_devices;
	int num_of_cpus = mst->num_of_cpus;
	int i;
	for (i=num_of_devices; i<num_of_devices+num_of_cpus; i++)
	{
		free (dm[i].junct.unitigs);
	}
}
