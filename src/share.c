/*
 * share.c
 *
 *  Created on: 2017-8-23
 *      Author: qiushuang
 */

#include <omp.h>
#include "../include/dbgraph.h"
#include "../include/share.h"
#include "../include/graph.h"
#include "../include/comm.h"
#include "../include/msp.h"
#include "../include/share.h"

extern thread_function shift_dictionary[];
extern int cutoff;

void inclusive_prefix_sum (int * array, int num)
{
	int i;
	for (i = 0; i < num - 1; i++)
	{
		array[i + 1] += array[i];
	}
}

voff_t get_max (subsize_t * subs, goffset_t * joff, goffset_t * toff, voff_t * max_sub_size, voff_t * max_jsize, voff_t * max_lsize, int intra_num_of_partitions, int total_num_of_partitions)
{
	int i;
	if (subs != NULL)
		*max_sub_size = subs[0].size;
	if (joff != NULL)
		*max_jsize = joff[0];
	if (toff != NULL)
		*max_lsize = toff[1] - joff[0];
	for (i=1; i<intra_num_of_partitions; i++)
	{
		if (subs != NULL && *max_sub_size < subs[i].size)
		{
			*max_sub_size = subs[i].size;
		}
	}
	voff_t max_ss = *max_sub_size;
	for (i=1; i<total_num_of_partitions; i++)
	{
		if (joff != NULL && *max_jsize < joff[i])
		{
			*max_jsize = joff[i];
		}
		if (toff != NULL && *max_lsize < toff[i+1] - toff[i] - joff[i])
		{
			*max_lsize = toff[i+1] - toff[i] - joff[i];
		}
	}

	return max_ss;
}

void get_subgraph_sizes (subgraph_t * subgraph, int num_of_partitions)
{
	int i;
	subgraph->total_graph_size=0;
	for (i=0; i<num_of_partitions; i++)
	{
		subgraph->total_graph_size += (subgraph->subgraphs)[i].size;
	}
}

int query_partition_id_from_idoffsets (vid_t id, int num_of_partitions, goffset_t * id_offsets)
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
	printf ("QQQQQQQQQQQQQQQuery partition id out of range!!!!!!!\n");
//	exit(0);
	return -1; // error: vertex id is out of range!!!
}

void get_global_offsets (voff_t * goff, voff_t * loff, int num_of_partitions, int cpu_threads)
{
	int i;
	for (i=0; i<num_of_partitions+1; i++)
	{
		int j;
		for (j=1; j<cpu_threads; j++)
		{
			loff[j*(num_of_partitions+1)+i] += loff[(j-1)*(num_of_partitions+1)+i];
		}
		goff[i] = loff[(cpu_threads-1)*(num_of_partitions+1)+i];
	}
//	printf ("testing~~~~~~~~~\n");
//	print_offsets(goff, num_of_partitions+1);
}

void init_mssg_count (voff_t * intra_mssgs, voff_t * inter_mssgs)
{
	memset (intra_mssgs, 0, sizeof(voff_t) * (NUM_OF_PROCS*MAX_NUM_ITERATION));
	memset (inter_mssgs, 0, sizeof(voff_t) * (NUM_OF_PROCS*MAX_NUM_ITERATION));
}

void get_mssg_count (master_t * mst, voff_t * intra_mssgs, voff_t * inter_mssgs, int iter)
{
	int i;
	int num_of_devices = mst->num_of_devices;
	int num_of_cpus = mst->num_of_cpus;
	int num_of_procs = num_of_devices + num_of_cpus;
	for (i=0; i<num_of_procs; i++)
	{
		int num_of_partitions = mst->num_partitions[i+1] - mst->num_partitions[i];
		int total_num_partitions = mst->num_partitions[num_of_procs];
		voff_t num_intra = mst->roff[i][num_of_partitions];
		voff_t num_inter = mst->soff[i][num_of_partitions];
		intra_mssgs[(iter-1)*num_of_procs + i] = num_intra;
		inter_mssgs[(iter-1)*num_of_procs + i] = num_inter;
	}
}

void print_mssg_count (int num_of_procs, voff_t * intra_mssgs, voff_t * inter_mssgs, int iters)
{
	int i;
	printf ("INTRA PROCS MESSAGES:\n");
	for (i = 0; i < iters; i++)
	{
		int j;
		for (j=0; j<num_of_procs; j++)
		{
			printf ("%u\t", intra_mssgs[i*num_of_procs+j]);
		}
		printf ("\n");
	}

	printf ("INTER PROCS MESSAGES:\n");
	for (i = 0; i < iters; i++)
	{
		int j;
		for (j=0; j<num_of_procs; j++)
		{
			printf ("%u\t", inter_mssgs[i*num_of_procs+j]);
		}
		printf ("\n");
	}
}

void print_offsets (voff_t * array, int num)
{
	int i;
	for (i = 0; i < num; i++)
	{
		printf ("%u\t", array[i]);
	}
	printf ("\n");
}

void write_hashtab (char * file_dir, node_t * tab, voff_t size, voff_t elem_hashed, int pid, int total_num_partitions, int did)
{
	FILE * file;
	char filename[FILENAME_LENGTH];
	memset (filename, 0, sizeof(char) * FILENAME_LENGTH);
	sprintf (filename, "%s/sub%d_%d", file_dir, pid, total_num_partitions);

 	if ((file = fopen (filename, "w")) == NULL)
 	{
 		printf ("OPEN subgraph %d adj subgraph file for device %d error\n", pid, did);
 		// exit(0);
 	}
 	printf ("writing subgraph size %u for hashtab %d\n", size, pid);
	fwrite (&size, sizeof(voff_t), 1, file);
	fwrite (&elem_hashed, sizeof(voff_t), 1, file);
	fwrite (tab, sizeof(node_t), size, file);
	fclose (file);
}

void write_ids_cpu (dbmeta_t * dbm, master_t * mst, voff_t total_num_partitions, int did)
{
	FILE * file;
	char filename[FILENAME_LENGTH];
	sprintf (filename, "%s/ids_%d", mst->file_dir, total_num_partitions);
	if ((file = fopen(filename, "w")) == NULL)
	{
		printf ("open id file error!\n");
		// exit(0);
	}
	vertex_t * vs = dbm->vs;
	vertex_t * local_vs;
	voff_t * index_offset = mst->index_offset[did];
	voff_t i;
	voff_t count = 0;
	for (i=0; i<total_num_partitions; i++)
	{
		int poffset = mst->num_partitions[did];
		voff_t size = index_offset[i+1] - index_offset[i];
		local_vs = vs + index_offset[i];
		int j;
		for (j=0; j<size; j++)
		{
			if (local_vs[j].vid == 0)
			{
				count++;
				continue;
			}
			fprintf (file, "%u\n", local_vs[j].vid);
		}
	}
	fclose (file);
	printf ("number of vertices filtered: %u\n", count);
}

void output_vertices_cpu (dbmeta_t * dbm, master_t * mst, voff_t jsize, voff_t lsize, int pid, int total_num_partitions, int did, d_jvs_t * djs, d_lvs_t * dls, subgraph_t * subgraph)
{
	int world_size = mst->world_size;
	int world_rank = mst->world_rank;
	int np_per_node = (total_num_partitions + world_size - 1)/world_size;

	int start_partition_id = np_per_node*world_rank;
	msp_id_t partition_id = pid - start_partition_id;

	djs[pid - start_partition_id].size = jsize;
	int t;
	for (t=0; t<EDGE_DIC_SIZE; t++)
	{
		djs[partition_id].nbs[t] = (vid_t *)malloc(sizeof(vid_t) * jsize);
		CHECK_PTR_RETURN (djs[partition_id].nbs[t], "malloc djs[%d].nbs[%d] array error!\n", pid, t);
		memcpy (djs[partition_id].nbs[t], dbm->djs.nbs[t], sizeof(vid_t) * jsize);
	}
	djs[partition_id].spids = (ull *)malloc(sizeof(ull) * jsize);
	CHECK_PTR_RETURN (djs[partition_id].spids, "malloc spids for djs[%d] error!\n", pid);
	djs[partition_id].spidsr = (ull *)malloc(sizeof(ull) * jsize);
	CHECK_PTR_RETURN (djs[partition_id].spidsr, "malloc spidsr for djs[%d] error!\n", pid);
	memcpy (djs[partition_id].spids, dbm->djs.spids, sizeof(ull) * jsize);
	memcpy (djs[partition_id].spidsr, dbm->djs.spidsr, sizeof(ull) * jsize);

	dls[partition_id].esize = lsize;
	dls[partition_id].asize = 0;
	dls[partition_id].id = NULL;
	dls[partition_id].posts = (vid_t*) malloc (sizeof(vid_t) * lsize);
	CHECK_PTR_RETURN (dls[partition_id].posts, "malloc posts for partition %d error!\n", pid);
	dls[partition_id].pres = (vid_t *) malloc (sizeof(vid_t) * lsize);
	CHECK_PTR_RETURN (dls[partition_id].pres, "malloc pres for partition %d error!\n", pid);
	dls[partition_id].spids = (uint *) malloc (sizeof(uint) * lsize);
	CHECK_PTR_RETURN (dls[partition_id].spids, "malloc spids for partition %d error!\n", pid);
	memcpy (dls[partition_id].posts, dbm->dls.posts, sizeof(vid_t) * lsize);
	memcpy (dls[partition_id].pres, dbm->dls.pres, sizeof(vid_t) * lsize);
	memcpy (dls[partition_id].spids, dbm->dls.spids, sizeof(uint) * lsize);

	(subgraph->subgraphs)[partition_id].id = pid;
	(subgraph->subgraphs)[partition_id].size = lsize;
}

void write_kmers_edges_cpu (dbmeta_t * dbm, master_t * mst, voff_t jsize, voff_t lsize, int pid, int total_num_partitions, int did)
{
	FILE * file;
	FILE * kfile;
	FILE * efile;
	char filename[FILENAME_LENGTH];
	char fname[FILENAME_LENGTH];
	char efname[FILENAME_LENGTH];

	sprintf (filename, "%s/ledge%d_%d", mst->file_dir, pid, total_num_partitions); // edge file for linear posts and pres
	sprintf (fname, "%s/jkmer%d_%d", mst->file_dir, pid, total_num_partitions); // kmer file for junctions
	sprintf (efname, "%s/jedge%d_%d", mst->file_dir, pid, total_num_partitions); // edge file (ull) for junctions

	if ((file = fopen (filename, "w")) == NULL)
	{
		printf ("OPEN subgraph %d linear subgraph kmer file for device %d error\n", pid, did);
	}
	if ((kfile = fopen (fname, "w")) == NULL)
	{
		printf ("OPEN subgraph %d junction kmer file for device %d error\n", pid, did);
		exit(0);
	}
	if ((efile = fopen (efname, "w")) == NULL)
	{
		printf ("OPEN subgraph %d junction edge file for device %d error\n", pid, did);
		exit(0);
	}

	fwrite (dbm->dls.pre_edges, sizeof(edge_type), lsize, file);
	fwrite (dbm->dls.post_edges, sizeof(edge_type), lsize, file);

	fwrite (dbm->djs.kmers, sizeof(kmer_t), jsize, kfile);

	fwrite (dbm->djs.edges, sizeof(ull), jsize, efile);

	fclose (file);
	fclose (kfile);
	fclose (efile);
}

void write_junctions_cpu (dbmeta_t * dbm, master_t * mst, voff_t jsize, voff_t lsize, int pid, int total_num_partitions, int did)
{
	FILE * file;
	FILE * kfile;
	FILE * efile;
	char filename[FILENAME_LENGTH];
	char fname[FILENAME_LENGTH];
	char efname[FILENAME_LENGTH];
	sprintf (filename, "%s/jv%d_%d", mst->file_dir, pid, total_num_partitions);
	sprintf (fname, "%s/jkmer%d_%d", mst->file_dir, pid, total_num_partitions);
	sprintf (efname, "%s/jedge%d_%d", mst->file_dir, pid, total_num_partitions);

	if ((file = fopen (filename, "w")) == NULL)
	{
		printf ("OPEN subgraph %d junction subgraph file for device %d error\n", pid, did);
		// exit(0);
	}
	if ((kfile = fopen (fname, "w")) == NULL)
	{
		printf ("OPEN subgraph %d junction kmer file for device %d error\n", pid, did);
		// exit(0);
	}
	if ((efile = fopen (efname, "w")) == NULL)
	{
		printf ("OPEN subgraph %d junction edge file for device %d error\n", pid, did);
		// exit(0);
	}

	vid_t * nbs[EDGE_DIC_SIZE];
	int i;
	for (i=0; i<EDGE_DIC_SIZE; i++)
	{
		nbs[i] = dbm->djs.nbs[i];
	}
	ull cpy_size = 0;
	void * buf = mst->send[did];
	for (i=0; i<EDGE_DIC_SIZE; i++)
	{
		memcpy (buf + cpy_size, nbs[i], sizeof(vid_t) * jsize);
		cpy_size += sizeof(vid_t) * jsize;
	}

	fwrite (&jsize, sizeof(voff_t), 1, file);
	fwrite (buf, 1, cpy_size, file);

	fwrite (dbm->djs.kmers, sizeof(kmer_t), jsize, kfile);

	fwrite (dbm->djs.edges, sizeof(ull), jsize, efile);

	fclose (file);
	fclose (kfile);
	fclose (efile);
}

void write_linear_vertices_cpu (dbmeta_t * dbm, master_t * mst, voff_t jsize, voff_t lsize, int pid, int total_num_partitions, int did)
{
	FILE * file;
	FILE * kfile;
	char filename[FILENAME_LENGTH];
	char fname[FILENAME_LENGTH];
	sprintf (filename, "%s/lv%d_%d", mst->file_dir, pid, total_num_partitions);
//	sprintf (fname, "/home/sqiuac/data/result/ecoli/lkmer%d_%d", pid, total_num_partitions);
	sprintf (fname, "%s/ledge%d_%d", mst->file_dir, pid, total_num_partitions);

	if ((file = fopen (filename, "w")) == NULL)
	{
		printf ("Open subgraph %d linear subgraph file for device %d error!\n", pid, did);
		// exit(0);
	}
	if ((kfile = fopen (fname, "w")) == NULL)
	{
		printf ("OPEN subgraph %d linear subgraph edge file for device %d error\n", pid, did);
		// exit(0);
	}

	vid_t * posts = dbm->dls.posts;
	vid_t * pres = dbm->dls.pres;
	ull cpy_size = 0;
	void * buf = mst->send[did];

	memcpy (buf + cpy_size, posts, sizeof(vid_t) * lsize);
	cpy_size += sizeof(vid_t) * lsize;
	memcpy (buf + cpy_size, pres, sizeof(vid_t) * lsize);
	cpy_size += sizeof(vid_t) * lsize;

	fwrite (&lsize, sizeof(voff_t), 1, file);
	fwrite (buf, 1, cpy_size, file);

	fwrite (dbm->dls.pre_edges, sizeof(edge_type), lsize, kfile);
	fwrite (dbm->dls.post_edges, sizeof(edge_type), lsize, kfile);

	fclose (file);
	fclose (kfile);
}

void write_contigs_cpu (meta_t * dm, master_t * mst, int did, int k)
{
	FILE * file;
	char filename[FILENAME_LENGTH];
	int world_rank = mst->world_rank;
	sprintf (filename, "%s/contig%d_%d", mst->contig_dir, did, world_rank);

	if ((file = fopen (filename, "w")) == NULL)
	{
		printf ("Open contig file error with CPU %d!\n", did);
		exit(0);
	}

	uint count_empty=0;
	char * unitigs = dm->junct.unitigs;
	int i;
	int num_of_partitions = mst->num_partitions[did+1] - mst->num_partitions[did];
	for (i=0; i<num_of_partitions; i++)
	{
		voff_t j;
		voff_t * local_offs = dm->junct.offs + mst->jindex_offset[did][i] + i;
		size_t * local_ulens = dm->junct.ulens + mst->jnb_index_offset[did][i];
		int poffset = mst->num_partitions[did];
		int pid = mst->partition_list[poffset + i];
		voff_t num_js = mst->jid_offset[pid];
		for (j=0; j<num_js; j++)
		{
			voff_t nb_off = local_offs[j];
			voff_t num_nbs = local_offs[j+1] - local_offs[j];
			int num;
			for (num=0; num<num_nbs; num++)
			{
				size_t len = (nb_off+num)==0 ? local_ulens[nb_off+num] : (local_ulens[nb_off+num]-local_ulens[nb_off+num-1]);
				if (len==0)
					continue;
				vid_t contig_id = nb_off+num;
				if (unitigs[0] == '\0' | unitigs[k] == '\0')
				{
					count_empty++;
	//				printf("Error in unitigs!\n");
				}
				if (len>k+1 && unitigs[k+1] == '\0')
				{
					count_empty++;
//					printf ("CPU: from partition %d: junction id: %u (%u), neighbor th: %d, junction size %u\n", pid, j+mst->id_offsets[pid], j, num, num_js);
//					unitigs = unitigs + len;
//					continue;
				}
				fprintf (file, ">%lu length %lu cvg_%d_tip_0\n", contig_id, len, cutoff);
				size_t l;
				for (l=0; l<len; l++)
				{
					fprintf(file, "%c", unitigs[l]);
					if (unitigs[l] != 'A' && unitigs[l] != 'C' && unitigs[l] != 'G' && unitigs[l] != 'T')
					{
//						printf ("Error in output unitigs!\n");
					}
				}
				fprintf (file, "\n");
				unitigs = unitigs + len;
			}
		}
	}

	printf ("!!!!!!!!!!!!!!!!total number of dropped unigits: %u\n", count_empty);
	fclose(file);
}
