/*
 * io.c
 *
 *  Created on: 2017-12-8
 *      Author: qiushuang
 */
#include <getopt.h>
#include "../include/utility.h"
#include "../include/dbgraph.h"
#include "../include/preprocess.h"
#include "../include/comm.h"
#include "../include/share.h"
#include "../include/distribute.h"
#include "../include/malloc.h"

#define FILENAME_LENGTH 150

#define put_id(id) (id + 1)
#define get_id(id) (id - 1)

void usage (void)
{
	printf ("\n\n\tUNIPAR USAGE:::::::::::::::::::::::::::::::::::::::::\n\n"
			"\t-i [STRING]: input file, either a fasta or fastq file\n"
			"\t-r [INT]: read length, the first r number of base pairs in a read will be taken\n"
			"\t-k [INT]: kmer length, no longer than the read length\n\n"
			"\tThe following parameters are optional, and can be omitted:::::::::::::::::::::\n\n"
			"\t-n [INT]: [Optional] number of partitions, set to 512 by default\n"
			"\t-c [INT]: [Optional] number of CPUs to run, either 0 or 1, set to 1 by default\n"
			"\t-g [INT]: [Optional] number of GPUs to run, either set to 0 or the maximum number of GPUs in this system, \n"
			"\t\t             set to be the maximum number of GPUs detected\n"
			"\t-d [STRING]: [Optional] intermediate partitioning output directory, set to ./partitions by default\n"
			"\t-o [STRING]: [Optional] unitig output directory, set to be the current directory by default\n"
			"\t-t [INT]: [Optional] The cutoff threshold for the number of kmer coverage, set to 1 by default\n\n\n");
}

int get_opt (int argc, char * const argv[], char * input, int * r, int * k, int * p, int * n, int * c, int * g, \
		char * dir, char * out, int * t, float * f, int * m)
{
        int opt;
        int count_arg = 0;
        opterr = 0;
        while ((opt = getopt(argc, argv, "i:r:k:p:n:c:g:d:o:t:f:m:")) != -1) {
                switch (opt) {
                case 'i':
                		++count_arg;
                        strcpy(input, optarg);
                        break;
                case 'r':
                        ++count_arg;
                        *r = atoi(optarg);
                        break;
                case 'k':
                        ++count_arg;
                        *k = atoi(optarg);
                        break;
                case 'p':
                        ++count_arg;
                        *p = atoi(optarg);
                        break;
                case 'n':
                		++count_arg;
                		*n = atoi(optarg);
                		break;
                case 'c':
                		++count_arg;
                		*c = atoi(optarg);
                		break;
                case 'g':
                		++count_arg;
                		*g = atoi(optarg);
                		break;
                case 'd':
                		++count_arg;
                		strcpy(dir, optarg);
                		break;
                case 'o':
                		++count_arg;
                		strcpy(out, optarg);
                		break;
                case 't':
                        ++count_arg;
                        *t = atoi(optarg);
                        break;
                case 'f':
                		++count_arg;
                		*f = atof (optarg);
                		break;
                case 'm':
                		++count_arg;
                		*m = atoi (optarg);
                		break;
                default:        /* '?' */
                        usage();
                        exit(-1);
                        break;
                }
        }

        if (count_arg < 3) {
                usage();
                exit (-1);
        }

        if (input[0] == '\0')
        {
        	printf ("Please specify an input fasta or fastq file with -i <file name> \n");
        	exit (-1);
        }
        if (*r <= 0)
        {
        	printf ("Please specify a positive integer for read length with -r <read length>\n");
        	exit (-1);
        }
        if (*k <= 0 || *k > *r)
        {
        	printf ("Please specify the kmer length with an integer in [1, r] with -k <kmer length>\n");
        	exit (-1);
        }
        if (*p > *k)
        {
        	printf ("P-minimum-substring length set error!\n");
        }
        if (*p == 0)
        {
        	if (*k <= 27)
        	{
        		*p = 11;
        	}
        	else
        	{
        		*p = 19;
        	}
        }

        return 0;
}

void read_dbgraph_hashtab (char * file_dir, dbtable_t * tbs, subgraph_t * subgraph, int total_num_partitions, int num_of_devices, int num_of_cpus, int world_size, int world_rank)
{
	int np_per_node = (total_num_partitions + world_size - 1)/world_size;
	int np_node;
	if (world_rank == world_size - 1)
		np_node = total_num_partitions - (world_rank) * np_per_node;
	else
		np_node = np_per_node;
	int start_partition_id = np_per_node * world_rank;
	int end_partition_id = np_per_node * world_rank + np_node;

	int i;
	char fname[FILENAME_LENGTH];
	FILE * file;
	for (i=start_partition_id; i < end_partition_id; i++)
	{
		memset(fname, 0, sizeof(char) * FILENAME_LENGTH);
		sprintf (fname, "%s/sub%d_%d", file_dir, i, total_num_partitions);
		if ((file = fopen (fname, "r")) == NULL)
		{
			printf ("CANNOT OPEN partition file %d!\n", i);
			// exit(0);
		}
		voff_t size;
		voff_t num_hash_elems;
		fread (&size, sizeof(voff_t), 1, file);
		fread (&num_hash_elems, sizeof(voff_t), 1, file);
		tbs[i-start_partition_id].buf = (entry_t *) malloc (sizeof(entry_t) * size);
		tbs[i-start_partition_id].size = size;
		tbs[i-start_partition_id].num_elems = num_hash_elems;
		(subgraph->subgraphs)[i-start_partition_id].size = size;
		(subgraph->subgraphs)[i-start_partition_id].id = i;
		fread (tbs[i-start_partition_id].buf, sizeof(entry_t), size, file);
/*		int j;
		for (j=0; j<10; j++)
		{
			printf ("%u, %u; %lu\n", tbs[i-start_partition_id].buf[j].kmer.x, tbs[i-start_partition_id].buf[j].kmer.y, tbs[i-start_partition_id].buf[j].edge);
		}*/
//		printf ("hashtable %d: num_of_elems = %u, size = %u\n",  i, num_hash_elems, (subgraph->subgraphs)[i-start_partition_id].size);
	}
}

void return_dbgraph_hashtab (dbtable_t * tbs, subgraph_t * subgraph, node_t * tab, voff_t size, voff_t elem_hashed, int pid)
{
	tbs[pid].buf = (entry_t *)tab;
	tbs[pid].size = size;
	tbs[pid].num_elems = elem_hashed;
	(subgraph->subgraphs)[pid].size = size;
	(subgraph->subgraphs)[pid].id = pid;
}

void free_dbgraph_hashtab (int num_of_partitions, dbtable_t * tbs)
{
/*	int i;
	for (i=0; i<num_of_partitions; i++)
	{
		free(tbs[i].buf);
	}*/
	free(tbs);
}

void allgather_subgraph_sizes (int total_num_partitions, d_jvs_t * js, d_lvs_t *ls, master_t * mst, int world_size, int world_rank)
{
	int np_per_node = (total_num_partitions + world_size - 1)/world_size;
	int num_of_partitions; // this is the real number of partitions in this compute node
	if (world_rank == world_size - 1)
		num_of_partitions = total_num_partitions - (world_rank) * np_per_node;
	else
		num_of_partitions = np_per_node;
	int start_partition_id = np_per_node*world_rank;
	goffset_t * tmp_id_offsets = (goffset_t *) malloc (sizeof(goffset_t) * num_of_partitions);
	goffset_t * tmp_jid_offset = (goffset_t *) malloc (sizeof(goffset_t) * num_of_partitions);

	int i;
	for (i = 0; i < num_of_partitions; i++)
	{
		tmp_id_offsets[i] = ls[i].esize + ls[i].asize;
		tmp_jid_offset[i] = js[i].size;
	}
	printf ("MPI ALLGATHER ID OFFSETS::::::::::::::\n");
	mpi_allgatherv(tmp_id_offsets, mst->id_offsets + 1, total_num_partitions, world_size, world_rank, sizeof(goffset_t));
	mpi_allgatherv(tmp_jid_offset, mst->jid_offset, total_num_partitions, world_size, world_rank, sizeof(goffset_t));
	mst->id_offsets[0] = 0;
	if (world_size == 1)
	{
		for (i=0; i<total_num_partitions; i++)
		{
			mst->jid_offset[i] = tmp_jid_offset[i];
			mst->id_offsets[i+1] = tmp_id_offsets[i];
		}
	}
	for (i = 1; i < total_num_partitions; i++)
	{
		mst->id_offsets[i+1] += mst->id_offsets[i];
	}

	free (tmp_id_offsets);
	free (tmp_jid_offset); //free it after gathering contigs?????????
}

void read_subgraph_sizes (subgraph_t * subgraph, int total_num_partitions, \
		d_jvs_t * js, d_lvs_t *ls, master_t * mst, int world_size, int world_rank)
{
	FILE * file;
	FILE * jfile;
	char fname[FILENAME_LENGTH];
	char jname[FILENAME_LENGTH];

	voff_t jsize, lsize;
	char * file_dir = mst->file_dir;

	int np_per_node = (total_num_partitions + world_size - 1)/world_size;
	int num_of_partitions; // this is the real number of partitions in this compute node
	if (world_rank == world_size - 1)
		num_of_partitions = total_num_partitions - (world_rank) * np_per_node;
	else
		num_of_partitions = np_per_node;
	int start_partition_id = np_per_node*world_rank;
	goffset_t * tmp_id_offsets = (goffset_t *) malloc (sizeof(goffset_t) * num_of_partitions);
	goffset_t * tmp_jid_offset = (goffset_t *) malloc (sizeof(goffset_t) * num_of_partitions);

	d_jvs_t * djs = js;
	d_lvs_t * dls = ls;

	int i, t;
	for (i = 0; i < num_of_partitions; i++)
	{
		memset(fname, 0, sizeof(char) * FILENAME_LENGTH);
		memset(jname, 0, sizeof(char) * FILENAME_LENGTH);
		sprintf (fname, "%s/lv%d_%d", mst->file_dir, i+start_partition_id, total_num_partitions);
		sprintf (jname, "%s/jv%d_%d", mst->file_dir, i+start_partition_id, total_num_partitions);

//		if (i==0)
//			printf ("First input file: %s\n", fname);

		if ((file = fopen (fname, "r")) == NULL)
		{
			printf ("OPEN subgraph %d reading file error\n", i+start_partition_id);
			exit(0);
		}

//		if (i==0)
//			printf ("First input file: %s\n", jname);
		if ((jfile = fopen (jname, "r")) == NULL)
		{
			printf ("OPEN subgraph %d reading file error\n", i+start_partition_id);
			exit(0);
		}

		fread(&lsize, sizeof(voff_t), 1, file);
		dls[i].esize = lsize;
		dls[i].asize = 0;

		fread(&jsize, sizeof(voff_t), 1, jfile);
		djs[i].size = jsize;

		(subgraph->subgraphs)[i].id = i+start_partition_id;
		(subgraph->subgraphs)[i].size = lsize;
		tmp_id_offsets[i] = jsize + lsize;
		tmp_jid_offset[i] = jsize;

//		printf ("#######partition %d: jsize = %u, lsize = %u\n", i, jsize, lsize);
		fclose (file);
		fclose (jfile);
	}

//	printf ("WORLD RANK %d: MPI ALLGATHER ID OFFSETS::::::::::::::\n", mst->world_rank);
	mpi_allgatherv(tmp_id_offsets, mst->id_offsets + 1, total_num_partitions, world_size, world_rank, sizeof(goffset_t));
	mpi_allgatherv(tmp_jid_offset, mst->jid_offset, total_num_partitions, world_size, world_rank, sizeof(goffset_t));
	mst->id_offsets[0] = 0;
	if (world_size == 1)
	{
		for (i=0; i<total_num_partitions; i++)
		{
			mst->jid_offset[i] = tmp_jid_offset[i];
			mst->id_offsets[i+1] = tmp_id_offsets[i];
		}
	}
	for (i = 1; i < total_num_partitions; i++)
	{
		mst->id_offsets[i+1] += mst->id_offsets[i];
	}

	free (tmp_id_offsets);
	free (tmp_jid_offset);
//	printf ("SSSSSSSSSSSSSSSSSSSSSSSS statistics: jsize = %u, lsize = %u\n", total_jsize, total_lsize);
//	printf ("TTTTTTTTTTTTTTTTTTTTTTTT test id offsets: \n");
//	print_offsets(mst->id_offsets, total_num_partitions+1);
//	print_offsets(mst->jid_offset, total_num_partitions);
}

void read_junctions (int total_num_partitions, d_jvs_t * js, master_t * mst, int world_size, int world_rank)
{
	FILE * file;
	int i, t;
	char fname[FILENAME_LENGTH];
	char * file_dir = mst->file_dir;

	voff_t jsize;
	d_jvs_t * djs = js;

	int np_per_node = (total_num_partitions + world_size - 1)/world_size;
	int num_of_partitions; // this is the real number of partitions in this compute node
	if (world_rank == world_size - 1)
		num_of_partitions = total_num_partitions - (world_rank) * np_per_node;
	else
		num_of_partitions = np_per_node;
	int start_partition_id = np_per_node*world_rank;

	for (i = 0; i < num_of_partitions; i++)
	{
		memset(fname, 0, sizeof(char) * FILENAME_LENGTH);
		sprintf (fname, "%s/jv%d_%d", mst->file_dir, i+start_partition_id, total_num_partitions);

//		if (i==0)
//			printf ("First input file: %s\n", fname);
//		printf ("Input file: %d\n", i+start_partition_id);
		if ((file = fopen (fname, "r")) == NULL)
		{
			printf ("OPEN subgraph %d reading file error\n", i+start_partition_id);
			// exit(0);
		}
		fread(&jsize, sizeof(voff_t), 1, file);
		jsize = djs[i].size;

		djs[i].id = NULL;
		for (t = 0; t < EDGE_DIC_SIZE; t++)
		{
			djs[i].nbs[t] = (vid_t *) malloc (sizeof(vid_t) * jsize);
			CHECK_PTR_RETURN (djs[i].nbs[t], "malloc djs[%d].nbs[%d] array error!\n", i, t);
		}

		for (t = 0; t < EDGE_DIC_SIZE; t++)
		{
			fread(djs[i].nbs[t], sizeof(vid_t), jsize, file);
		}

		fclose (file);
	}
}

void read_linear_vertices (int total_num_partitions, d_lvs_t * ls, master_t * mst, int world_size, int world_rank)
{
	FILE * file;
	int i;
	char fname[FILENAME_LENGTH];
	char * file_dir = mst->file_dir;

	voff_t lsize;
	d_lvs_t * dls = ls;

	int np_per_node = (total_num_partitions + world_size - 1)/world_size;
	int num_of_partitions; // this is the real number of partitions in this compute node
	if (world_rank == world_size - 1)
		num_of_partitions = total_num_partitions - (world_rank) * np_per_node;
	else
		num_of_partitions = np_per_node;
	int start_partition_id = np_per_node*world_rank;

	for (i = 0; i < num_of_partitions; i++)
	{
		memset(fname, 0, sizeof(char) * FILENAME_LENGTH);
		sprintf (fname, "%s/lv%d_%d", mst->file_dir, i+start_partition_id, total_num_partitions);

//		if (i==0)
//			printf ("First input file: %s\n", fname);
//		printf ("Input file: %d\n", i+start_partition_id);
		if ((file = fopen (fname, "r")) == NULL)
		{
			printf ("OPEN subgraph %d reading file error\n", i+start_partition_id);
			// exit(0);
		}
		fread(&lsize, sizeof(voff_t), 1, file);
		lsize = dls[i].asize + dls[i].esize;
		dls[i].id = NULL;
		dls[i].pres = (vid_t *) malloc (sizeof(vid_t) * lsize);
		CHECK_PTR_RETURN (dls[i].pres, "malloc dls[%d].pres array error!\n", i);
		dls[i].posts = (vid_t *) malloc (sizeof(vid_t) * lsize);
		CHECK_PTR_RETURN (dls[i].posts, "malloc dls[%d].posts array error!\n", i);

		fread(dls[i].posts, sizeof(voff_t), lsize, file);
		fread(dls[i].pres, sizeof(voff_t), lsize, file);
		fclose (file);
	}
}

void read_kmers_edges_for_gather_contig (int total_num_partitions, d_jvs_t * js, d_lvs_t * ls, master_t * mst, int world_size, int world_rank)
{
	FILE * jfile;
	FILE * lfile;
	FILE * jefile;
	int i, t;
	char jname[FILENAME_LENGTH];
	char lname[FILENAME_LENGTH];
	char jename [FILENAME_LENGTH];
	char * file_dir = mst->file_dir;

	voff_t jsize;
	voff_t lsize;
	d_jvs_t * djs = js;
	d_lvs_t * dls = ls;

	int np_per_node = (total_num_partitions + world_size - 1)/world_size;
	int num_of_partitions; // this is the real number of partitions in this compute node
	if (world_rank == world_size - 1)
		num_of_partitions = total_num_partitions - (world_rank) * np_per_node;
	else
		num_of_partitions = np_per_node;
	int start_partition_id = np_per_node*world_rank;

	for (i = 0; i < num_of_partitions; i++)
	{
		memset(jname, 0, sizeof(char) * FILENAME_LENGTH);
		memset(lname, 0, sizeof(char) * FILENAME_LENGTH);
		memset(jename, 0, sizeof(char) * FILENAME_LENGTH);
		sprintf (jname, "%s/jkmer%d_%d", mst->file_dir, i+start_partition_id, total_num_partitions);
		sprintf (lname, "%s/ledge%d_%d", mst->file_dir, i+start_partition_id, total_num_partitions);
		sprintf (jename, "%s/jedge%d_%d", mst->file_dir, i+start_partition_id, total_num_partitions);

		jsize = djs[i].size;
		djs[i].kmers = (kmer_t *) malloc (sizeof(kmer_t) * jsize);
		CHECK_PTR_RETURN (djs[i].kmers, "malloc djs[%d].kmer array error!\n", i);
		djs[i].edges = (ull *) malloc (sizeof(ull) * jsize);
		CHECK_PTR_RETURN (djs[i].edges, "malloc djs[%d].edge array error!\n", i);

		if ((jfile = fopen (jname, "r")) == NULL)
		{
			printf ("open subgraph %d kmer file error!\n", i+start_partition_id);
			exit(0);
		}
		fread(djs[i].kmers, sizeof(kmer_t), jsize, jfile);

		if ((jefile = fopen (jename, "r")) == NULL)
		{
			printf ("open subgraph %d edge file error!\n", i+start_partition_id);
			exit(0);
		}
		fread(djs[i].edges, sizeof(ull), jsize, jefile);

		lsize = dls[i].asize + dls[i].esize;
		dls[i].pre_edges = (edge_type *) malloc (sizeof(edge_type) * lsize);
		CHECK_PTR_RETURN (dls[i].pre_edges, "malloc dls[%d].pre_edges array error!\n", i);
		dls[i].post_edges = (edge_type *) malloc (sizeof(edge_type) * lsize);
		CHECK_PTR_RETURN (dls[i].post_edges, "malloc dls[%d].post_edges array error!\n", i);

		if ((lfile = fopen (lname, "r")) == NULL)
		{
			printf ("open subgraph %d kmer file error!\n", i+start_partition_id);
		}
		fread(dls[i].pre_edges, sizeof(edge_type), lsize, lfile);
		fread(dls[i].post_edges, sizeof(edge_type), lsize, lfile);

		fclose(jfile);
		fclose(jefile);
		fclose(lfile);
	}
}

// load junctions in a compute node, get statistics for mst
void junction_csr (d_jvs_t * js, int num_of_partitions, master_t * mst, subgraph_t * subgraph)
{
	if (num_of_partitions != subgraph->num_of_subs) // number of partitions residing in a computer node
	{
		printf ("Warning!!! number of subgraphs in this computer node is different from records in subgraph_t!!!\n");
		// exit(0);
	}
	subgraph->num_jnbs = (vid_t *) malloc (sizeof(vid_t) * num_of_partitions);
	CHECK_PTR_RETURN (subgraph->num_jnbs, "malloc number of neighbor of junctions in subgraph_t error!\n");

	voff_t * tmp = (voff_t *) malloc (sizeof(voff_t) * (num_of_partitions+1));
	CHECK_PTR_RETURN (tmp, "malloc tmp for junctions csr error!\n");
	js->csr_offs_offs = (voff_t *) malloc (sizeof(voff_t) * (num_of_partitions + 1));
	CHECK_PTR_RETURN (js->csr_offs_offs, "malloc joffsets to record offsets of partitions in each processor error!\n");
	voff_t * joffsets = js->csr_offs_offs;
	js->csr_nbs_offs = (voff_t *) malloc (sizeof(voff_t) * (num_of_partitions + 1));
	CHECK_PTR_RETURN (js->csr_nbs_offs, "malloc jnboffsets to record offsets of partitions in each processor error!\n");
	voff_t * jnboffsets = js->csr_nbs_offs;
	js->csr_spids_offs = (voff_t *) malloc (sizeof(voff_t) * (num_of_partitions + 1));
	CHECK_PTR_RETURN (js->csr_spids_offs, "malloc csr spids offs error!\n");
	voff_t * spids_offs = js->csr_spids_offs;

	int i;
	ull total_num_js = 0;
	ull total_num_ns = 0;
	tmp[0] = 0;
	joffsets[0] = 0;
	jnboffsets[0] = 0;
	spids_offs[0] = 0;
	size_t csr_spids_size = 0;
	for (i=0; i<num_of_partitions; i++)
	{
		total_num_js += js[i].size;
		(subgraph->subgraphs)[i].size = js[i].size;
	}
//#pragma omp parallel for reduction(+:total_num_ns)
	for (i=0; i<num_of_partitions; i++)
	{
		voff_t num_ns = 0;
		voff_t j;
		int e;
		for (j=0; j<js[i].size; j++)
		{
			for (e=0; e<EDGE_DIC_SIZE; e++)
			{
				if (js[i].nbs[e][j] != DEADEND)
				{
					num_ns++;
				}
			}
		}
		tmp[i+1] = num_ns;
		spids_offs[i + 1] = num_ns/2 + num_ns%2 + spids_offs[i];
		jnboffsets[i + 1] = num_ns + jnboffsets[i];
		joffsets[i + 1] = js[i].size + joffsets[i];
		total_num_ns += num_ns;
		(subgraph->num_jnbs)[i] = num_ns;
	}
	csr_spids_size = spids_offs[num_of_partitions];
	printf ("WORLD RANK %d: total number of junctions: %lu, total number of neighbors of junctions: %lu\n", mst->world_rank, total_num_js, total_num_ns);
	js->csr_offs = (voff_t *) malloc (sizeof(voff_t) * (total_num_js + num_of_partitions));
	CHECK_PTR_RETURN (js->csr_offs, "malloc junction neighbor offsets error!\n");
	voff_t * joffs = js->csr_offs;
	js->csr_nbs = (vid_t *) malloc (sizeof(vid_t) * total_num_ns);
	CHECK_PTR_RETURN (js->csr_nbs, "malloc junction neighbor array error!\n");
	vid_t * jnbs = js->csr_nbs;
	inclusive_prefix_sum (tmp, num_of_partitions+1);
	js->csr_spids = (uint *) malloc (sizeof(uint) * csr_spids_size);
	CHECK_PTR_RETURN (js->csr_spids, "malloc junction spids for neighbors of junctions error!\n");
	memset (js->csr_spids, 0, sizeof(uint) * csr_spids_size);
	uint * csr_spids = js->csr_spids;


	for (i=0; i<num_of_partitions; i++)
	{
		voff_t j;
		int e;
		voff_t num_ns = 0;
		voff_t offset = tmp[i];
		voff_t ns;
		joffs[0] = 0;
		ull * spids = js[i].spids;
		ull * spidsr = js[i].spidsr;
		for (j=0; j<js[i].size; j++)
		{
			ns = 0;
			for (e=0; e<EDGE_DIC_SIZE/2; e++)
			{
				if (js[i].nbs[e][j] != DEADEND)
				{
					jnbs[offset+num_ns] = js[i].nbs[e][j];
					csr_spids[get_spid_index(num_ns)] |= (spids[j] >> (e*SPID_BITS) & SPID_MASK) << ((get_spid_unit_offset(num_ns)) * SPID_BITS);
					num_ns++;
					ns++;
				}
			}
			for (e=EDGE_DIC_SIZE/2; e<EDGE_DIC_SIZE; e++)
			{
				if (js[i].nbs[e][j] != DEADEND)
				{
					jnbs[offset+num_ns] = js[i].nbs[e][j];
					csr_spids[get_spid_index(num_ns)] |= (spidsr[j] >> ((e-EDGE_DIC_SIZE/2)*SPID_BITS) & SPID_MASK) << ((get_spid_unit_offset(num_ns)) * SPID_BITS);
					num_ns++;
					ns++;
				}
			}
			joffs[j+1] = ns;
		}
		tbb_scan_uint (joffs, joffs, js[i].size+1);
		joffs += js[i].size+1;
		csr_spids += num_ns/2 + num_ns%2;
	}

	free(tmp);
}

void get_junction_info_processors (master_t * mst, subgraph_t * subgraph)
{
	int num_of_cpus = mst->num_of_cpus;
	int num_of_devices = mst->num_of_devices;
	int world_size = mst->world_size;
	int world_rank = mst->world_rank;
	int total_num_partitions = mst->total_num_partitions;

	int np_per_node;
	int np_node;
	get_np_node (&np_per_node, &np_node, total_num_partitions, world_size, world_rank);

	int start_partition_id = world_rank * np_per_node;
	int i;
	for (i=0; i<num_of_cpus+num_of_devices; i++)
	{
		mst->num_js[i] = 0;
		mst->num_jnbs[i] = 0;
		mst->spids_size[i] = 0;
//		int start_id_processor = mst->num_partitions[i];
		int j;
		for (j=mst->num_partitions[i]; j<mst->num_partitions[i+1]; j++)
		{
			int pid = mst->partition_list[j];
			mst->num_js[i] += (subgraph->subgraphs)[pid - start_partition_id].size;
			mst->num_jnbs[i] += (subgraph->num_jnbs)[pid - start_partition_id];
			mst->spids_size[i] += (subgraph->num_jnbs)[pid - start_partition_id]/2 + (subgraph->num_jnbs)[pid - start_partition_id]%2;
		}
	}
}

void free_junction_csr (d_jvs_t * js, subgraph_t * subgraph)
{
	free (js->csr_nbs);
	free (js->csr_nbs_offs);
	free (js->csr_offs);
	free (js->csr_offs_offs);
	free_subgraph_num_jnbs (subgraph);
}

void free_adj_linear (int num_of_partitions, d_lvs_t * dls)
{
	int i;
	for (i=0; i<num_of_partitions; i++)
	{
		free(dls[i].posts);
		free(dls[i].pres);
	}
}

void free_adj_junction (int num_of_partitions, d_jvs_t * djs)
{
	int i;
	for (i=0; i<num_of_partitions; i++)
	{
		int j;
		for (j=0; j<EDGE_DIC_SIZE; j++)
		{
			free(djs[i].nbs[j]);
		}
	}
}

void free_kmers_edges_after_contig (int num_of_partitions, d_jvs_t * djs, d_lvs_t * dls)
{
	int i;
	for (i=0; i<num_of_partitions; i++)
	{
		free (djs[i].kmers);
		free (djs[i].edges);
		free (dls[i].post_edges);
		free (dls[i].pre_edges);
	}
	free (djs);
	free(dls);
}

