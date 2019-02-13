/*
 * main.c
 *
 *  Created on: 2015-7-21
 *      Author: qiushuang
 */
#include <unistd.h>
#include <sys/stat.h>
#include <limits.h>
#include <string.h>
#include "../include/utility.h"
#include "../include/io.h"
#include "../include/msp.h"
#include "../include/dbgraph.h"
#include "../include/comm.h"
#include "../include/malloc.h"
#include "../include/io.h"
#include "../include/distribute.h"

void msp_partition (char *, char *, int, int, int, int, int, int, int, int);
void construct_dbgraph_hetero (int, int, char *, char *, char *, int, int, int, subgraph_t *, dbtable_t *, int, int);
void pre_process_dbgraph (int, int, int, dbtable_t *, master_t *, subgraph_t *, d_jvs_t *, d_lvs_t *, int, int);
void traverse_dbgraph (int, subgraph_t *, d_jvs_t *, d_lvs_t *, master_t *, int, int, int);


int mpi_run = 1;
int cutoff = 1;

float elem_factor = 1;
extern int read_length;
float inmemory_time = 0;
extern ull kmer_count[4];
long cpu_threads;

void preprocessing (char * hash_dir, char * contig_dir, subgraph_t * subgraph, dbtable_t * tbs, int total_num_partitions, int k, int p, int num_of_devices, int num_of_cpus, int world_size, int world_rank)
{
		evaltime_t start, end;

		master_t mst;
		mst.file_dir = hash_dir;
		mst.contig_dir = contig_dir;
		mst.num_of_cpus = num_of_cpus;
		mst.num_of_devices = num_of_devices;
		mst.total_num_partitions = total_num_partitions;
		mst.world_size = world_size;
		mst.world_rank = world_rank;
		mst.id_offsets = (goffset_t *)malloc(sizeof(goffset_t) * (total_num_partitions+1));
		mst.jid_offset = (goffset_t *)malloc(sizeof(goffset_t) * total_num_partitions);
		memset (mst.id_offsets, 0, sizeof(goffset_t) * (total_num_partitions + 1));
		memset (mst.jid_offset, 0, sizeof(goffset_t) * total_num_partitions);
		init_counts_displs(total_num_partitions, world_size, world_rank);

		int np_per_node;
		int num_of_partitions; // this is the real number of partitions in this compute node
		get_np_node (&np_per_node, &num_of_partitions, total_num_partitions, world_size, world_rank);
		d_jvs_t *djs = (d_jvs_t *) malloc (sizeof(d_jvs_t) * num_of_partitions); // freed in free_kmers_edges_after_contig
		CHECK_PTR_RETURN (djs, "malloc junction pointers error!\n"); // freed in free_kmers_edges_after_contig
		d_lvs_t *dls = (d_lvs_t *) malloc (sizeof(d_lvs_t) * num_of_partitions);
		CHECK_PTR_RETURN (dls, "malloc linear node pointers error!\n");

#ifdef USE_DISK_IO
		printf ("WORLD RANK %d: BBBBBBBBBBBBBBB begin: read hash tables:\n", world_rank);
		gettimeofday (&start, NULL);
		read_dbgraph_hashtab (hash_dir, tbs, subgraph, total_num_partitions, num_of_devices, num_of_cpus, world_size, world_rank);
		gettimeofday (&end, NULL);
		print_exec_time (start, end, "WORLD RANK %d: Reading hash tables time: ", world_rank);
#endif
		printf("\nPre-process De Bruijn graph:\n", world_rank);
		gettimeofday (&start, NULL);
		pre_process_dbgraph (total_num_partitions, k, p, tbs, &mst, subgraph, djs, dls, world_size, world_rank);
		gettimeofday (&end, NULL);
		print_exec_time (start, end, "Pre-process time: ", world_rank);

		printf ("\nWWWWWWWWWWWWWWord rank %d: overall in-memory graph processing time meaured: %f\n", inmemory_time);

		printf ("WORLD RANK %d: BBBBBBBBBBBBBBB begin: read adj graph:\n", world_rank);
#ifdef USE_DISK_IO
		read_subgraph_sizes (subgraph, total_num_partitions, djs, dls, &mst, world_size, world_rank);
		read_junctions (total_num_partitions, djs, &mst, world_size, world_rank);
		read_linear_vertices (total_num_partitions, dls, &mst, world_size, world_rank);
#endif
		/* traverse de bruijn graph: */
		printf("\nWORLD RANK %d: Traverse De Bruijn graph:\n", world_rank);
		gettimeofday (&start, NULL);
		traverse_dbgraph (total_num_partitions, subgraph, djs, dls, &mst, k, world_size, world_rank);
		gettimeofday (&end, NULL);
		print_exec_time (start, end, "WORLD RANK %d: Traversal time: ", world_rank);

		printf ("\nWWWWWWWWWWWWWWord rank %d: overall in-memory graph processing time meaured: %f\n", inmemory_time);

		finalize_counts_displs ();

		free(mst.id_offsets);
		free(mst.jid_offset);
}

int
main (int argc, char ** argv)
{
	evaltime_t start, end;

	read_buf_t reads;
	msp_t msp_arr;
	uch * ptr;
	uint size;

	char input_file[PATH_MAX];
	memset (input_file, 0, sizeof(char) * PATH_MAX);
	char msp_dir[PATH_MAX];
	memset (msp_dir, 0, sizeof(char) * PATH_MAX);
	char hash_dir[PATH_MAX];
	memset (hash_dir, 0, sizeof(char) * PATH_MAX);
	char contig_dir[PATH_MAX];
	memset (contig_dir, 0, sizeof(char) * PATH_MAX);
	int num_of_partitions = 512;
	int num_of_hash_cpus = 1;
	int num_of_hash_devices = get_device_config();
	int k;
	int p=0;
	float factor = 0.8;
	if (get_opt(argc, argv, input_file, &read_length, &k, &p, &num_of_partitions, \
			&num_of_hash_cpus, &num_of_hash_devices, msp_dir, contig_dir, &cutoff, &factor, &mpi_run) != 0)
	{
		usage();
		exit(-1);
	}

	if (read_length > 121 + k)
	{
		printf ("!!! ERROR: the read length %d exceeds the maximum read length allowed %d (121+k) in UNIPAR!\n", read_length, 121+k);
		exit (0);
	}
	if (num_of_partitions > NUM_OF_PARTITIONS)
	{
		printf ("!!! ERROR: the number of partitions %d exceeds the maximum number of partitions allowed in UNIPAR!\n", num_of_partitions);
		exit (0);
	}
	if (num_of_hash_cpus == 0 && num_of_hash_devices == 0)
	{
		num_of_hash_cpus = 1;
		printf ("Warning: the number of hash cpus and devices specified are both 0; the number of hash cpus was set to 1!\n");
	}
	if (num_of_hash_devices > NUM_OF_HASH_DEVICES)
	{
		printf ("!!! ERROR: the number of hash devices %d exceeds the maximum number of hash devices allowed in UNIPAR!\n", num_of_hash_devices);
		exit (0);
	}
	if (num_of_hash_cpus > NUM_OF_HASH_CPUS)
	{
		printf ("!!! ERROR: the number of hash cpus %d exceeds the maximum number of hash cpus allowed in UNIPAR!\n", num_of_hash_cpus);
		exit (0);
	}
	if (factor != 0)
	{
		elem_factor = factor;
	}
	int num_of_msp_devices = num_of_hash_devices;
	int num_of_msp_cpus = num_of_hash_cpus;

	int world_size;
	int world_rank;
	if (mpi_run == 0)
	{
		world_size = 1;
		world_rank = 0;
	}
	else
	{
		int provided;
		mpi_init (&provided, &world_size, &world_rank);
	}

	char cwd[PATH_MAX];
	if (getcwd(cwd, sizeof(cwd)) != NULL) {
		if(world_rank == 0)
	       printf("Current working dir: %s\n", cwd);
	} else {
	       perror("getcwd() error");
	       exit(0);
	}

	if (contig_dir[0] == '\0')
	{
		strcpy (contig_dir, cwd);
	}
	strcat (cwd, "/partitions");
	if (msp_dir[0] == '\0')
	{
		struct stat sb = {0};
		if (stat(cwd, &sb) == 0 && S_ISDIR(sb.st_mode))
		{
			strcpy (msp_dir, cwd);
			strcpy (hash_dir, cwd);
		}
		else
		{
			mkdir (cwd, 0700);
			strcpy (msp_dir, cwd);
			strcpy (hash_dir, cwd);
		}
	}

	if (world_rank == 0)
	{
	printf ("UNIPAR running with parameters:\n"
			"read length: %d\n"
			"kmer length: %d\n"
			"minimum substring partition - common length p: %d\n"
			"number of partitions: %d\n"
			"number of hash devices: %d\n"
			"number of cpus: %d\n"
			"mpi_run: %d\n"
			"input file: %s\n"
			"minimum substring partitioning directory: %s\n"
			"hash table directory: %s\n"
			"contig output directory: %s\n"
			"hash element factor: %f\n",
			read_length, k, p, num_of_partitions, num_of_hash_devices, num_of_hash_cpus, mpi_run,\
			input_file, msp_dir, hash_dir, contig_dir, elem_factor);
	}

	cpu_threads = sysconf(_SC_NPROCESSORS_ONLN);
	if (cpu_threads > MAX_NUM_THREADS)
	{
		printf ("Error: pre-set maximum number of threads smaller than the real number of CPU cores %d!"
				"Please set the macro MAX_NUM_THREADS to be no less than %d in the file msp.h!\n", cpu_threads, cpu_threads);
	}
	printf ("WORLD_RANK %d: ############ Number of CPU cores found on the machine: %ld\n", world_rank, cpu_threads);
	int np_per_node, np_node;
	get_np_node (&np_per_node, &np_node, num_of_partitions, world_size, world_rank);
	subgraph_t subgraph;
	malloc_subgraph_subgraphs (&subgraph, np_node);
	dbtable_t * tbs = (dbtable_t *) malloc (sizeof(dbtable_t) * np_node);

	evaltime_t overs, overe;
	gettimeofday(&overs, NULL);
	/* Minimum substring partition */
	printf("\nWORLD RANK %d: Minimum substring partition:\n", world_rank);
	gettimeofday (&start, NULL);
	msp_partition (input_file, msp_dir, k, p, read_length, num_of_partitions, num_of_msp_devices, num_of_msp_cpus, world_size, world_rank);
	gettimeofday (&end, NULL);
	print_exec_time (start, end, "WORLD RANK %d: Total Minimum substring partition time: ", world_rank);

	//******** barrier here with multi-processes to make sure the partitions are ready **********
	if (mpi_run > 0)
		mpi_barrier();
	/* construct graph: */
	printf("\nWORLD RANK %d: Construct graph:\n", world_rank);
	gettimeofday (&start, NULL);
	construct_dbgraph_hetero (k, p, "out", msp_dir, hash_dir, num_of_partitions, num_of_hash_devices, num_of_hash_cpus, &subgraph, tbs, world_size, world_rank);
	gettimeofday (&end, NULL);
	print_exec_time (start, end, "WORLD RANK %d: Constructing graph time: ", world_rank);
	ull total_dkmers = 0;
	int i;
	for (i=0; i<num_of_hash_devices; i++)
		total_dkmers += kmer_count[i];
	printf ("!!!!!!!!!! total number of kmers processed on devices: %lu\n", total_dkmers);

	printf ("\nWORLD RANK %d: Preprocessing graph: \n", world_rank);
	preprocessing (hash_dir, contig_dir, &subgraph, tbs, num_of_partitions, k, p, num_of_hash_devices, num_of_hash_cpus, world_size, world_rank);
	gettimeofday(&overe, NULL);
	print_exec_time(overs, overe, "Overall running time with UNIPAR: \n");

	free_subgraph_subgraphs (&subgraph);
	if (mpi_run)
		mpi_finalize();

	return 0;
}
