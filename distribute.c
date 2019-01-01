/*
 * distribute.c
 *
 *  Created on: 2018-3-6
 *      Author: qiushuang
 */


#include <mpi.h>
#include <math.h>
#include "include/utility.h"
#include "include/dbgraph.h"
#include "include/comm.h"
#include "include/io.h"
#include "include/distribute.h"

#define ALL2ALL_SEND_EXPAND 8 // corresponds to the number of processors
#define MAX_DEVICE_MSIZE (12*1073741824UL) //12GB DEVICE MEMORY SET

static int * sdispls;
static int * rdispls;
static int * scounts;
static int * rcounts;
static int * tmp_partition_list = NULL;
// sizeof total_num_of_partitions, gathered from all the processes to know the partition distribution among compute nodes, while mst->partition_list is the list of partitions within this compute node
static voff_t * tmp_offset_counts = NULL; // used to count the number of elems of each partition from multiple processors in each process
static voff_t * tmp_gathered_offsets = NULL; // gathered of partition offsets from all the processes
static void * tmp_all2all_send = NULL; // send buffer for each process to store messages of each inter-compute-node partition
static void * tmp_all2all_receive = NULL; // receive buffer in each processor to store message of each partition within this compute node, from each other compute node
static voff_t * tmp_receive_offsets = NULL; // receive offset buffer
static int init_all2all = 0;
static ull tmp_all2all_send_size = 0;
static ull tmp_all2all_receive_size = 0;
float comm_time = 0;

static evaltime_t start, end;

int debug = 0;

void get_np_node (int * np_per_node, int * np_node, int total_num_partitions, int world_size, int world_rank)
{
	*np_per_node = (total_num_partitions + world_size - 1)/world_size;
	if (world_rank == world_size - 1)
		*np_node = total_num_partitions - world_rank * (*np_per_node);
	else
		*np_node = *np_per_node;
}

void mpi_barrier (void)
{
	MPI_Barrier (MPI_COMM_WORLD);
}

int mpi_allsum (const void * send, void * receive)
{
	return MPI_Allreduce (send, receive, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
}

int mpi_init (int * provided, int * world_size, int * world_rank)
{
	MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, provided);
	MPI_Comm_size(MPI_COMM_WORLD, world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, world_rank);
	return *provided;
}

void mpi_finalize (void)
{
	MPI_Finalize();
}

void init_counts_displs (int total_num_of_partitions, int world_size, int world_rank)
{
	sdispls = (int *) malloc (sizeof(int) * total_num_of_partitions);
	CHECK_PTR_RETURN (sdispls, "malloc send displacement array for partition list error!\n");
	rdispls = (int *) malloc (sizeof(int) * total_num_of_partitions);
	CHECK_PTR_RETURN (rdispls, "malloc receive displacement array for partition list error!\n");
	scounts = (int *) malloc (sizeof(int) * total_num_of_partitions);
	CHECK_PTR_RETURN (scounts, "malloc send counts array error!\n");
	rcounts = (int *) malloc (sizeof(int) * total_num_of_partitions);
	CHECK_PTR_RETURN (rcounts, "malloc receive counts array error!\n");
	memset (sdispls, 0, sizeof(int) * total_num_of_partitions);
	memset (rdispls, 0, sizeof(int) * total_num_of_partitions);
	memset (scounts, 0, sizeof(int) * total_num_of_partitions);
	memset (rcounts, 0, sizeof(int) * total_num_of_partitions);

	int np_per_node;
	int np_node;
	get_np_node (&np_per_node, &np_node, total_num_of_partitions, world_size, world_rank);

	printf ("number of subgraphs in this compute node: %d\n", np_node);
}

void finalize_counts_displs (void)
{
	free (sdispls);
	free (rdispls);
	free (scounts);
	free (rcounts);
}

// use it to distribute partitions to processors in a single machine, and to multiple machines
void init_distribute_partitions (int total_num_of_partitions, master_t * mst, int world_size)
{
	memset (mst->num_vs, 0, sizeof(uint) * NUM_OF_PROCS);
	memset (mst->num_partitions, 0, sizeof(int) * (NUM_OF_PROCS + 1));
	mst->r2s = (int *) malloc (sizeof(int) * total_num_of_partitions);
	CHECK_PTR_RETURN (mst->r2s, "malloc r2s array for all the partitions error!\n");
	mst->partition_list = (int *) malloc (sizeof(int) * total_num_of_partitions); // this use be np_node, partition list for each computer node
	CHECK_PTR_RETURN (mst->partition_list, "malloc mst->partition_list error!\n");
	tmp_partition_list = (int *) malloc (sizeof(int) * total_num_of_partitions);
	CHECK_PTR_RETURN (tmp_partition_list, "malloc tmp_partition_list error!\n");

	tmp_gathered_offsets = (voff_t *) malloc (sizeof(voff_t) * (total_num_of_partitions + 1) * world_size);
	CHECK_PTR_RETURN (tmp_gathered_offsets, "malloc tmp_gathered_offsets error!\n");
	tmp_offset_counts = (voff_t *) malloc (sizeof(voff_t) * (total_num_of_partitions+1));
	CHECK_PTR_RETURN (tmp_offset_counts, "malloc tmp_offset_counts error!!!\n");

	memset (tmp_gathered_offsets, 0, sizeof(int) * (total_num_of_partitions + 1) * world_size);
	memset (tmp_offset_counts, 0, sizeof(int) * (total_num_of_partitions + 1));
}

// finalize distribution of all processors
void finalize_distribute_partitions (master_t * mst)
{
	int num_of_procs = mst->num_of_cpus + mst->num_of_devices;
	int i;
	for (i = 0; i < num_of_procs; i++)
	{
		free (mst->not_reside[i]);
		free (mst->id2index[i]);
		free (mst->index2id[i]);
		free (mst->pfrom[i]);
	}
	free (mst->partition_list);
	free (mst->r2s);

	free (tmp_gathered_offsets);
	free (tmp_offset_counts);
	free (tmp_partition_list);
	init_all2all = 0;
}

// use it to record the total number of messages sent to a specific processor
void init_temp_offset_counts (int total_num_of_partitions, master_t * mst)
{
	memset (tmp_offset_counts, 0, sizeof(int) * (total_num_of_partitions + 1));
	int num_of_devices = mst->num_of_devices;
	int num_of_cpus = mst->num_of_cpus;
	int num_of_procs = num_of_devices + num_of_cpus;
	int world_size = mst->world_size;
	int world_rank = mst->world_rank;
	int np_per_node;
	int np_node;
	get_np_node (&np_per_node, &np_node, total_num_of_partitions, world_size, world_rank);

	int start_partition_id = np_per_node * world_rank;
	int end_partition_id = np_per_node * world_rank + np_node;
	int i, j;
	for (i=0; i<num_of_procs; i++)
	{
		for(j=0; j<total_num_of_partitions; j++)
		{
			int pid = mst->index2id[i][j];
			if (pid >= start_partition_id && pid < end_partition_id)
				continue;
			tmp_offset_counts[pid] += mst->roff[i][j+1] - mst->roff[i][j];
		}
	}
}

void * mpi_allgather_offsets (int total_num_of_partitions, int world_size, int world_rank)
{
	memset (rcounts, 0, sizeof(int) * total_num_of_partitions);
	memset (tmp_gathered_offsets, 0, sizeof(int) * (total_num_of_partitions + 1) * world_size);

	gettimeofday(&start, NULL);
	if (world_size > 1)
		MPI_Allgather(tmp_offset_counts, total_num_of_partitions, MPI_INT, tmp_gathered_offsets, total_num_of_partitions, MPI_INT, MPI_COMM_WORLD);
	gettimeofday(&end, NULL);
	comm_time += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;

	return tmp_gathered_offsets;
}

// this function is to initialize receive buffers for messages from other processors
void * init_count_displs_receive_all2all (int total_num_of_partitions, master_t * mst, voff_t * gathered_offsets)
{
	int world_size = mst->world_size;
	int world_rank = mst->world_rank;
	int mssg_size = mst->mssg_size;
	int num_of_procs = mst->num_of_cpus + mst->num_of_devices;
	memset (sdispls, 0, sizeof(int) * total_num_of_partitions);
	memset (rdispls, 0, sizeof(int) * total_num_of_partitions);
	memset (scounts, 0, sizeof(int) * total_num_of_partitions);
	memset (rcounts, 0, sizeof(int) * total_num_of_partitions);

	int np_node;
	int np_per_node;
	get_np_node (&np_per_node, &np_node, total_num_of_partitions, world_size, world_rank);

	int i, j;
	scounts[world_rank] = 0;
	for (i=0; i<world_size; i++)
	{
		if (i==world_rank)
			continue;
		int start_partition_id = np_per_node * i;
		int end_partition_id;
		if (i==world_size-1)
			end_partition_id = total_num_of_partitions;
		else
			end_partition_id = np_per_node * (i+1);
		for (j=start_partition_id; j<end_partition_id; j++)
		{
			scounts[i] += tmp_offset_counts[j];
		}
	}
	for (i=0; i<world_size-1; i++)
	{
		sdispls[i+1] = sdispls[i] + scounts[i];
	}

	rcounts[world_rank] = 0;
	for (i=0; i<world_size; i++)
	{
		if (i==world_rank)
			continue;
		int * ptr = gathered_offsets + total_num_of_partitions * i;
		int start_partition_id = np_per_node * world_rank;
		int end_partition_id = np_per_node * world_rank + np_node;
		for (j=start_partition_id; j<end_partition_id; j++)
		{
			rcounts[i] += ptr[j]; // be carefull!!!! the gathered offsets are number of elems for each partition, not the offsets!!!
		}
	}
	for (i=0; i<world_size-1; i++)
	{
		rdispls[i+1] = rdispls[i] + rcounts[i];
	}

	voff_t total_send_size = sdispls[world_size-1]+scounts[world_size-1];
//	while (debug==0) {}

	if (init_all2all == 0)
	{
		//********* use send size in this computer node to estimate and malloc buffer size of tmp_all2all_receive **********
		tmp_all2all_receive = (void *) malloc ((ull)mssg_size * total_send_size * world_size * ALL2ALL_SEND_EXPAND);
		printf (":::::::::::::::::: Preallocated number of messages: %u :::::::::::::::::::\n", total_send_size * world_size * ALL2ALL_SEND_EXPAND);
		// Be CAREFUL:::::: using total_send_size to estimate overall receive_size!!!!!!!!
		CHECK_PTR_RETURN (tmp_all2all_receive, "malloc tmp_all2all_receive buffer error!!!\n");
		tmp_all2all_send = (void *) malloc ((ull)mssg_size * total_send_size * ALL2ALL_SEND_EXPAND);
		CHECK_PTR_RETURN (tmp_all2all_send, "malloc tmp_all2all_send buffer error!!!\n");
		tmp_receive_offsets = (voff_t *) malloc ((np_node+1) * sizeof(voff_t) * world_size);
		CHECK_PTR_RETURN (tmp_receive_offsets, "malloc temp_receive_offsets error!!!\n");
		init_all2all = 1;
		tmp_all2all_receive_size = (ull)mssg_size * total_send_size * world_size * ALL2ALL_SEND_EXPAND;
		tmp_all2all_send_size = (ull)mssg_size * total_send_size * ALL2ALL_SEND_EXPAND;
	}

	// ******** copy partitions to be sent to other compute nodes
	voff_t cpy_offset = 0;
	for (i=0; i<world_size; i++)
	{
		if (i==world_rank)
			continue;
		int start_partition_id = np_per_node * i;
		int end_partition_id;
		if (i==world_size-1)
			end_partition_id = total_num_of_partitions;
		else
			end_partition_id = np_per_node * (i+1);
		for (j=start_partition_id; j<end_partition_id; j++)
		{
			int p;
			for (p = 0; p < num_of_procs; p++)
			{
				int intra_processor_partitions = mst->num_partitions[p+1] - mst->num_partitions[p];
				int index = mst->id2index[p][j];
				void * buf = mst->receive[p] + (ull)(mst->roff[p][index] - mst->roff[p][intra_processor_partitions]) * mssg_size; // start position of send buffer
				// be carefull here !!!!!!!!!!!!!!!
				voff_t size = mst->roff[p][index+1] - mst->roff[p][index];
//				voff_t size = mst->roff[p][index+1] - mst->roff[p][index];
				memcpy (tmp_all2all_send + (ull) cpy_offset * mssg_size, buf, (ull) mssg_size * size);
				cpy_offset += size;
				if (cpy_offset >= tmp_all2all_send_size)
				{
					printf ("EEEEEEEEEEEError:: malloced all2all send size %lu out of space: offset=%lu\n", tmp_all2all_send_size, cpy_offset);
					exit(0);
				}
//				printf ("WORLD RANK %d ::::::::::: copying all2all send:::::::::: %u\n", world_rank, cpy_offset);
			}

		}
	}
	return tmp_all2all_receive;
}

// this function returns a received offsets records for receive buffer, used for single machine of multiple processors, and multiple machines
// receive is actuall tmp_all2all_receive in this file
void * mpi_all2allv (void * send, void * receive, int total_num_of_partitions, int mssg_size, int world_size, int world_rank)
{
	int np_per_node;
	int np_node;
	get_np_node (&np_per_node, &np_node, total_num_of_partitions, world_size, world_rank);
	memset (tmp_receive_offsets, 0, (np_node+1) * sizeof(voff_t) * world_size);

	if (world_size == 1)
		return tmp_receive_offsets;

	MPI_Datatype mpi_type;
	int length = mssg_size / sizeof(voff_t);

	gettimeofday (&start, NULL);
	MPI_Type_contiguous(length, MPI_INT, &mpi_type);
	MPI_Type_commit(&mpi_type);

	if (rdispls[total_num_of_partitions-1] + rcounts[total_num_of_partitions-1] >= tmp_all2all_receive_size)
	{
		printf ("EEEEEEEEError:: malloced all2all receive buffer out of space: %lu - %lu!\n", \
				tmp_all2all_receive_size, rdispls[total_num_of_partitions-1] + rcounts[total_num_of_partitions-1]);
		exit(-1);
	}
	MPI_Alltoallv (tmp_all2all_send, scounts, sdispls, mpi_type, receive, rcounts, rdispls, mpi_type, MPI_COMM_WORLD);

	MPI_Type_free(&mpi_type);
	gettimeofday (&end, NULL);
	comm_time += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;

	memset (sdispls, 0, sizeof(int) * total_num_of_partitions);
	memset (rdispls, 0, sizeof(int) * total_num_of_partitions);
	memset (scounts, 0, sizeof(int) * total_num_of_partitions);
	memset (rcounts, 0, sizeof(int) * total_num_of_partitions);

	int i;
	for (i=0; i<world_size-1; i++)
	{
		scounts[i] = np_per_node;
		rcounts[i] = np_node;
		sdispls[i+1] = sdispls[i] + scounts[i];
		rdispls[i+1] = rdispls[i] + rcounts[i];
	}
	scounts[i] = total_num_of_partitions - (world_size-1) * np_per_node;
	rcounts[i] = np_node;

	gettimeofday (&start, NULL);
	MPI_Alltoallv (tmp_offset_counts, scounts, sdispls, MPI_INT, tmp_receive_offsets + 1, rcounts, rdispls, MPI_INT, MPI_COMM_WORLD);
	gettimeofday (&end, NULL);
	comm_time += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;

	return tmp_receive_offsets;
}

void finalize_receive_all2all (int world_size)
{
	if (init_all2all == 1)
	{
		free (tmp_all2all_receive);
		free (tmp_all2all_send);
		free (tmp_receive_offsets);
	}
	init_all2all = 0;
	tmp_all2all_send_size = 0;
	tmp_all2all_receive_size = 0;
}

void * mpi_allgatherv_inplace (void * send, void * receive, int total_num_of_partitions, int world_size, int world_rank)
{
	if (world_size == 1)
		return NULL;

	void * tmp_buf = (int *) malloc (sizeof(int) * total_num_of_partitions);
	CHECK_PTR_RETURN (tmp_buf, "malloc tmp buffer for mpi allgatherv inplace error!!!\n");

	memset (rdispls, 0, sizeof(int) * total_num_of_partitions);
	memset (rcounts, 0, sizeof(int) * total_num_of_partitions);
	int np_per_node;
	int np_node; // this is the real number of partitions in this compute node
	get_np_node (&np_per_node, &np_node, total_num_of_partitions, world_size, world_rank);
	int i;
	for (i=0; i<world_size-1; i++)
	{
		rcounts[i] = np_per_node;
		rdispls[i+1] = rdispls[i] + rcounts[i];
	}
	rcounts[i] = total_num_of_partitions - (world_size - 1) * np_per_node;
	MPI_Allgatherv(send, np_node, MPI_INT, tmp_buf, rcounts, rdispls, MPI_INT, MPI_COMM_WORLD);

	memcpy(receive, tmp_buf, sizeof(int) * total_num_of_partitions);
	free (tmp_buf);
	return receive;
}

void * mpi_allgatherv (void * send, void * receive, int total_num_of_partitions, int world_size, int world_rank)
{
	if (world_size == 1)
		return NULL;

	memset (rdispls, 0, sizeof(int) * total_num_of_partitions);
	memset (rcounts, 0, sizeof(int) * total_num_of_partitions);
	int np_per_node;
	int np_node; // this is the real number of partitions in this compute node
	get_np_node (&np_per_node, &np_node, total_num_of_partitions, world_size, world_rank);
	int i;
	for (i=0; i<world_size-1; i++)
	{
		rcounts[i] = np_per_node;
		rdispls[i+1] = rdispls[i] + rcounts[i];
	}
	rcounts[i] = total_num_of_partitions - (world_size - 1) * np_per_node;
	MPI_Allgatherv(send, np_node, MPI_INT, receive, rcounts, rdispls, MPI_INT, MPI_COMM_WORLD);

	return receive;
}

void distribute_partitions (int total_num_of_partitions, master_t * mst, subgraph_t * subgraph, ptype partitioning, int world_size, int world_rank, ull graph_size, double mssg_size)
{
	int np_per_node;
	int np_node; // this is the real number of partitions in this compute node
	get_np_node (&np_per_node, &np_node, total_num_of_partitions, world_size, world_rank);

	printf ("NNNNNNNNNNNNNNN number of partitions to be distributed among processors: %d\n", np_node);
	int i;
	int j = 0;
	bool break_flag = false;
	int num_of_devices = mst->num_of_devices;
	int num_of_cpus = mst->num_of_cpus;
	int num_of_procs = num_of_devices + num_of_cpus;
	int plist[NUM_OF_PROCS][NUM_OF_PARTITIONS];
	size_t max_device_size[NUM_OF_DEVICES];
	size_t max_device_graph_size[NUM_OF_DEVICES];
	double device_graph_size[NUM_OF_DEVICES];

	for (i=0; i<num_of_devices; i++)
	{
		max_device_size[i] = get_device_memory_size (i);
		max_device_graph_size[i] = (size_t)(floor((double)(max_device_size[i])/mssg_size));
		if (num_of_cpus > 0)
			device_graph_size[i] = 2*((double)graph_size/(2*num_of_devices+num_of_cpus)) < (double)max_device_graph_size[i] ? \
							2*((double)graph_size/(2*num_of_devices+num_of_cpus)) : (double)max_device_graph_size[i];
		else
			device_graph_size[i] = graph_size;
		printf ("total number of vertices: %lu, "
				"maximum number of vertices allowed with %lu memory on GPU: %lu, "
				"device_graph_size limit = %f\n", graph_size, max_device_size[i], max_device_graph_size[i], device_graph_size[i]);
	}

	if (partitioning == uneven)
	{
//		graph_size = graph_size * (1/2.5/(1/2.5*4+1/7.87));
//		graph_size = graph_size * (9.0/(9.0*4+4.5));
//		graph_size = graph_size * (9.0/(9.0*4+4.5));
//		ull device_graph_size = 1.5*((float)graph_size/(1.5*num_of_devices+num_of_cpus)) < max_device_graph_size ? \
				1.5*((float)graph_size/(1.5*num_of_devices+num_of_cpus)) : max_device_graph_size;
//		ull device_graph_size = graph_size * (9.0/(9.0*4+4.5)) < max_device_graph_size ? graph_size * (9.0/(9.0*4+4.5)): max_device_graph_size;

		printf ("***UNEVEN*** DISTRIBUTION ::::::::::::::::::::::\n");
		while (j < np_node)
		{
			if (num_of_devices == 0)
				break;
			for (i = 0; i < num_of_devices; i++)
			{
				if (j == np_node - 1 && num_of_cpus != 0)
				{
					break_flag = true;
					break;
				}
				if (mst->num_vs[i] + (subgraph->subgraphs)[j].size > device_graph_size[i] || j >= np_node)
				{
					break_flag = true;
					break;
				}
				mst->num_vs[i] += (subgraph->subgraphs)[j].size;
				plist[i][mst->num_partitions[i+1]++] = (subgraph->subgraphs)[j].id;
				j++;
			}
			if (break_flag)
				break;
		}
		break_flag = false;
		while (j < np_node)
		{
			if (num_of_cpus == 0)
			{
				printf ("Resource not sufficient for all the partitions!!! Please set \"-c 1\" to use CPU to help running the program!!!\n");
				int tt;
				for (tt=0; tt<np_node; tt++)
				{
					printf ("%d: subgraph size %u\n", tt, subgraph->subgraphs[tt].size);
				}
				exit (0);
			}
			for (i = 0; i < num_of_cpus; i++)
			{
				if (j >= np_node)
				{
					break_flag=true;
					break;
				}
				mst->num_vs[num_of_devices + i] += (subgraph->subgraphs)[j].size;
				plist[num_of_devices + i][mst->num_partitions[num_of_devices + i + 1]++] = (subgraph->subgraphs)[j].id;
				j++;
			}
			if (break_flag)
				break;
		}
	}
	else
	{
		printf ("***EVEN*** DISTRIBUTION::::::::::::::::::::::\n");
		while (j < np_node)
		{
			if (num_of_procs == 0)
			{
				printf ("Resource not sufficient for all the partitions!!!\n");
				exit (0);
			}
			for (i = 0; i < num_of_devices; i++)
			{
				if (j >= np_node)
				{
					break_flag=true;
					break;
				}
				if (mst->num_vs[i] + (subgraph->subgraphs)[j].size > max_device_graph_size[i])
				{
					continue;
				}
				mst->num_vs[i] += (subgraph->subgraphs)[j].size;
				plist[i][mst->num_partitions[i+1]++] = (subgraph->subgraphs)[j].id;
				j++;
			}
			for (; i < num_of_procs; i++)
			{
				if (j >= np_node)
				{
					break_flag=true;
					break;
				}
				mst->num_vs[i] += (subgraph->subgraphs)[j].size;
				plist[i][mst->num_partitions[i+1]++] = (subgraph->subgraphs)[j].id;
				j++;
			}
			if (break_flag)
			{
				break;
			}
		}
	}
	if (j<np_node)
	{
		printf ("Memory not sufficient for a total number of vertices %lu\n", graph_size);
		exit(0);
	}
	//test:::::::::::::::
	int c=0;
	int g=0;
	for (i=0; i<num_of_devices; i++)
	{
		printf ("num_vs %d: %lu\n", i, mst->num_vs[i]);
		if (mst->num_vs[i] > 0)
			g++;
	}
	for (i=0; i<num_of_cpus; i++)
	{
		printf ("num_vs %d: %lu\n", num_of_devices+i, mst->num_vs[num_of_devices+i]);
		if (mst->num_vs[num_of_devices+i] > 0)
			c++;
	}
	mst->num_of_devices = g;
	mst->num_of_cpus = c;
	num_of_devices = mst->num_of_devices;
	num_of_cpus = mst->num_of_cpus;
	num_of_procs = num_of_devices + num_of_cpus;
	// allocate memory for partition list transferred from each processor
	for (i = 0; i < num_of_procs; i++)
	{
		mst->not_reside[i] = (int *) malloc (sizeof(int) * (total_num_of_partitions - mst->num_partitions[i + 1]));
		mst->id2index[i] = (int *) malloc (sizeof(int) * total_num_of_partitions);
		mst->index2id[i] = (int *) malloc (sizeof(int) * (total_num_of_partitions));
		mst->pfrom[i] = (int *) malloc (sizeof(int) * (mst->num_partitions[i+1] * (num_of_procs-1)));
		CHECK_PTR_RETURN (mst->pfrom[i], "malloc partition_from array in master for device %d error!\n", i);
		memset(mst->pfrom[i], 0, sizeof(int) * (mst->num_partitions[i+1] * (num_of_procs-1)));
	}

	for (i = 0; i < num_of_procs; i++)
		mst->num_partitions[i + 1] += mst->num_partitions[i]; // ????????????? check num_partitions here!!!!!!!!!!

	// ************ partition id to send buffer id *************
	voff_t offset;
	int mspid;
	for (i = 0; i < num_of_procs; i++)
	{
		printf ("Partition in Proc %d: \n", i);
		offset = mst->num_partitions[i];
		for (j = 0; j < mst->num_partitions[i + 1] - mst->num_partitions[i]; j++)
		{
			mspid = plist[i][j];
			mst->partition_list[offset + j] = mspid;
//			printf ("%d\t", mspid);
			mst->r2s[mspid] = i;
		}
//		printf ("\n");
	}

	//debug: print partition list in the compute node:
	printf ("Partition list in process %d: \n", world_rank);
	for (i=0; i<np_node; i++)
	{
		printf ("%d\t", mst->partition_list[i]);
	}
	printf ("\n");
	// gather entire partition list of all compute nodes in tmp_partition_list, return tmp_partition_list
	mpi_allgatherv(mst->partition_list, tmp_partition_list, total_num_of_partitions, world_size, world_rank);
	// ********* generate id2index array for each processor to identify message buffer offset of partitions
	int pid;
	int p;
	int poffset;
	int ptr=0;
	int start_partition_id = np_per_node * world_rank;
	int end_partition_id = np_per_node * world_rank + np_node;
	// *********** Assigning id2index for partitions from other compute node *************
	if (world_size > 1)
	{
	for (i=0; i<total_num_of_partitions; i++)
	{
		pid = tmp_partition_list[i]; // the globally gathered partition_list
		if(pid >= start_partition_id && pid < end_partition_id)
		{
			continue;
		}
		else
		{
//			printf("i=%d ptr=%d  !!!!!!\n", i, ptr);
			for(p=0; p<num_of_procs; p++)
			{
				mst->index2id[p][np_node+ptr] = pid;
				mst->id2index[p][pid] = np_node + ptr;
			}
		}
		ptr++;
	}
	}
	// ************ Assigning id2index for partitions in this compute node ************
	for (i = 0; i < num_of_procs; i++)
	{
		for (p = 0; p < num_of_procs; p++)
		{
			poffset = mst->num_partitions[i + 1] - mst->num_partitions[i];
			offset = mst->num_partitions[p];
			for (j = 0; j < mst->num_partitions[p + 1] - mst->num_partitions[p]; j++)
			{
				pid = mst->partition_list[offset + j];
				if (p < i)
				{
					mst->id2index[i][pid] = poffset + mst->num_partitions[p] + j;
					mst->index2id[i][poffset + mst->num_partitions[p] + j] = pid;
				}
				else if (p == i)
				{
					mst->id2index[i][pid] = j;
					mst->index2id[i][j] = pid;
				}
				else
				{
					mst->id2index[i][pid] = mst->num_partitions[p] + j;
					mst->index2id[i][mst->num_partitions[p] + j] = pid;
				}
			}
		}
	}

	// ************** generate not_reside partition list for each processor ******************
	for (i = 0; i < num_of_procs; i++)
	{
		int start = mst->num_partitions[i];
		int end = mst->num_partitions[i + 1];
		int index = 0;
		for (j = 0; j < np_node; j++)
		{
			if (j >= start && j < end)
			{
				int pid=mst->partition_list[j];
				int ij;
				int pr=0;
				for (ij=0; ij<num_of_procs; ij++)
				{
					if (ij==i) continue;
					int nps = mst->num_partitions[ij+1] - mst->num_partitions[ij];
					mst->pfrom[i][(j-start)*(num_of_procs-1) + pr++] = mst->id2index[ij][pid] - nps;
				}
				continue;
			}
			mst->not_reside[i][index++] = mst->partition_list[j];
		}
	}


//	printf ("buffer index array: \n");
	for (i = 0; i < num_of_procs; i++)
	{
//		printf ("PPPPPPPPPP print id2index array for processor %d:\n", i);
//		print_offsets (mst->id2index[i], total_num_of_partitions);
	}
}
