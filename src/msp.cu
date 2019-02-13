/*
 * get_data.cu
 *
 *  Created on: 2015-3-5
 *      Author: qiushuang
 */

#include <omp.h>
#include <pthread.h>
#include "../include/dbgraph.h"
#include "../include/io.h"
#include "../include/hash.h"
#include "../include/msp.h"
#include "bitkmer.cuh"
#include "hash.cuh"
#include "dbgraph.cuh"

extern long cpu_threads;
extern int prd;
extern int rd;
extern int rdy;
extern int wrt;
extern ull * read_size;
extern int * dflag;
extern int * queue;
extern ull * max_msp_malloc;
static uint num_reads_gpu;
extern double read_ratio;
extern int mpi_run;
extern float inmemory_time;

extern __shared__ seq_t shared_reads[]; // the size should be: THREADS_PER_BLOCK * (read_length + 3) / 4 + RID_SIZE

#define read_ptr (shared_reads + index) // pointer to read of each thread in shared memory


__device__ int get_first_kmer_stream (kmer_t * kmer, seq_t * read, int len)
{
	uint * ptr = (uint *) kmer;
	int i;

	if (len < KMER_CHAR_LENGTH)
	{
		for (i = 0; i < (2 * len + 31) / 32 - 1; i++)
		{
			*ptr |= (uint) (*read++);
			*ptr <<= SEQ_BIT_LENGTH;
			*ptr |= (uint) (*read++);
			*ptr <<= SEQ_BIT_LENGTH;
			*ptr |= (uint) (*read++);
			*ptr <<= SEQ_BIT_LENGTH;
			*ptr |= (uint) (*read++);
			ptr++;
		}
		for (i = 0; i < ((2 * len) % 32 + SEQ_BIT_LENGTH - 1) / SEQ_BIT_LENGTH; i++)
		{
			*ptr <<= SEQ_BIT_LENGTH;
			*ptr |= (uint) (*read++);
		}
		*ptr >>= (SEQ_BIT_LENGTH - (2 * len) % SEQ_BIT_LENGTH) % SEQ_BIT_LENGTH;
		*ptr <<= 32 - i * SEQ_BIT_LENGTH + (SEQ_BIT_LENGTH - (2 * len) % SEQ_BIT_LENGTH) % SEQ_BIT_LENGTH;
		return len;
	}
	for (i = 0; i < KMER_BIT_LENGTH / KMER_UNIT_BITS; i++)
	{
		*ptr |= (uint) (*read++);
		*ptr <<= SEQ_BIT_LENGTH;
		*ptr |= (uint) (*read++);
		*ptr <<= SEQ_BIT_LENGTH;
		*ptr |= (uint) (*read++);
		*ptr <<= SEQ_BIT_LENGTH;
		*ptr |= (uint) (*read++);
		ptr++;
	}
	return KMER_CHAR_LENGTH;
}

/* This function is to compute the number of minimum substrings of kmers of one read */
__global__ void compute_num_of_ms (seq_t * reads, uint dsize, uint shared_size, int k, int p, int read_length, int num_of_partitions, uint unit_size, uint num_of_reads, uch * d_msp_ptr)
{
	msp_t d_msp;
	d_msp.nums = d_msp_ptr;
	d_msp.poses = d_msp_ptr + sizeof(uch) * num_of_reads;
	uint max_num_ms = read_length - k + 1;
	d_msp.ids = (msp_id_t *)(d_msp_ptr + sizeof(uch) * (max_num_ms + 1) * num_of_reads + (sizeof(msp_id_t) - sizeof(uch) * (max_num_ms + 1) * num_of_reads % sizeof(msp_id_t)));

	const uint gid = blockIdx.x * blockDim.x + threadIdx.x; // 1D block and 1D thread, global id
	const uint tid = threadIdx.x; // thread id within a block

	int r, w;
	w = (num_of_reads + TOTAL_THREADS - 1) / TOTAL_THREADS;

	for (r = 0; r < w; r++)
	{

	/* load data from device to shared memory */

	uint load_size = MIN(THREADS_PER_BLOCK * unit_size, dsize - THREADS_PER_BLOCK * unit_size * r);
	uint size_per_load = THREADS_PER_BLOCK * LOAD_WIDTH_PER_THREAD;
	seq_t * rptr = reads + TOTAL_THREADS * unit_size * r + blockIdx.x * THREADS_PER_BLOCK * unit_size;
	if (rptr - reads > dsize)
		return;
	uint bound = MIN(dsize + LOAD_WIDTH_PER_THREAD - TOTAL_THREADS * unit_size * r - blockIdx.x * THREADS_PER_BLOCK * unit_size, shared_size);

	int t;
	for (t = 0; t < (load_size + size_per_load -1) / size_per_load; t++)
	{
		if ( (tid + 1) * LOAD_WIDTH_PER_THREAD + t * size_per_load < bound)
		{
			/*
			shared_reads[tid * LOAD_WIDTH_PER_THREAD + t * size_per_load] = rptr[tid * LOAD_WIDTH_PER_THREAD + t * size_per_load];
			shared_reads[tid * LOAD_WIDTH_PER_THREAD + 1 + t * size_per_load] = rptr[tid * LOAD_WIDTH_PER_THREAD + 1 + t * size_per_load];
			shared_reads[tid * LOAD_WIDTH_PER_THREAD + 2 + t * size_per_load] = rptr[tid * LOAD_WIDTH_PER_THREAD + 2 + t * size_per_load];
			shared_reads[tid * LOAD_WIDTH_PER_THREAD + 3 + t * size_per_load] = rptr[tid * LOAD_WIDTH_PER_THREAD + 3 + t * size_per_load];
			*/
			*((uint *) (shared_reads + tid * LOAD_WIDTH_PER_THREAD + t * size_per_load)) = *((uint *) (rptr + tid * LOAD_WIDTH_PER_THREAD + t * size_per_load));
		}
	}

	__syncthreads();

	if (gid + r * TOTAL_THREADS >= num_of_reads)
		return;

	int i; // point to the start position of current on read
	int j; // point to the start position of current p-substring on kmer
	const int index = tid * unit_size; // index of start position of the read

	uch num = 0;
	msp_id_t old_id, new_id;
	minstr_t minpstr = 0, rminpstr = 0;
	minstr_t curr = 0;
	minstr_t pstr = 0, rpstr = 0;
#ifdef LONG_KMER
	kmer_t kmer = {0, 0, 0, 0};
	kmer_t reverse = {0, 0, 0, 0};
#else
	kmer_t kmer = {0, 0};
	kmer_t reverse = {0, 0};
#endif

	int table_index;

	/* for debug only */
//	printf ("gid: %u, start of string: %o\n", gid, shared_reads[index]);

	/* Get first kmer and compute p minimum substring for this first kmer */
	get_first_kmer_stream (&kmer, read_ptr, KMER_CHAR_LENGTH);

	get_reverse (&kmer, &reverse);
#ifdef LONG_KMER
	table_index = (128 - k * 2) / 32;
	shift_dictionary[table_index] (&reverse, 128 - k * 2);
#else
	table_index = (64 - k * 2) / 32;
	shift_dictionary[table_index] (&reverse, 64 - k * 2);
#endif

	/* get first minimum p-substring */
	get_first_pstr ((unit_kmer_t *)&kmer, &pstr, p);
	get_first_pstr ((unit_kmer_t *)&reverse, &rpstr, p);
	minpstr = pstr;
	rminpstr = rpstr;
	for (j = 1; j < k - p + 1; j++)
	{
		right_shift_pstr ((unit_kmer_t *)&kmer, &pstr, p, j);
		right_shift_pstr ((unit_kmer_t *)&reverse, &rpstr, p, j);
		if (pstr < minpstr) minpstr = pstr;
		if (rpstr < rminpstr) rminpstr = rpstr;
	}
	curr = rminpstr < minpstr ? rminpstr : minpstr;
	d_msp.ids[(TOTAL_THREADS * r + gid) * max_num_ms + num] = get_partition_id (curr, p, num_of_partitions);
	old_id = d_msp.ids[(TOTAL_THREADS * r + gid) * max_num_ms + num];
	if (d_msp.ids[(TOTAL_THREADS * r + gid) * max_num_ms + num] >= num_of_partitions)
	{
		printf ("error in  partition id!\n");
	}
	d_msp.poses[(TOTAL_THREADS * r + gid) * max_num_ms + num] = 0;

	/* minimum p substrings for the following kmers: */
	for (i = 1; i < read_length - k + 1; i++)
	{
		kmer_32bit_left_shift (&kmer, 2);
		get_reverse (&kmer, &reverse);
#ifdef LONG_KMER
		table_index = (128 - k * 2) / 32;
		shift_dictionary[table_index] (&reverse, 128 - k * 2);
#else
		table_index = (64 - k * 2) / 32;
		shift_dictionary[table_index] (&reverse, 64 - k * 2);
#endif
		get_first_pstr ((unit_kmer_t *)&kmer, &pstr, p);
		get_first_pstr ((unit_kmer_t *)&reverse, &rpstr, p);
		minpstr = pstr;
		rminpstr = rpstr;
		for (j = 1; j < k - p + 1; j++)
		{
			right_shift_pstr ((unit_kmer_t *)&kmer, &pstr, p, j);
			right_shift_pstr ((unit_kmer_t *)&reverse, &rpstr, p, j);
			if (pstr < minpstr) minpstr = pstr;
			if (rpstr < rminpstr) rminpstr = rpstr;
		}
		curr = rminpstr < minpstr ? rminpstr : minpstr;
		new_id = get_partition_id (curr, p, num_of_partitions);

		if (new_id != old_id)
		{
			num++;
			d_msp.ids[(TOTAL_THREADS * r + gid) * max_num_ms + num] = new_id;
			if (d_msp.ids[(TOTAL_THREADS * r + gid) * max_num_ms + num] >= num_of_partitions)
			{
				printf ("error in  partition id!\n");
			}
			d_msp.poses[(TOTAL_THREADS * r + gid) * max_num_ms + num] = i; // test this
			old_id = new_id;
		}
		if ((i % 4) == 0 && ((KMER_CHAR_LENGTH + i) / 4 <= unit_size))
		{
			//get_next_byte_kmer (read_ptr + i / 4, &kmer);
#ifdef LONG_KMER
			kmer.w |= (unit_kmer_t) (*(read_ptr + KMER_CHAR_LENGTH / 4 + i / 4 - 1));
#else
			kmer.y |= (unit_kmer_t) (*(read_ptr + KMER_CHAR_LENGTH / 4 + i / 4 - 1));
#endif
		}
	}

	d_msp.nums[TOTAL_THREADS * r + gid] = num;

	/* for debug only: */
#ifdef DEBUG
	if (num >= max_num_ms)
	{
		printf ("Careful: number of superkmers exceeds predefined size!\n");
		return;
	}
#endif

	__syncthreads();
	}
}

extern "C"
{
static void gpu_msp_workflow (char * read_buffer[], uch * h_msp_ptr[], read_buf_t * reads, uch * d_msp_ptr, seq_t * d_reads, char ** rbufs[], uint * rnums[], int p, int k, int read_length, int num_of_partitions, int nstreams, int did)
{
	cudaSetDevice (did);

	evaltime_t start, end;
	evaltime_t gpus, gpue;
	uint num_of_reads = 0;
	uint total_num_of_reads = 0;
	float gpu_qtime = 0;
	float parse_time = 0;
	float gpu_comtime = 0;

	// device variables
	ull malloc_size;
	ull align_size;
	uint unit_size = (read_length + 3) / 4;
	uint shared_size = THREADS_PER_BLOCK * (unit_size + 1);
	uint size;
	uint scan[cpu_threads];
	uint num_of_blocks;
	uint block_size;

	gettimeofday (&gpus, NULL);
	while (rd < nstreams)
	{
	gettimeofday (&start, NULL);
	int q = atomic_increase (&rd, 1);
	while (q > prd) {}
//	printf ("@@@@@@@@@@@@@@@@@@ queue id: %d, prd = %d, rd = %d @@@@@@@@@@@@@@@@@@\n", q, prd, rd);
	gettimeofday (&end, NULL);
	gpu_qtime += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	while (q - wrt > 5) {} // wait if producing is much faster than writing
	h_msp_ptr[q] = (uch *) malloc (sizeof(uch) * max_msp_malloc[q]);
	CHECK_PTR_RETURN (h_msp_ptr[q], "malloc msp ptr %d error!!!!!!!!!!!\n", q);

	gettimeofday (&start, NULL);
	num_of_reads = parse_data (read_buffer, reads, read_size[q], q, rbufs, rnums);
	if (num_of_reads > num_reads_gpu)
	{
		printf ("Error in initiation of GPU! number of reads set: %u, current number of reads: %u\n", num_reads_gpu, num_of_reads);
		exit(0);
	}
	total_num_of_reads += num_of_reads;
	gettimeofday (&end, NULL);
	parse_time += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
//	num_of_reads += compute_msp (q, p, k, read_buffer, read_size[q], d_msp_ptr[q], rbufs, rnums);

	gettimeofday (&start, NULL);
	scan[0] = 0;
	int l;
	for (l = 1; l < cpu_threads; l++)
	{
		scan[l] = scan[l - 1] + reads->offset[l - 1];
	}
	size = scan[cpu_threads - 1] + reads->offset[cpu_threads - 1];
	size += LOAD_WIDTH_PER_THREAD - size % LOAD_WIDTH_PER_THREAD; // align size for device malloc

	num_of_blocks = (num_of_reads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	block_size = num_of_blocks > MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : num_of_blocks;

	ull max_num_ms = read_length - k + 1;
	align_size = sizeof(uch) * (max_num_ms + 1) * num_of_reads + (sizeof(msp_id_t) - sizeof(uch) * (max_num_ms + 1) * num_of_reads % sizeof(msp_id_t));
	malloc_size = align_size + sizeof(msp_id_t) * max_num_ms * num_of_reads;
	if (malloc_size > max_msp_malloc[q])
	{
		printf ("Exception: MSP required a buffer size larger than maximum msp malloc size - %lu\n", malloc_size);
		exit (1);
	}

	int thid;
	for (thid = 0; thid < cpu_threads; thid++)
	{
		seq_t * s_ptr = reads->buf + (CODE_BUF_SIZE / cpu_threads * thid);
		seq_t * d_ptr = d_reads + scan[thid];

//		CUDA_CHECK_RETURN (cudaMemcpyAsync(d_ptr, s_ptr, (reads[turn].offset)[thid], cudaMemcpyHostToDevice, streams[turn]));
		CUDA_CHECK_RETURN (cudaMemcpy(d_ptr, s_ptr, (reads->offset)[thid], cudaMemcpyHostToDevice));
	}
	compute_num_of_ms <<<block_size, THREADS_PER_BLOCK, shared_size>>> (d_reads, size, shared_size, k, p, read_length, num_of_partitions, unit_size, num_of_reads, d_msp_ptr);

	CUDA_CHECK_RETURN (cudaMemcpy(h_msp_ptr[q], d_msp_ptr, malloc_size, cudaMemcpyDeviceToHost));
	gettimeofday (&end, NULL);
	gpu_comtime += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;

	dflag[q] = 1;
	int wrt_id = atomic_increase (&rdy, 1);
	queue[wrt_id + 1] = q;
	}
	gettimeofday (&gpue, NULL);
	print_exec_time (gpus, gpue, "~~~~~~~~~~~~~~~~~overall msp compute time on GPU %d: \n", did);
	printf ("~~~~~~~~~~~~~~number of reads on GPU %d: %d\n", did, total_num_of_reads);
	printf ("~~~~~~~~~~~~~~GPU %d MSP parsing data time: %f\n", did, parse_time);
	printf ("~~~~~~~~~~~~~~GPU %d MSP computing time: %f\n", did, gpu_comtime);
	printf ("~~~~~~~~~~~~~~GPU %d MSP queuing time %f\n", gpu_qtime);

}

void *
gpu_partition_workflow (void * arg)
{
	gpu_msp_arg * garg = (gpu_msp_arg *) arg;
	gpu_msp_workflow (garg->read_buffer, garg->h_msp_ptr, garg->reads, garg->d_msp_ptr, garg->d_reads, garg->rbufs, garg->rnums, garg->p, garg->k, garg->read_length, garg->num_of_partitions, garg->nstreams, garg->did);
	return ((void *)0);
}

void init_gpu_msp (read_buf_t * reads, seq_t * d_reads[], uch * d_msp_ptr[], int num_of_msp_devices, int read_length, int k, uint num_reads)
{
	ull max_num_ms = read_length - k + 1;
	ull max_msp_malloc = (max_num_ms * sizeof(msp_id_t) + sizeof(uch) * max_num_ms + 1) * ((ull)num_reads)\
			+ (sizeof(msp_id_t) - sizeof(uch) * (max_num_ms + 1) * num_reads % sizeof(msp_id_t));
	printf ("number of reads used to initialize GPU: %u, max_msp_malloc for GPU %lu\n", num_reads, max_msp_malloc);
	num_reads_gpu = num_reads;
	int i;
	for (i = 0; i < num_of_msp_devices; i++)
	{
		cudaSetDevice (i);
		CUDA_CHECK_RETURN (cudaMallocHost (&reads[i].buf, sizeof(seq_t) * CODE_BUF_SIZE)); // code buffer
		CUDA_CHECK_RETURN (cudaMalloc(&d_reads[i], sizeof(seq_t) * CODE_BUF_SIZE));
		CUDA_CHECK_RETURN (cudaMalloc(&d_msp_ptr[i], max_msp_malloc));
	}
}

void finalize_gpu_msp (read_buf_t * reads, seq_t * d_reads[], uch * d_msp_ptr[], int num_of_msp_devices)
{
	int i;
	for (i = 0; i < num_of_msp_devices; i++)
	{
		cudaFreeHost (reads[i].buf);
		cudaFree (d_reads[i]);
		cudaFree (d_msp_ptr[i]);
	}
}

void msp_partition (char * input_file, char * file_dir, int k, int p, int read_length, int num_of_partitions, int num_of_msp_devices, int num_of_msp_cpus, int world_size, int world_rank)
{
	evaltime_t start, end;
	evaltime_t overs, overe;
	cpu_msp_arg cpu_arg[NUM_OF_MSP_CPUS];
	output_msp_arg output_arg;
	gpu_msp_arg gpu_arg[NUM_OF_MSP_DEVICES];
	pthread_t cpu_thread[NUM_OF_MSP_CPUS];
	pthread_t gpu_thread[NUM_OF_MSP_DEVICES];
	pthread_t output_thread;

	/* initialize input */
	gettimeofday (&start, NULL);
	int nstreams;
	if (mpi_run >= 0)
		nstreams = init_input (input_file, read_length, world_size, world_rank);
	else
		nstreams = init_mpi_input (input_file, read_length, world_size, world_rank);
	init_lookup_table();
	estimate_num_reads_from_input (input_file, read_length);
	printf ("read ratio estimated from input file size: %f\n", read_ratio);
	init_msp_output (file_dir, num_of_partitions, world_rank);
	init_msp_meta (num_of_partitions, read_ratio, read_length, k);
	set_length_range (num_of_partitions, read_length);

	offset_t * max_kmers = (offset_t *) malloc (sizeof(offset_t) * world_size);
	offset_t * max_spks = (offset_t *) malloc (sizeof(offset_t) * world_size);
	offset_t * max_spksizes = (offset_t *) malloc (sizeof(offset_t) * world_size);
	CHECK_PTR_RETURN (max_kmers, "malloc maximum kmer array of comm world error!\n");
	CHECK_PTR_RETURN (max_spks, "malloc maximum spk array of comm world error!\n");
	CHECK_PTR_RETURN (max_spksizes, "malloc maximum spksize array of comm world error!\n");
	memset (max_kmers, 0, sizeof(offset_t) * world_size);
	memset (max_spks, 0, sizeof(offset_t) * world_size);
	memset (max_spksizes, 0, sizeof(offset_t) * world_size);

	float readfile_time = 0;
	char *** rbufs;
	uint ** rnums;

	char ** read_buffer = (char **) malloc (sizeof(char *) * nstreams);
	uch ** d_msp_ptr = (uch **) malloc (sizeof(uch *) * nstreams);
	read_size = (ull *) malloc (sizeof(ull) * nstreams);
	rbufs = (char ***) malloc (sizeof(char **) * nstreams);
	rnums = (uint **) malloc (sizeof(uint **) * nstreams);
	dflag = (int *) malloc (sizeof(int) * nstreams);
	queue = (int *) malloc (sizeof(int) * nstreams);
	max_msp_malloc = (ull *) malloc (sizeof(ull) * nstreams);
	memset (max_msp_malloc, 0, sizeof(ull) * nstreams);
	CHECK_PTR_RETURN (read_buffer, "init read buffers error!!!!!!!!!!\n");
	CHECK_PTR_RETURN (d_msp_ptr, "init msp array buffer error!!!!!!!!!\n");
	CHECK_PTR_RETURN (rbufs, "init rbufs pointer error!!!!!!!!\n");
	CHECK_PTR_RETURN (rnums, "init rnums pointer error!!!!!!!\n");
	CHECK_PTR_RETURN (max_msp_malloc, "init max_msp_malloc array error!!!!\n");
	int i;
	int d;
	int c;

	for (i = 0; i < nstreams; i++)
	{
		rbufs[i] = (char **) malloc (sizeof(char *) * cpu_threads);
		rnums[i] = (uint *) malloc (sizeof(uint) * cpu_threads);
		CHECK_PTR_RETURN (rbufs[i], "int threads read buffers error!!!!!!!!!!!!\n");
		CHECK_PTR_RETURN (rnums[i], "init threads number of reads error!!!!!!!!!!!!!!\n");
	}

	output_arg.d_msp_ptr = d_msp_ptr;
	output_arg.rbufs = rbufs;
	output_arg.read_buffer = read_buffer;
	output_arg.rnums = rnums;
	output_arg.k = k;
	output_arg.p = p;
	output_arg.read_length = read_length;
	output_arg.num_of_partitions = num_of_partitions;
	output_arg.nstreams = nstreams;
	output_arg.world_size = world_size;
	output_arg.world_rank = world_rank;


	read_buf_t reads[NUM_OF_MSP_DEVICES];
	seq_t * d_reads[NUM_OF_MSP_DEVICES];
	uch * dd_msp_ptr[NUM_OF_MSP_DEVICES];


	gettimeofday (&end, NULL);
	print_exec_time (start, end, "Initiating msp input and output time: ");

	gettimeofday (&overs, NULL);
	for (i = 0; i < nstreams; i++)
	{
		while (prd - rd > WAIT) {} //waiting when serving is much faster than consuming
		gettimeofday (&start, NULL);
		read_buffer[i] = (char *) malloc (sizeof(char) * (BUF_SIZE + LINE));

		if (mpi_run >= 0)
			read_size[i] = read_file (read_buffer[i], world_rank);
		else
			read_size[i] = mpi_read_file (read_buffer[i], world_size, world_rank);
		uint max_num_ms = read_length - k + 1;
		uint num_reads = ceil(read_size[i] * read_ratio);
		uint num_reads_per_thread = (num_reads + cpu_threads - 1) / cpu_threads;
		ull align_size = sizeof(msp_id_t) - sizeof(uch) * (max_num_ms + 1) * num_reads_per_thread % sizeof(msp_id_t);
		ull usize = max_num_ms * sizeof(msp_id_t) + sizeof(uch) * max_num_ms + 1;
		max_msp_malloc[i] = (usize * num_reads_per_thread + align_size) * cpu_threads;
//		printf ("WORLD RANK %d: read buffer size in input read stream %d: %lu\n", world_rank, i, read_size[i]);
		gettimeofday (&end, NULL);
		readfile_time += (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
		atomic_increase (&prd, 1);
		if (i == 0)
		{
//			cpu_arg.read_size = read_size;
			//Call CPU MSP:
			for (c = 0; c < num_of_msp_cpus; c++)
			{
				cpu_arg[c].d_msp_ptr = d_msp_ptr;
				cpu_arg[c].k = k;
				cpu_arg[c].nstreams = nstreams;
				cpu_arg[c].p = p;
				cpu_arg[c].read_buffer = read_buffer;
				cpu_arg[c].rbufs = rbufs;
				cpu_arg[c].rnums = rnums;
				cpu_arg[c].read_length = read_length;
				cpu_arg[c].num_of_partitions = num_of_partitions;
				cpu_arg[c].world_size = world_size;
				cpu_arg[c].world_rank = world_rank;
				if (pthread_create (&cpu_thread[c], NULL, cpu_partition_workflow, &cpu_arg) != 0)
				{
					printf ("create thread for msp on CPU error!!!!\n");
				}
			}
			// Call GPU MSP:
			uint num_rd = ceil(read_size[0] * read_ratio);
			init_gpu_msp (reads, d_reads, dd_msp_ptr, num_of_msp_devices, read_length, k, num_rd);

			for (d = 0; d < num_of_msp_devices; d++)
			{
				gpu_arg[d].d_msp_ptr = dd_msp_ptr[d];
				gpu_arg[d].d_reads = d_reads[d];
				gpu_arg[d].did = d;
				gpu_arg[d].h_msp_ptr = d_msp_ptr;
				gpu_arg[d].k = k;
				gpu_arg[d].nstreams = nstreams;
				gpu_arg[d].p = p;
				gpu_arg[d].rbufs = rbufs;
				gpu_arg[d].read_buffer = read_buffer;
				gpu_arg[d].reads = &reads[d];
				gpu_arg[d].rnums = rnums;
				gpu_arg[d].read_length = read_length;
				gpu_arg[d].num_of_partitions = num_of_partitions;
				gpu_arg[d].world_size = world_size;
				gpu_arg[d].world_rank = world_rank;
				if (pthread_create (&gpu_thread[d], NULL, gpu_partition_workflow, &gpu_arg[d]) != 0)
				{
					printf ("create thread for msp on CPU error!!!!\n");
				}
			}
			//Call Output MSP:
			if (pthread_create (&output_thread, NULL, output_msp_workflow, &output_arg) != 0)
			{
				printf ("create thread for output superkmer partitions on CPU error!!!!!\n");
			}

		}
	}
	for (c = 0; c < num_of_msp_cpus; c++)
	{
		if (pthread_join (cpu_thread[c], NULL) != 0)
		{
				printf ("Join thread on CPU MSP failure!!!!!!\n");
		}
	}
	for (d = 0; d < num_of_msp_devices; d++)
	{
		if (pthread_join (gpu_thread[d], NULL) != 0)
		{
				printf ("Join thread on GPU MSP failure!!!!!!\n");
		}
	}
	if (pthread_join (output_thread, NULL) != 0)
	{
			printf ("Join thread on CPU MSP OUTPUT failure!!!!!!\n");
	}

/*	printf ("^^^^^^^^^^^^^^^^ queue ^^^^^^^^^^^^\n");
	for (i = 0; i < nstreams; i++)
	{
		printf ("%d\t", queue[i]);
	}
	printf ("\n");*/
	gettimeofday (&overe, NULL);
	inmemory_time += (float)((overe.tv_sec * 1000000 + overe.tv_usec) - (overs.tv_sec * 1000000 + overs.tv_usec)) / 1000;
	print_exec_time (overs, overe, "******************* Inner MSP time *******************\n");

	free (read_buffer);
	free (d_msp_ptr);
	free (read_size);
	free (dflag);
	free (queue);
	free (max_msp_malloc);
	for (i = 0; i < nstreams; i++)
	{
		free (rbufs[i]);
		free (rnums[i]);
	}
	free (rbufs);
	free (rnums);

	gettimeofday (&start, NULL);
	finalize_msp_output (num_of_partitions);
	finalize_msp_meta (k, num_of_partitions, max_kmers, max_spks, max_spksizes, world_size, world_rank);
	if (mpi_run >= 0)
		finalize_input ();
	else
		finalize_mpi_input ();
	finalize_gpu_msp (reads, d_reads, dd_msp_ptr, num_of_msp_devices);

	free (max_kmers);
	free (max_spks);
	free (max_spksizes);
//	printf ("total number of reads: %lu\n", total_num_reads);
	gettimeofday (&end, NULL);
	print_exec_time (start, end, "Finalizing msp input and output time: ");
	printf ("************msp reading file time: %f******************\n", readfile_time);
}

}
