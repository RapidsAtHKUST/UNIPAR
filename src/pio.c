/*
 * pio.c: parallel input/output and CPU msp compute
 *
 *  Created on: 2015-3-5
 *      Author: qiushuang
 */

#include <omp.h>
#include <mpi.h>
#include <math.h>
#include "../include/io.h"
#include "../include/dbgraph.h"
#include "../include/msp.h"
#include "../include/hash.h"
#include "../include/bitkmer.h"
#include "../include/share.h"

#define USING_MPI_IO

#define THREADS_MSP_OUTPUT THREADS_MSP_COMPUTE // parallel output threads should be the same number of parallel compute
#define THREADS_MSP_META THREADS_MSP_COMPUTE // parallel output threads should be the same number of parallel compute
#define THREADS_WRITE_GRAPH 24

#define MAX_IO_THREADS 1
#define MSSG_ROUNDUP 0.1 // should be set higher when each processor only contains a small number of partitions
#define READ_RATIO_ROUNDUP 0.002

extern long cpu_threads;
int read_length; // set the inital value in: init_input!!!
static ull rid = 1; //read id with initialization to be 1, global to uniformly identify all reads;
static uint unit_size; // unit size of one encoded read (without read id), set the inital value in: init_input!!!
extern int cutoff;
extern int mpi_run;

/* for debug: */
int last_flag;
int filter_number = 0;
//static evaltime_t lds, lde;
//extern float ldtime;
float readfile_time = 0;

double read_ratio;
double mssg_factor = 0;

ull * max_msp_malloc;
extern int * dflag;
extern int * queue;
ull * read_size;
size_t total_file_size_worker = 0; // total read size of a worker on an input file
size_t file_size_offset = 0; // current file size that has been read of a worker

MPI_File input_file;
FILE * input;
FILE * output;
FILE ** mspout;
length_range_t h_range;

/******** input & output ********/
static seq_t * write_buffer; // buffer to write data to output file
static ull write_offset;
static ull write_size; // data size allocated for the write buffer

/******** msp output & input *******/
static msp_meta_t * msp_meta[THREADS_MSP_META]; // msp output meta data

//uint max_num_kmers = 0; //get the statistics when doing minimum substring partitioning
//uint max_num_spks = 0;
//uint max_spksize = 0;
//uint max_num_kmers = 75609096; //hg7-512 data; used for initializing and testing, updated at the end of msp
//uint max_num_spks = 7029665;
//uint max_spksize = 68416697;
//uint max_num_kmers=824616557; //bbb-512
//uint max_num_spks=60774697;
//uint max_spksize=647034176;
uint max_num_spks = 66774;
uint max_spksize = 989232;
uint max_num_kmers = 744293;

ull total_num_edges = 0;
ull distinct_edges = 0;
ull total_num_vs = 0;
int biggest_partition = 0;

static uint spksize[NUM_OF_PARTITIONS];
static uint numspks[NUM_OF_PARTITIONS];
static uint numkmers[NUM_OF_PARTITIONS];

/* definitions of sequence information */
int table[256]; // lookup table for encoding data
int check[256]; // check table for filtering illegal data
static char rev_table[4] = {'A', 'C', 'G', 'T'};


#define INDEX(a, b) ((a) > (b) ? 1 : 0)
#define sum(result, arr, n) {		\
	int i;							\
	for (i = 0; i < n; i++) 		\
		result += arr[i];	}

#define prefix(result, arr, n) {			\
	int i;							\
	for (i = 0; i < n; i++)			\
		result += arr[i];		}

void
init_lookup_table (void)
{
	int i;
	for (i = 0; i < 256; i++)
	{
		table[i] = check[i] = -1;
	}
	table['A'] = table['a'] = 0;
	table['C'] = table['c'] = 1;
	table['G'] = table['g'] = 2;
	table['T'] = table['t'] = 3;
	table['N'] = table['n'] = 0;

	check['A'] = check['a'] = 0;
	check['C'] = check['c'] = 1;
	check['G'] = check['g'] = 2;
	check['T'] = check['t'] = 3;
}

void get_rev (char * read, char * rev, int len)
{
	int i;
	for (i = 0; i < len; i++)
	{
		rev[i] = rev_table[read[len - i - 1] - 'A'];
	}
}

/* This function read one line to the str and return the length ('\n' included if it existed in the line),
 * or return 0 if the data is illegal
 */
int
get_one_read (char ** pstr, offset_t * read_offset, size_t end)
{
	int num_of_char = 0;
	int count = 0; // count number of illegal characters

	/* get one read -- be careful of illegal memory access if last line of data is abnormal */
//		*pstr += *read_offset;
		if (**pstr == '\n')
		{
			(*read_offset)++;
			(*pstr)++;
			return 0;
		}
		while (*(*pstr + num_of_char) != '\n')
		{
			if(check[*(*pstr + num_of_char)] == -1)
			{
				++count;
//				*(*pstr + num_of_char) = 'A';
			}
			++num_of_char;
		}

		if (count > CUTOFF_N || num_of_char != read_length)
		{
			*read_offset += num_of_char + 1;
			*pstr += num_of_char + 1;
			return 0;
		}/* illegal read: skip this line */

		return (num_of_char + 1);

}

int get_one_read_2lines (char ** pstr, offset_t * read_offset, size_t end)
{
	/* get one read -- be careful of illegal memory access if last line of data is abnormal */
		if (**pstr == '+' || **pstr == '-') //must be fastq file: skip two lines
		{
			while (**pstr != '\n' && (*read_offset) < end)
			{
				(*read_offset)++;
				(*pstr)++;
			}
			(*read_offset)++;
			(*pstr)++;
			while (**pstr != '\n' && (*read_offset) < end)
			{
				(*read_offset)++;
				(*pstr)++;
			}
			(*read_offset)++;
			(*pstr)++;
			return 0;
		}
		if (**pstr == '>' || **pstr == '@') //skip one line and get one read
		{
			while (**pstr != '\n' && (*read_offset) < end)
			{
				(*read_offset)++;
				(*pstr)++;
			}
			(*read_offset)++;
			(*pstr)++;
		}

		int num_of_char = 0;
		int count = 0; // count number of illegal characters
		while (*(*pstr + num_of_char) != '\n' && (*read_offset + num_of_char) < end)
		{
			if(check[*(*pstr + num_of_char)] == -1)
			{
				++count;
//				*(*pstr + num_of_char) = 'A';
			}
			++num_of_char;
		}

		if (count > CUTOFF_N || num_of_char != read_length)
		{
			*read_offset += num_of_char + 1;
			*pstr += num_of_char + 1;
			return 0;
		}/* illegal read: skip this line */

		return (num_of_char + 1);
}

void
skip_one_line (char ** pstr, offset_t * read_offset)
{
	if (*(*pstr - 1) == '\n')
		return;
	while (*(*pstr)++ != '\n')
	{
		(*read_offset)++;
	}
	return;
}

double my_round(double number, unsigned int bits) {
    long long integerPart = number;
    number -= integerPart;
    unsigned int i;
    for (i = 0; i < bits; ++i)
        number *= 10;
    number = (long long) (number + 0.5);
    for (i = 0; i < bits; ++i)
        number /= 10;
    return integerPart + number;
}

// return number of reads / read size
double
estimate_num_reads_from_input (char * filename, int read_length)
{
	FILE * input;
	if ((input = fopen(filename, "r")) == NULL)
	{
		printf ("Error: cannot open input file!\n");
		exit (0);
	}
	int count = 0;
	char buf[2048];
	char * ptr;
	offset_t roff;
	int i;
	int total_len = 0;
	for (i=0; i<128; i++)
	{
		int len;
		if ((len = strlen(fgets(buf, 2048, input))) == 2048)
		{
			printf ("Error in input file! please check it!\n");
			exit (0);
		}
		total_len += len;
		int read_len;
		roff = 0;
		ptr = buf;
		if ((read_len = get_one_read (&ptr, &roff, 2048*1024)) != 0)
//		if ((read_len = get_one_read_2lines (&ptr, &roff, 2048*1024)) != 0)
			count++;
	}
	printf ("read ratio before return: %f\n", (float)count/total_len);
	read_ratio = (double)count/total_len;
	read_ratio = my_round (read_ratio, 3) + (READ_RATIO_ROUNDUP);
	return read_ratio;
}

int
finalize_input (void)
{
	fclose (input);

	return 0;
}

int
finalize_mpi_input (void)
{
	return MPI_File_close (&input_file);
}

int
finalize_msp_input (FILE ** mspinput, int world_size)
{
	int i;
	for (i=0; i<world_size; i++)
		fclose (mspinput[i]);
	return 0;
}

int
init_code (seq_t * hcode_buffer)
{
	return 0;
}

void
reset_code_buffer (seq_t * code_buffer)
{
	memset (code_buffer, 0, sizeof(seq_t) * CODE_BUF_SIZE);
}

int
finalize_code (void)
{
	return 0;
}


int
init_output (char * filename)
{
	if ((output = fopen (filename, "w")) == NULL)
	{
		printf ("Cannot open output file %s\n", filename);
		exit (0);
	}

	write_buffer = (seq_t *) malloc (sizeof(seq_t) * BUF_SIZE);
	CHECK_PTR_RETURN (write_buffer, "init write buffer malloc\n");

	write_offset = 0;
	write_size = BUF_SIZE;

	return 0;
}

int
finalize_output (void)
{
	fclose (output);
	free (write_buffer);

	write_offset = 0;
	write_size = 0;

	return 0;
}

int
init_msp_output (char * file_dir, int num_of_partitions, int world_rank)
{
	int i;

	mspout = (FILE **) malloc (sizeof(FILE *) * num_of_partitions);
	CHECK_PTR_RETURN (mspout, "init msp output file pointers\n");


	char temp[FILENAME_LENGTH];
	memset (temp, 0, FILENAME_LENGTH * sizeof(char));
	for (i = 0; i < num_of_partitions; i++)
	{
		sprintf (temp, "%s/msp%d_%d", file_dir, i, world_rank);
		if ((mspout[i] = fopen (temp, "w")) == NULL)
		{
			printf ("Can't open mspout file %d\n", i);
			exit (0);
		}
	}

	return 0;
}

void
reset_msp_buffer (seq_t * msp_buf, offset_t * offset_ptr, uint mspid, uint part_buf_size)
{
	memset (msp_buf + (ull) mspid * part_buf_size, 0, sizeof(seq_t) * part_buf_size);
	offset_ptr[mspid] = 0;
}

int
finalize_msp_output (int num_of_partitions)
{
	int i;
	for (i = 0; i < num_of_partitions; i++)
	{
		fclose (mspout[i]);
	}

	free (mspout);

	return 0;
}



void
set_length_range (int k, int read_length)
{
//	length_range_t range;
	int ave = read_length / AVE_NUM_SPK;
	h_range.l1 = 0;
	h_range.l4 = ave;
	h_range.l2 = ave / 3;
	h_range.l3 = ave / 3 * 2;
	h_range.l5 = h_range.l4 + (read_length - k - h_range.l4) / 2;
//	return range;
}

/* return the number of runs to process the whole input file */
int
init_input (char * filename, int rlen, int world_size, int world_rank)
{
	if ((input = fopen(filename, "r")) == NULL)
	{
		printf ("Error: cannot open input file!\n");
		exit (0);
	}

	fseek (input, 0, SEEK_END);
	size_t file_size = ftell (input);
	size_t filesize_per_worker = (file_size + world_size - 1)/world_size;
	if (filesize_per_worker <= world_size)
		filesize_per_worker = file_size/world_size;
	size_t file_start = filesize_per_worker * world_rank;
	size_t file_end = file_start + filesize_per_worker;
	if(file_start > file_size)
	{
		file_start = file_size;
	}
	if(file_end > file_size)
	{
		file_end = file_size;
	}
	total_file_size_worker = file_end - file_start;
	int nstreams = (total_file_size_worker + BUF_SIZE - 1) / BUF_SIZE; //maximum number of streams
	fseek (input, file_start, SEEK_SET);

	read_length = rlen;
	unit_size = (read_length + 3) / 4;
	printf ("WORLD_RANK %d: number of streams : %d, unit_size of an encoded read: %d\n", world_rank, nstreams, unit_size);

	return nstreams;
}

size_t
read_file (char * read_buf, int world_rank)
{
	if (file_size_offset == 0 && world_rank != 0) // skip a line if necessary
	{
		fseek (input, -1, SEEK_CUR);
		fread (read_buf, 1, LINE, input);
		int offset=0;
		while (read_buf[offset++] != '\n') // skip a line
		{
			if (offset >= LINE)
			{
				printf ("error in reading an extra line!\n");
				exit (-1);
			}
		}
		fseek (input, offset-LINE, SEEK_CUR);
		file_size_offset += offset;
	}
	// begin reading a bulk of file
	size_t read_size = fread ((char *)read_buf, 1, BUF_SIZE, input);
	if (total_file_size_worker <= file_size_offset + read_size) // end of reading file
	{
		int offset = 0;
		char *ptr = read_buf + (total_file_size_worker - file_size_offset);
		while (ptr[offset] != '\n' && offset < file_size_offset + read_size - total_file_size_worker)
		{
			offset++;
		}
		total_file_size_worker += offset; // point to the end
		read_size = total_file_size_worker - file_size_offset;
	}
	else
	{
		char * ptr = read_buf + read_size;
		int offset = 0;
		while (*(--ptr) != '\n')
		{
			offset++;
			if (offset >= LINE)
			{
				printf ("error in backing an extra line!\n");
				exit (-1);
			}
		}
		fseek (input, ptr + 1 - (read_buf + read_size), SEEK_CUR);
		read_size = ptr + 1 - read_buf;
	}

	file_size_offset += read_size;
	return read_size;
}

int init_mpi_input (char * file_name, int rlen, int world_size, int world_rank)
{
	MPI_Offset file_size, file_size_per_worker, file_start, file_end;

	if(MPI_File_open (MPI_COMM_WORLD, file_name, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file) != 0)
	{
		debug("Worker %d: can not open file %s for read!\n", world_rank, file_name);
		MPI_Abort (MPI_COMM_WORLD, -1);
	}
	MPI_File_get_size (input_file, &file_size);
	printf ("world rank %d: file size got: %lu\n", world_rank, file_size);

	file_size_per_worker = (file_size + world_size - 1) / world_size;
	file_start = world_rank * file_size_per_worker;
	file_end = file_start + file_size_per_worker;
	if(file_start > file_size)
	{
		file_start = file_size;
	}
	if(file_end > file_size)
	{
		file_end = file_size;
	}

	total_file_size_worker = file_end - file_start;
	int nstreams = (total_file_size_worker + BUF_SIZE - 1) / BUF_SIZE; //maximum number of streams
//	fseek (input, file_start, SEEK_SET);
	MPI_File_seek (input_file, file_start, MPI_SEEK_SET);

	read_length = rlen;
	unit_size = (read_length + 3) / 4;
	printf ("WORLD_RANK %d: number of streams : %d, unit_size of an encoded read: %d, total_file_size_worker=%lu, "
			"file_start=%lu, file_end=%lu\n", world_rank, nstreams, unit_size, total_file_size_worker, file_start, file_end);

	return nstreams;
}

size_t mpi_read_file (char * read_buf, int world_size, int world_rank)
{
	if (file_size_offset == 0 && world_rank != 0) // skip a line if necessary
	{
		MPI_File_seek (input_file, -1, MPI_SEEK_CUR);
		MPI_File_read (input_file, read_buf, LINE, MPI_CHAR, MPI_STATUS_IGNORE);
//		fseek (input, -1, SEEK_CUR);
//		fread (read_buf, 1, LINE, input);
		int offset=0;
		while (read_buf[offset++] != '\n') // skip a line
		{
			if (offset >= LINE)
			{
				printf ("error in reading an extra line!\n");
				MPI_Abort (MPI_COMM_WORLD, -1);
			}
		}
		MPI_File_seek (input_file, offset-LINE, MPI_SEEK_CUR);
//		fseek (input, offset-LINE, SEEK_CUR);
		file_size_offset += offset;
		printf ("WORLD RANK %d: offset skipped: %lu\n", world_rank, offset);
	}

	// begin reading a bulk of file
	size_t read_size;
	if (total_file_size_worker <= file_size_offset + BUF_SIZE) // end of reading file
	{
		read_size = total_file_size_worker - file_size_offset;
		if (world_rank != world_size - 1)
		{
			MPI_File_read (input_file, read_buf, read_size + LINE, MPI_CHAR, MPI_STATUS_IGNORE);
			int offset = 0;
			char *ptr = read_buf + read_size;
			while (ptr[offset++] != '\n')
			{
				if (offset >= LINE)
				{
					printf ("error in reading an extra line!\n");
					MPI_Abort (MPI_COMM_WORLD, -1);
				}
			}
			total_file_size_worker += offset; // point to the end
			read_size = total_file_size_worker - file_size_offset;
			printf ("world rank %d: read_size = %d\n", world_rank, read_size);
//			MPI_File_seek (input_file, (offset-LINE), MPI_SEEK_CUR);
		}
		else
		{
			printf ("world rank %d: read_size = %d\n", world_rank, read_size);
			MPI_File_read (input_file, read_buf, read_size, MPI_CHAR, MPI_STATUS_IGNORE);
		}
	}
	else
	{
		MPI_File_read (input_file, read_buf, BUF_SIZE, MPI_CHAR, MPI_STATUS_IGNORE);
		char * ptr = read_buf + BUF_SIZE;
		while (*(--ptr) != '\n')
		{}
		MPI_File_seek (input_file, -(read_buf + BUF_SIZE - ptr - 1), MPI_SEEK_CUR);
//		fseek (input, ptr + 1 - (read_buf + read_size), SEEK_CUR);
		read_size = ptr + 1 - read_buf;
	}
	file_size_offset += read_size;
	return read_size;
}

/* Encode one read with 2 bit per character, length for one read: (read_length + 3) / 4 */
uch
bitcode (seq_t * code_buf, seq_t * line_buf, uch len)
{
	uch i;
	for (i = 0; i < len; i++)
	{
		code_buf[i / 4] |= table[line_buf[i]] << ((3 - (i % 4)) * 2);
	}
	return ( (len + 3) / 4 );
}

/* test this: */
uch
bitcode_reverse (seq_t * code_buf, seq_t * line_buf, uch len)
{
	uch i;
	for (i = 0; i < len; i++)
	{
		code_buf[i / 4] |= (3 - table[line_buf[len - 1 - i]]) << ((3 - (i % 4)) * 2);
	}

	return ( (len + 3) / 4 );
}

static int encode_kmer (unit_kmer_t * kmer, seq_t * read, int k)
{
	if (k*2 > (KMER_UNIT_LENGTH * KMER_UNIT_BITS))
	{
		printf ("kmer length exceeds the limit!\n");
		exit(0);
	}
	int unit_length = k*2/KMER_UNIT_BITS;
	int i = 0;
	int j = 0;
	for (; i<unit_length; i++)
	{
		*kmer = 0;
		int j;
		for (j=0; j<(KMER_UNIT_BITS)/2; j++)
		{
			*kmer |= table[read[i*(KMER_UNIT_BITS)/2 + j]] << (KMER_UNIT_BITS - 2 - j*2);
		}
		kmer++;
	}
	if ((k*2)%KMER_UNIT_BITS)
		*kmer = 0;
	for (j=0; j<((k*2)%KMER_UNIT_BITS)/2; j++)
	{
		*kmer |= table[read[i*(KMER_UNIT_BITS)/2 + j]] << (KMER_UNIT_BITS - 2 - j*2);
	}
	return unit_length;
}

/* decode encoded sequences (either reads or superkmers) */
uch
decode (seq_t * dec_buf, seq_t * read_ptr, uch len)
{
	int i;
	for (i = 0; i < len; i++)
	{
		dec_buf[i] = rev_table[(read_ptr[i / 4] >> ((3 - i % 4) * 2)) & 0x3];
	}
	dec_buf[i] = '\n';

	return ((len + 3) / 4); // return number of bytes of encoded string
}

char * get_min (char * read, char * rev, int k, int p)
{
	int i = 0;
	char * pstr;
	char * rpstr;
	char * minpstr;
	char * rminpstr;
	pstr = minpstr = read;
	rpstr = rminpstr = rev;

	for (i = 1; i < k - p + 1; i++)
	{
		pstr++;
		rpstr++;
		if (strncmp (pstr, minpstr, p) < 0)
		{
			minpstr = pstr;
		}
		if (strncmp (rpstr, rminpstr, p) < 0)
		{
			rminpstr = rpstr;
		}
	}
	if (strncmp (minpstr, rminpstr, p) <= 0)
	{
		return minpstr;
	}
	else
		return rminpstr;
}

uint compute_msp (int t, int p, int k, int num_of_partitions, char * read_buffer[], ull read_size, uch * d_msp_ptr, char ** rbufs[], uint * rnums[])
{
	omp_set_num_threads (cpu_threads);

#pragma omp parallel
{
	/* debug: test number of threads */
	int nths = omp_get_num_threads ();
	int thid = omp_get_thread_num ();
	if (nths != cpu_threads)
	{
		printf ("ERROR!!!!!! NUMBER OF THREADS: %d\n", nths);
		exit(1);
	}
	int turn = t;
	ull msp_malloc = max_msp_malloc[turn];
	ull max_num_ms = read_length - k + 1;
//	float time = 0;
	ull time = 0;
	evaltime_t start, end;
//	printf ("id: %d, num of threads: %d\n", thid, nths);

	/* Be careful: read_size may not be divided by cpu_threads! */
	size_t read_size_per_thread = (read_size + cpu_threads - 1) / cpu_threads;
	if (read_size_per_thread <= cpu_threads)
		read_size_per_thread = read_size / cpu_threads;
	char * read_ptr = read_buffer[turn] + thid * read_size_per_thread;
	if (thid == cpu_threads - 1)
	{
		read_size_per_thread = read_size - read_size_per_thread * thid;
		if (read_size - read_size_per_thread * thid < 0)
			printf ("ATTENTION!!! calculate read size per thread error!!!\n");
	}
	uint rnum = 0;

	int len;
	offset_t roffset = 0;
//	offset_t coffset = 0;

	if (thid > 0)
		skip_one_line (&read_ptr, &roffset);
	/* if the thread reads starting at the middle of a line, then this line will be processed by its predecessor, thus it skip this line */

	rbufs[turn][thid] = read_ptr; // Store the start position of each read buffer area for each thread

	/* initiate msp variables */
	char * read;
	char revs[read_length];

	char * minpos;
	char * search;
	msp_t d_msp;
	uint num_of_reads = read_size * read_ratio;
	uint num_reads_per_thread = (num_of_reads + cpu_threads - 1) / cpu_threads;
	if (num_reads_per_thread <= cpu_threads)
		num_reads_per_thread = num_of_reads / cpu_threads;
//		ull msp_malloc = max_msp_malloc[turn];
	ull usize = max_num_ms * sizeof(msp_id_t) + sizeof(uch) * max_num_ms + 1;
	ull align_size = sizeof(msp_id_t) - sizeof(uch) * (max_num_ms + 1) * num_reads_per_thread % sizeof(msp_id_t);
	d_msp.nums = d_msp_ptr + (usize * num_reads_per_thread + align_size) * thid;
	d_msp.poses = d_msp.nums + sizeof(uch) * num_reads_per_thread;
	d_msp.ids = (msp_id_t *)(d_msp.nums + sizeof(uch) * (max_num_ms + 1) * num_reads_per_thread + (sizeof(msp_id_t) - sizeof(uch) * (max_num_ms + 1) * num_reads_per_thread % sizeof(msp_id_t)));

	size_t read_size_end;
	if (thid == nths - 1)
		read_size_end = read_size_per_thread - 1;
	else
		read_size_end = read_size_per_thread;
	while (roffset < read_size_end)
	{
		if (rnum >= num_reads_per_thread)
		{
			printf ("maximum number of reads exceeds the initial set up! please reset the input buffer!!"
					"rnum = %u, num_reads_per_thread = %u\n", rnum, num_reads_per_thread);
			exit(0);
		}
		if ((len = get_one_read (&read_ptr, &roffset, read_size_end)) == 0)
//		if ((len = get_one_read_2lines (&read_ptr, &roffset, read_size_end)) == 0)
				continue;
		read = read_ptr;

	gettimeofday (&start, NULL);

	/* COMPUTING MSP INFO FOR A READ BEGINS */
	get_rev (read, revs, read_length);
	int i = 0;
	uch num = 0;
	minpos =  get_min (read + i, revs + (read_length - k - i), k, p);
#ifdef LONG_KMER
	kmer_t kmer = {0, 0, 0, 0};
#else
	kmer_t kmer = {0, 0};
#endif
	encode_kmer ((unit_kmer_t*)&kmer, read+i, k);
//	d_msp.ids[rnum * max_num_ms + num] = get_partition_id_from_string (minpos, p, num_of_partitions);
	d_msp.ids[rnum * max_num_ms + num] = get_pid_from_kmer(&kmer, k, p, num_of_partitions);
	d_msp.poses[rnum * max_num_ms + num] = 0;
	msp_id_t old, new;
	old = d_msp.ids[rnum * max_num_ms + num];
	for (i = 1; i < read_length - k + 1; i++)
	{
		search = get_min (read + i, revs + (read_length - k - i), k, p);
		kmer.x=0; kmer.y=0;
#ifdef LONG_KMER
		kmer.z=0; kmer.w=0;
#endif
		encode_kmer ((unit_kmer_t*)&kmer, read+i, k);
		new = get_pid_from_kmer(&kmer, k, p, num_of_partitions);
//		if (strncmp (search, minpos, p) != 0)
		if (new != old)
		{
			num++; // be careful: num may exceeds max_num_ms!
			if (num > max_num_ms)
			{
				printf ("number of %d-minimum-substring exceeds the predefined limit for output!\n", p);
				exit(0);
			}
//			d_msp.ids[rnum * max_num_ms + num] = get_partition_id_from_string (search, p, num_of_partitions);
			d_msp.ids[rnum * max_num_ms + num] = new;
			d_msp.poses[rnum * max_num_ms + num] = i; // test this
			minpos = search;
			old = new;
		}
	}
	d_msp.nums[rnum] = num;
#ifdef DEBUG
	if (num >= max_num_ms)
	{
		printf ("Careful: number of superkmers exceeds predefined size!\n");
		exit (0);
	}
#endif
	/* COMPUTING MSP INFO FOR A READ ENDS */

	gettimeofday (&end, NULL);
	time += ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec));

		roffset += len;
		read_ptr += len;
//		coffset += unit_size;
		rnum++;
	}

//	reads->offset[thid] = coffset;
	rnums[turn][thid] = rnum;
//	msp_time[thid] += time;
}

	uint num_of_reads = 0;
	sum(num_of_reads, rnums[t], cpu_threads);
//	printf ("number of reads counted: %d\n", num_of_reads);
	return num_of_reads;
}

uint
parse_data (char ** read_buffer, read_buf_t * reads, ull read_size, int turn, char *** rbufs, uint ** rnums)
{
	seq_t * code_buffer = reads->buf;
	reset_code_buffer (code_buffer); // set element to 0!

	omp_set_num_threads (cpu_threads);

#pragma omp parallel
{
	/* debug: test number of threads */
	int nths = omp_get_num_threads ();
	int thid = omp_get_thread_num ();
	if (nths != cpu_threads)
	{
		printf ("ERROR!!!!!! NUMBER OF THREADS: %d\n", nths);
		exit(1);
	}

	/* Be careful: read_size may not be divided by cpu_threads! */
	size_t read_size_per_thread = (read_size + cpu_threads - 1) / cpu_threads;
	if (read_size_per_thread <= cpu_threads)
		read_size_per_thread = read_size / cpu_threads;
	char * read_ptr = read_buffer[turn] + thid * read_size_per_thread;
	seq_t * code_ptr = code_buffer + thid * (CODE_BUF_SIZE / cpu_threads);

	if (thid == cpu_threads - 1)
	{
		read_size_per_thread = read_size - read_size_per_thread * (cpu_threads - 1);
		if (read_size - read_size_per_thread * thid < 0)
			printf ("!!! ATTENTION: read_size per thread for the last thread error!!!\n");
	}
	uint rnum = 0;

	int len;
	offset_t roffset = 0;
	offset_t coffset = 0;

	if (thid > 0)
		skip_one_line (&read_ptr, &roffset);
	/* if the thread reads starting at the middle of a line, then this line will be processed by its predecessor, thus it skip this line */

	rbufs[turn][thid] = read_ptr; // Store the start position of each read buffer area for each thread

	size_t end;
	if (thid == nths - 1)
		end = read_size_per_thread - 1;
	else
		end = read_size_per_thread;
	while (roffset < end)
	{
		if ((len = get_one_read (&read_ptr, &roffset, end)) == 0)
//		if ((len = get_one_read_2lines (&read_ptr, &roffset, end)) == 0)
				continue;

		bitcode (code_ptr + coffset, read_ptr, read_length);
		roffset += len;
		read_ptr += len;
		coffset += unit_size;
		rnum++;
	}
	if (roffset >= read_size_per_thread)
	{
//		printf ("thid: %d, roffset - read_size_per_thread = %lu\n", thid, roffset - read_size_per_thread);
//		exit(0);
	}

	reads->offset[thid] = coffset;
	rnums[turn][thid] = rnum;
}
//	reads->buf = code_buffer;

	uint num_of_reads = 0;
	sum(num_of_reads, rnums[turn], cpu_threads);
//	printf ("total number of reads processed in parsing data for GPU: %u\n", num_of_reads);
	return num_of_reads;
}


void
init_msp_meta (int num_of_partitions, double read_ratio, int read_length, int k)
{
	int i, j;
	uint msp_meta_size = ceil(BUF_SIZE * read_ratio * (read_length - k + 1) * 2/ (num_of_partitions * NUM_OF_RANGES * cpu_threads));
	printf ("msp meta size: %u\n", msp_meta_size);
	for (i = 0; i < cpu_threads; i++)
	{
		msp_meta[i] = (msp_meta_t *) malloc (sizeof(msp_meta_t) * num_of_partitions * NUM_OF_RANGES);
		CHECK_PTR_RETURN (msp_meta[i], "init msp meta array\n");
		for (j = 0; j < num_of_partitions * NUM_OF_RANGES; j++)
		{
			msp_meta[i][j].idarr = (rid_t *) malloc (sizeof(rid_t) * msp_meta_size);
			msp_meta[i][j].lenarr = (uch *) malloc (sizeof(uch) * msp_meta_size);
			msp_meta[i][j].spkbuf = (seq_t *) malloc (sizeof(seq_t) * msp_meta_size * (unit_size+1));
			CHECK_PTR_RETURN (msp_meta[i][j].idarr, "init msp meta id array\n");
			CHECK_PTR_RETURN (msp_meta[i][j].lenarr, "init msp meta length array\n");
			CHECK_PTR_RETURN (msp_meta[i][j].spkbuf, "init msp meta spkbuf error!\n");
			msp_meta[i][j].size = msp_meta_size;
			msp_meta[i][j].spksize = msp_meta_size * unit_size;
			msp_meta[i][j].offset = 0;
			msp_meta[i][j].spkoffset = 0;
			msp_meta[i][j].num_kmers = 0;
		}
	}
	memset (spksize, 0, sizeof(uint) * NUM_OF_PARTITIONS);
	memset (numspks, 0, sizeof(uint) * NUM_OF_PARTITIONS);
	memset (numkmers, 0, sizeof(uint) * NUM_OF_PARTITIONS);

}

/* set offset of one msp buffer meta information to 0 */
void
reset_msp_meta (msp_meta_t * meta)
{
	meta->offset = 0;
	meta->spkoffset = 0;
	meta->num_kmers = 0;
	memset (meta->spkbuf, 0, sizeof(seq_t) * meta->spksize);
}

/* expand one particular msp meta buffer by twice */
void
expand_msp_meta (msp_meta_t * meta)
{
	meta->idarr = (rid_t *) realloc (meta->idarr, sizeof(rid_t) * meta->size * 2);
	meta->lenarr = (uch *) realloc (meta->lenarr, sizeof(uch) * meta->size * 2);
	CHECK_PTR_RETURN (meta->idarr, "expand msp meta id array\n");
	CHECK_PTR_RETURN (meta->lenarr, "expand msp length array\n");
	meta->size *= 2;
}

void
expand_meta_spks (msp_meta_t * meta)
{
	meta->spkbuf = (seq_t *) realloc (meta->spkbuf, sizeof(seq_t) * meta->spksize * 2);
	CHECK_PTR_RETURN (meta->spkbuf, "expand msp meta superkmer buffer\n");
	meta->spksize *= 2;
}

void
finalize_msp_meta (int k, int num_of_partitions, offset_t * max_kmers, offset_t * max_spks, offset_t * max_spksizes, int world_size, int world_rank)
{
	int i, j;
	for (i = 0; i < num_of_partitions; i++)
	{
		if (max_spks[world_rank] < numspks[i])
		{
			max_spks[world_rank] = numspks[i];
		}
		if (max_spksizes[world_rank] < spksize[i])
		{
			max_spksizes[world_rank] = spksize[i];
//			max_num_kmers = max_spksize * 4 - k * numspks[i];
		}
		if (max_kmers[world_rank] < numkmers[i])
		{
			max_kmers[world_rank] = numkmers[i];
			biggest_partition = i;
		}
	}
	for (i = 0; i < cpu_threads; i++)
	{
		for (j = 0; j < num_of_partitions * NUM_OF_RANGES; j++)
		{
			free(msp_meta[i][j].idarr);
			free(msp_meta[i][j].lenarr);
			free(msp_meta[i][j].spkbuf);
		}
		free(msp_meta[i]);
	}

	if (mpi_run > 0)
	{
		printf ("world rank %d: gathering numbers:::::::::::::::::::\n", world_rank);
		MPI_Allgather (&max_spks[world_rank], 1, MPI_INT, max_spks, 1, MPI_INT, MPI_COMM_WORLD);
		MPI_Allgather (&max_spksizes[world_rank], 1, MPI_INT, max_spksizes, 1, MPI_INT, MPI_COMM_WORLD);
		MPI_Allgather (&max_kmers[world_rank], 1, MPI_INT, max_kmers, 1, MPI_INT, MPI_COMM_WORLD);
	}
	max_num_kmers = 0;
	max_num_spks = 0;
	max_spksize = 0;
	for (i=0; i<world_size; i++)
	{
		max_num_kmers += max_kmers[i];
		max_num_spks += max_spks[i];
		max_spksize += max_spksizes[i];
	}
	printf ("WORLD_RANK %d: ----------------FINALIZE MSP META: max_num_spks %u, max_spksize %u, max_num_kmers %u, biggest partition %d "
			"----------\n", world_rank, max_num_spks, max_spksize, max_num_kmers, biggest_partition);
}

void
output_msp_cpu (uch * msp_arr, char ** rbufs[], uint * rnums[], int k, int num_of_partitions, int wrt_id, int world_rank)
{

	int turn = queue[wrt_id];
	omp_set_num_threads (cpu_threads);

#pragma omp parallel
{
	length_range_t range = h_range;

	/* debug: test number of threads */
	int nths = omp_get_num_threads ();
	int thid = omp_get_thread_num ();
	if (nths != cpu_threads)
	{
		printf ("ERROR!!!!!!!!!set number of threads failure!!!!!! real number of threads:%d\n", nths);
//		exit (1);
	}

	char * read_ptr = rbufs[turn][thid];
	int rlen;

	msp_id_t mspid;
	uch len;
	uch num;
	offset_t read_offset = 0; // rescan from the beginning of read buffer
	ull i, j;
	uint scan = 0;
	prefix(scan, rnums[turn], thid);
	uint local_rid = rid + scan;
	int local_k = k;
	uch * spk_nums;
	uch * spk_poses;
	msp_id_t * spk_ids;
	uint num_of_reads;
	ull max_num_ms = read_length - k + 1;

	msp_meta_t * meta_ptr = msp_meta[thid];
	if (dflag[turn] == 0) // turn is from processing result of CPU
	{
		num_of_reads = read_size[turn] * read_ratio;
		uint num_reads_per_thread = (num_of_reads + cpu_threads - 1) / cpu_threads;
		if (num_reads_per_thread <= cpu_threads)
			num_reads_per_thread = num_of_reads / cpu_threads;
//		ull msp_malloc = max_msp_malloc[turn];
		ull usize = max_num_ms * sizeof(msp_id_t) + sizeof(uch) * max_num_ms + 1;
		ull align_size = sizeof(msp_id_t) - sizeof(uch) * (max_num_ms + 1) * num_reads_per_thread % sizeof(msp_id_t);
		spk_nums = msp_arr + (usize * num_reads_per_thread + align_size) * thid;
		//******** be careful here, num_reads_per_thread may not be a precise estimation for each thread, in this case, thread access to partitioned buffer may cause error!!!!!!!!!!!!!!
//		spk_nums = msp_arr + msp_malloc / cpu_threads * thid;
		spk_poses = spk_nums + sizeof(uch) * num_reads_per_thread;
		spk_ids = (msp_id_t *)(spk_nums + sizeof(uch) * (max_num_ms + 1) * num_reads_per_thread + (sizeof(msp_id_t) - sizeof(uch) * (max_num_ms + 1) * num_reads_per_thread % sizeof(msp_id_t)));
	}
	else // turn is from processing result of GPU
	{
		num_of_reads = 0;
		sum(num_of_reads, rnums[turn], cpu_threads);
//		printf ("number of reads in output msp from GPU(s): %u\n", num_of_reads);
		msp_t mspptr;
		mspptr.nums = msp_arr;
		mspptr.poses = msp_arr + sizeof(uch) * num_of_reads;
		mspptr.ids = (msp_id_t *)(msp_arr + sizeof(uch) * (max_num_ms + 1) * num_of_reads + (sizeof(msp_id_t) - sizeof(uch) * (max_num_ms + 1) * num_of_reads % sizeof(msp_id_t)));

		spk_nums = mspptr.nums + scan;
		spk_poses = mspptr.poses + scan * max_num_ms;
		spk_ids = mspptr.ids + scan * max_num_ms;
	}
	uint read_num = rnums[turn][thid];
//	printf ("number of reads processed in msp_compute of partition %d thread %d: %u\n", turn, thid, read_num);

	size_t read_size_per_thread = (read_size[turn] + cpu_threads - 1) / cpu_threads;
	if (read_size_per_thread <= cpu_threads)
		read_size_per_thread = read_size[turn] / cpu_threads;
	if (thid == cpu_threads - 1)
	{
		read_size_per_thread = read_size[turn] - read_size_per_thread * (cpu_threads - 1);
		if (read_size[turn] - read_size_per_thread * thid < 0)
			printf ("!!! ATTENTION: read_size per thread for the last thread error!!!\n");
	}

	size_t read_size_end;
	if (thid == nths - 1)
		read_size_end = read_size_per_thread - 1;
	else
		read_size_end = read_size_per_thread;
	/* cut reads into superkmers; process read by read */
	for (i = 0; i < read_num; i++)
	{
		/* Get one legal read from  read buffer */
		while ((rlen = get_one_read (&read_ptr, &read_offset, read_size_end)) == 0) {}
//		while ((rlen = get_one_read_2lines (&read_ptr, &read_offset, read_size_end)) == 0) {}

		num = spk_nums[i]; // number of superkmers

		for (j = 0; j < num; j++)
		{
			mspid = spk_ids[i * max_num_ms + j];
			/* for debug only: */
#ifdef DEBUG
			if (mspid >= num_of_partitions)
			{
				print_error ("error in msp array id!\n");
				while (1) {};
			}
#endif
			len = spk_poses[i * max_num_ms + j + 1] - spk_poses[i * max_num_ms + j] + 1;
//			kmern[mspid] += len;
			mspid = mspid * NUM_OF_RANGES + INDEX(len, (range.l1)) + INDEX(len, (range.l2)) + INDEX(len, (range.l3)) + INDEX(len, (range.l4)) + INDEX(len, (range.l5)) - 1;
			meta_ptr[mspid].idarr[meta_ptr[mspid].offset] = local_rid;

			if (j == 0)
			{
				meta_ptr[mspid].spkoffset += bitcode (meta_ptr[mspid].spkbuf + meta_ptr[mspid].spkoffset, read_ptr + spk_poses[i * max_num_ms + j], len + local_k - 1);
				meta_ptr[mspid].num_kmers += len;
				meta_ptr[mspid].lenarr[meta_ptr[mspid].offset] = len - 1; // set the most significant bit to be 0
			}
			else
			{
				meta_ptr[mspid].spkoffset += bitcode (meta_ptr[mspid].spkbuf + meta_ptr[mspid].spkoffset, read_ptr + spk_poses[i * max_num_ms + j] - 1, len + local_k);
				meta_ptr[mspid].num_kmers += len;
				meta_ptr[mspid].lenarr[meta_ptr[mspid].offset] = len | 0x80; // set the most significant bit to be 1
			}

			meta_ptr[mspid].offset++;
			if (meta_ptr[mspid].offset >= meta_ptr[mspid].size-1)
			{
				printf ("Expand msp meta happened!\n");
				expand_msp_meta (meta_ptr + mspid);
				printf ("Expand msp meta spk happened!\n");
				expand_meta_spks (meta_ptr + mspid);
			}

		}

		/* end of the read */
		mspid = spk_ids[i * max_num_ms + j];
		/* for debug only: */
#ifdef DEBUG
		if (mspid >= num_of_partitions)
		{
			print_error ("error in msp array id!\n");
			while (1) {};
		}
#endif
		len = read_length - spk_poses[i * max_num_ms + j] - local_k + 1;
//		kmern[mspid] += len;
		mspid = mspid * NUM_OF_RANGES + INDEX(len, (range.l1)) + INDEX(len, (range.l2)) + INDEX(len, (range.l3)) + INDEX(len, (range.l4)) + INDEX(len, (range.l5)) - 1;
		meta_ptr[mspid].idarr[meta_ptr[mspid].offset] = local_rid;

		if (j == 0)
		{
			meta_ptr[mspid].spkoffset += bitcode (meta_ptr[mspid].spkbuf + meta_ptr[mspid].spkoffset, read_ptr + spk_poses[i * max_num_ms + j], len + local_k - 1);
			meta_ptr[mspid].num_kmers += len;
			meta_ptr[mspid].lenarr[meta_ptr[mspid].offset] = len; // set the most significant bit to be 0
		} /* full read: take care of this case ! */
		else
		{
			meta_ptr[mspid].spkoffset += bitcode_reverse (meta_ptr[mspid].spkbuf + meta_ptr[mspid].spkoffset, read_ptr + spk_poses[i * max_num_ms + j] - 1, len + local_k);
			meta_ptr[mspid].num_kmers += len;
			meta_ptr[mspid].lenarr[meta_ptr[mspid].offset] = len; // set the most significant bit to be 0
		}/* store the reverse instead of forward kmer */

		meta_ptr[mspid].offset++;
		if (meta_ptr[mspid].offset >= meta_ptr[mspid].size-1)
		{
			printf ("Expand msp meta happened!\n");
			expand_msp_meta (meta_ptr + mspid);
			printf ("Expand msp meta spk happened!\n");
			expand_meta_spks (meta_ptr + mspid);
		}
		read_offset += rlen;
		read_ptr += rlen;
		local_rid++; // global read id identification
	}
}

	/* finally write out msp buffers to files */
	FILE * file;
	int i, j, t;
	omp_set_num_threads (MAX_IO_THREADS);
#pragma omp parallel private(file, i, j, t)
	{
		int thid = omp_get_thread_num ();
		int io_th = num_of_partitions / MAX_IO_THREADS;
		int ios;
		if (thid == MAX_IO_THREADS - 1)
			ios = num_of_partitions - io_th * thid;
		else
			ios = io_th;
		int r;
		for (r = 0; r < ios; r++)
		{
		file = mspout[thid * io_th + r];
//		fwrite (&kmercnt[i], 1, sizeof(uint), file);
		for (j = 0; j < NUM_OF_RANGES; j++)
		{
			for (t = 0; t < cpu_threads; t++)
			{
				if (msp_meta[t][(thid * io_th + r) * NUM_OF_RANGES + j].offset == 0)
					continue;
				fwrite (&(msp_meta[t][(thid * io_th + r) * NUM_OF_RANGES + j].offset), 1, sizeof(uint), file);
				fwrite (&(msp_meta[t][(thid * io_th + r) * NUM_OF_RANGES + j].spkoffset), 1, sizeof(offset_t), file);
				fwrite (msp_meta[t][(thid * io_th + r) * NUM_OF_RANGES + j].idarr, sizeof(rid_t), msp_meta[t][(thid * io_th + r) * NUM_OF_RANGES + j].offset, file);
				fwrite (msp_meta[t][(thid * io_th + r) * NUM_OF_RANGES + j].lenarr, sizeof(uch), msp_meta[t][(thid * io_th + r) * NUM_OF_RANGES + j].offset, file);
				fwrite (msp_meta[t][(thid * io_th + r) * NUM_OF_RANGES + j].spkbuf, sizeof(seq_t), msp_meta[t][(thid * io_th + r) * NUM_OF_RANGES + j].spkoffset, file);
				spksize[thid * io_th + r] += msp_meta[t][(thid * io_th + r) * NUM_OF_RANGES + j].spkoffset;
				numspks[thid * io_th + r] += msp_meta[t][(thid * io_th + r) * NUM_OF_RANGES + j].offset;
				numkmers[thid * io_th + r] += msp_meta[t][(thid * io_th + r) * NUM_OF_RANGES +j].num_kmers;
				reset_msp_meta (&msp_meta[t][(thid * io_th + r) * NUM_OF_RANGES + j]);
			}
		}
		}
	}

	sum(rid, rnums[turn], cpu_threads);
}

uint *
prefix_sum (uch * lenarr, uint * indices, uint size, int k, uint * num_of_kmers)
{
	uint i;

	indices[0] = 0;
	uint len;
	for (i = 1; i < size + 1; i++)
	{
		len = lenarr[i - 1] & 0x7f;
		*num_of_kmers += len;
		indices[i] = (len + k) + 3;
		if (len + k - 1 == read_length)
			indices[i] -= 1;
		indices[i] /= 4;
		indices[i] += indices[i - 1];
		if (lenarr[i - 1] & 0x80)
			*num_of_kmers -= 1;
	}
	return indices;
}


uint
get_superkmers (char * filename[], spkmer_t * spkmers, int k, FILE ** mspinput)
{
	uint num_of_kmers = 0;
	uint num_of_spk;
	uint total_num_of_spk = 0;
	uint rid_offset = 0;
	uint len_offset = 0;
	rid_t * ridarr;
	uch * lenarr;
//	uint * indices;
	uint read_size = 0;

	offset_t mspsize; // size of superkmer string
	ull total_mspsize = 0;

	ull offset; // offset of reading input superkmer file
	size_t file_size[NUM_OF_RANGES];

	int i;
	for(i = 0; i < NUM_OF_RANGES; i++)
	{
		if( (mspinput[i] = fopen (filename[i], "r")) == NULL)
			printf ("Cannot open msp file %s\n", filename[i]);

		fseek (mspinput[i], 0, SEEK_END);
		if ((file_size[i] = ftell (mspinput[i])) == 0)
		{
//			fclose (mspinput[i]);
			continue;
		}

		fseek (mspinput[i], 0, SEEK_SET);

		while (ftell (mspinput[i]) < file_size[i])
		{
			fread (&num_of_spk, 1, sizeof(uint), mspinput[i]);
			fread (&mspsize, 1, sizeof(offset_t), mspinput[i]);
			total_num_of_spk += num_of_spk;
			total_mspsize += mspsize;
			offset = num_of_spk * (sizeof(rid_t) + sizeof(uch)) + mspsize * sizeof(seq_t);
			fseek (mspinput[i], offset, SEEK_CUR);
		}
	}

	if (total_num_of_spk == 0)
		return 0;

	ridarr = (rid_t *) malloc (sizeof(rid_t) * total_num_of_spk);
	lenarr = (uch *) malloc (sizeof(uch) * total_num_of_spk);
//	indices = (uint *) malloc (sizeof(uint) * total_num_of_spk);

	/* Allocate read buffer memory */
	seq_t * read_buffer = (seq_t *) malloc (sizeof(seq_t) * total_mspsize);

	CHECK_PTR_RETURN (ridarr, "init spk rid array malloc with file %s\n", filename[i]);
	CHECK_PTR_RETURN (lenarr, "init spk length array malloc with file %s\n", filename[i]);
	CHECK_PTR_RETURN (read_buffer, "init read buffer for getting superkmers with file %s\n", filename[i]);
//	CHECK_PTR_RETURN (indices, "malloc spk indices array in prefix_sum\n");
	for (i = 0; i < NUM_OF_RANGES; i++)
	{
		fseek (mspinput[i], 0, SEEK_SET);
		while (ftell (mspinput[i]) < file_size[i])
		{
			fread (&num_of_spk, 1, sizeof(uint), mspinput[i]);
			fread (&mspsize, 1, sizeof(offset_t), mspinput[i]);
			rid_offset += fread (ridarr + rid_offset, sizeof(rid_t), num_of_spk, mspinput[i]);
			len_offset += fread (lenarr + len_offset, sizeof(uch), num_of_spk, mspinput[i]);
			read_size += fread (read_buffer + read_size, sizeof(seq_t), mspsize, mspinput[i]);
		}
	}

//	prefix_sum (lenarr, indices, total_num_of_spk, k, &num_of_kmers);
	spkmers->spks = read_buffer;
	spkmers->ridarr = ridarr;
	spkmers->lenarr = lenarr;
	spkmers->num = total_num_of_spk;
//	spkmers->indices = indices;

	return total_num_of_spk;
}

msp_stats_t
get_spk_stats (int pid, int world_size, char * msp_dir, FILE ** mspinput)
{
	offset_t num_of_spk;
	offset_t total_num_of_spk = 0;
	msp_stats_t stats = {0, 0};
	char filename[FILENAME_LENGTH];

	offset_t mspsize; // size of superkmer string
	size_t total_mspsize = 0;

	size_t offset; // offset of reading input superkmer file
	size_t file_size;

	int i;
	for(i = 0; i < world_size; i++)
	{
		memset (filename, 0, sizeof(char)*FILENAME_LENGTH);
		sprintf (filename, "%s/msp%d_%d", msp_dir, pid, i);
		if( (mspinput[i] = fopen (filename, "r")) == NULL)
			printf ("Cannot open msp file %s\n", filename);

		fseek (mspinput[i], 0, SEEK_END);
		if ((file_size = ftell (mspinput[i])) == 0)
		{
//			fclose (mspinput[i]);
			continue;
//			return stats;
		}

		fseek (mspinput[i], 0, SEEK_SET);

		while (ftell (mspinput[i]) < file_size)
		{
			fread (&num_of_spk, 1, sizeof(uint), mspinput[i]);
			fread (&mspsize, 1, sizeof(offset_t), mspinput[i]);
			total_num_of_spk += num_of_spk;
			total_mspsize += mspsize;
			offset = num_of_spk * (sizeof(rid_t) + sizeof(uch)) + mspsize * sizeof(seq_t);
			fseek (mspinput[i], offset, SEEK_CUR);
		}
	}
	stats.spk_num = total_num_of_spk;
	stats.spk_size = total_mspsize;
	return stats;
}

uint
load_superkmers (FILE ** mspinput, rid_t * ridarr, uch * lenarr, uint * indices, seq_t * read_buffer, uint total_num_of_spk, int k, int world_size)
{
//	gettimeofday (&lds, NULL);
	uint num_of_kmers = 0;
	uint num_of_spk;
	uint rid_offset = 0;
	uint len_offset = 0;

	uint read_size = 0;

	offset_t mspsize; // size of superkmer string
	ull total_mspsize = 0;

	size_t file_size;

	int i;
	for (i = 0; i < world_size; i++)
	{
		fseek (mspinput[i], 0, SEEK_END);
		if ((file_size = ftell (mspinput[i])) == 0)
		{
//			fclose (mspinput[i]);
			continue;
//			return 0;
		}

		fseek (mspinput[i], 0, SEEK_SET);
		while (ftell (mspinput[i]) < file_size)
		{
			fread (&num_of_spk, 1, sizeof(uint), mspinput[i]);
			fread (&mspsize, 1, sizeof(offset_t), mspinput[i]);
			rid_offset += fread (ridarr + rid_offset, sizeof(rid_t), num_of_spk, mspinput[i]);
			len_offset += fread (lenarr + len_offset, sizeof(uch), num_of_spk, mspinput[i]);
			read_size += fread (read_buffer + read_size, sizeof(seq_t), mspsize, mspinput[i]);
		}
	}
//	gettimeofday (&lde, NULL);
//	ldtime += (float)((lde.tv_sec * 1000000 + lde.tv_usec) - (lds.tv_sec * 1000000 + lds.tv_usec)) / 1000;

	prefix_sum (lenarr, indices, total_num_of_spk, k, &num_of_kmers);

	return num_of_kmers;
}

void
decode_kmer (kmer_t * kmer, seq_t * line_buf, int k)
{
	int i;
	unit_kmer_t * kmer_ptr = (unit_kmer_t *) kmer;
	for (i = 0; i < k; i ++)
	{
		line_buf[i] = rev_table[(*(kmer_ptr + i / KMER_UNIT_CHAR_LENGTH) >> (KMER_UNIT_BITS - 2 * (i % KMER_UNIT_CHAR_LENGTH + 1))) & 0x3];
	}
}

uint
write_graph (dbgraph_t * graph, int k)
{
//	printf ("++++++++++++write graph+++++++++++\n");
//	uint num = graph->num;
	uint countn = 0;
	uint size = graph->size;
	node_t * nodes = graph->nodes;
	seq_t * line_buf = (seq_t *) malloc (sizeof(seq_t) * LINE);
	CHECK_PTR_RETURN (line_buf, "init line buffer for decoding kmer in write_graph\n");
	memset (line_buf, 0, sizeof(seq_t) * LINE);
//	printf ("graph size %u, graph nodes: %p\n", size, graph->nodes);

	uint i;
	uch j;
	int sum;
	uint total_edges = 0;
	uint total_distinct_edges = 0;
	omp_set_num_threads (THREADS_WRITE_GRAPH);
#pragma omp parallel private(i, j, sum) reduction(+:countn) reduction(+:total_edges) reduction(+:total_distinct_edges)
	{
		int thid = omp_get_thread_num ();
//		printf ("write graph id %d\n", thid);
		uint local_countn = 0;
		uint num_edges = 0;
		uint distinct_num_edges = 0;
		uint size_per_thread = (size + THREADS_WRITE_GRAPH - 1) / THREADS_WRITE_GRAPH;
		if (size_per_thread <= THREADS_WRITE_GRAPH)
			size_per_thread = size / THREADS_WRITE_GRAPH;
		node_t * local_nodes = nodes + size_per_thread * thid;
		if (thid == THREADS_WRITE_GRAPH - 1)
			size_per_thread = size - size_per_thread * (thid);
		if (size - size_per_thread * thid < 0)
			printf ("ATTENTION!!! size_per_thread error!!! %ld\n", size - size_per_thread);
	for (i = 0; i < size_per_thread; i++)
	{
		if (local_nodes[i].occupied == 0) continue;
/*		sum = 0;
		for (j = 0; j < EDGE_DIC_SIZE; j++)
		{
			sum += local_nodes[i].edge[j];
		}
		if (sum < cutoff) continue;*/
		if (local_nodes[i].occupied != 2)
		{
			printf ("error!\n");
			exit(0);
		}
		for (j = 0; j < EDGE_DIC_SIZE; j++)
		{
//			num_edges += local_nodes[i].edge[j] & 0xff;
			num_edges += *((ull*)&(local_nodes[i].edge)) >> (8 * j) & 0xff;
			if (*((ull*)&(local_nodes[i].edge)) >> (8 * j) & 0xff)
				distinct_num_edges += 1;
		}
		local_countn++;

/*		decode_kmer (&(nodes[i].kmer), line_buf, k);
		write_offset += sprintf (write_buffer + write_offset, "%s\t", line_buf);

		for (j = 0; j < 4; j++)
		{
			write_offset += sprintf (write_buffer + write_offset, "%c\t%d\t", rev_table[j], nodes[i].edge[j]);
		}
		for (; j < EDGE_DIC_SIZE; j++)
		{
			write_offset += sprintf (write_buffer + write_offset, "%c\t%d\t", rev_table[j - 4], nodes[i].edge[j]);
		}
		write_offset += sprintf (write_buffer + write_offset, "%u\n", nodes[i].rid);

		if (write_offset + LINE * 2 >= write_size)
		{
			fwrite (write_buffer, sizeof(seq_t), write_offset, output);
			write_offset = 0;
		}*/
	}
	countn += local_countn;
	total_edges += num_edges;
	total_distinct_edges += distinct_num_edges;
	}

//	fwrite (write_buffer, sizeof(seq_t), write_offset, output);
	write_offset = 0;
	free (line_buf);
	total_num_edges += total_edges;
	distinct_edges += total_distinct_edges;
	printf ("Number of distinct edges: %lu\n", distinct_edges);

	return countn;
}

uint
gather_sorted_dbgraph (dbgraph_t * graph, dbtable_t * tbs, subgraph_t * subgraph, uint num_of_kmers, int pid, int start_pid, int np_node)
{
	uint countn = 0;
	uint size = graph->size;
	node_t * nodes = graph->nodes;

	uint i;
	uch j;
	int sum;
	uint total_edges = 0;
	uint total_distinct_edges = 0;
	voff_t vs_offsets[THREADS_WRITE_GRAPH+1];
	memset (vs_offsets, 0, sizeof(voff_t) * (THREADS_WRITE_GRAPH+1));
	omp_set_num_threads (THREADS_WRITE_GRAPH);
#pragma omp parallel private(i, j, sum) reduction(+:countn) reduction(+:total_edges) reduction(+:total_distinct_edges)
	{
		int thid = omp_get_thread_num ();
		int nths = omp_get_num_threads ();
		if (nths != THREADS_WRITE_GRAPH)
		{
			printf ("Error in setting threads for gathering dbgraph!\n");
			exit(0);
		}
//		printf ("write graph id %d\n", thid);
		uint local_countn = 0;
		uint num_edges = 0;
		uint distinct_num_edges = 0;
		uint size_per_thread = (size + THREADS_WRITE_GRAPH - 1) / THREADS_WRITE_GRAPH;
		if (size_per_thread <= THREADS_WRITE_GRAPH)
			size_per_thread = size / THREADS_WRITE_GRAPH;
		node_t * local_nodes = nodes + size_per_thread * thid;
		if (thid == THREADS_WRITE_GRAPH - 1)
			size_per_thread = size - size_per_thread * (thid);
		if (size - size_per_thread * thid < 0)
			printf ("ATTENTION!!! size_per_thread error!!! %ld\n", size - size_per_thread);
		voff_t offset = 0;
		for (i = 0; i < size_per_thread; i++)
		{
			if (local_nodes[i].occupied == 0) continue;
			if (local_nodes[i].occupied != 2)
			{
				printf ("error!\n");
				exit(0);
			}
			int cf = 0;
			for (j = 0; j < EDGE_DIC_SIZE; j++)
			{
				cf += *((ull*)&(local_nodes[i].edge)) >> (8 * j) & 0xff;
			}
			if (cf < cutoff)
			{
				local_nodes[i].occupied = 0;
				continue;
			}
			for (j = 0; j < EDGE_DIC_SIZE; j++)
			{
				if (*((ull*)&(local_nodes[i].edge)) >> (8 * j) & 0xff)
					distinct_num_edges += 1;
			}
			num_edges += cf;
			offset++;
			local_countn++;

		}
		vs_offsets[thid+1] = offset;
		countn += local_countn;
		total_edges += num_edges;
		total_distinct_edges += distinct_num_edges;
	}
	tbs[pid].buf = (entry_t*) malloc (sizeof(entry_t) * countn);
	tbs[pid].size = countn;
	tbs[pid].num_elems = num_of_kmers;
	(subgraph->subgraphs)[pid].size = countn;
	(subgraph->subgraphs)[pid].id = pid + start_pid;
	inclusive_prefix_sum (vs_offsets, THREADS_WRITE_GRAPH+1);

	omp_set_num_threads (THREADS_WRITE_GRAPH);
#pragma omp parallel
	{
		int thid = omp_get_thread_num ();
		int nths = omp_get_num_threads ();
		if (nths != THREADS_WRITE_GRAPH)
		{
			printf ("Error in setting threads for gathering dbgraph!\n");
			exit(0);
		}
		uint size_per_thread = (size + THREADS_WRITE_GRAPH - 1) / THREADS_WRITE_GRAPH;
		if (size_per_thread <= THREADS_WRITE_GRAPH)
			size_per_thread = size / THREADS_WRITE_GRAPH;
		node_t * local_nodes = nodes + size_per_thread * thid;
		if (thid == THREADS_WRITE_GRAPH - 1)
			size_per_thread = size - size_per_thread * (thid);
		voff_t gather_start = vs_offsets[thid];
		entry_t * vs = tbs[pid].buf + gather_start;
		voff_t offset = 0;
		int i;
		for (i = 0; i < size_per_thread; i++)
		{
			if (local_nodes[i].occupied == 0) continue;
			if (local_nodes[i].occupied != 2)
			{
				printf ("error!\n");
				exit(0);
			}
			vs[offset].kmer = local_nodes[i].kmer;
			vs[offset].edge = local_nodes[i].edge;
			vs[offset].occupied = local_nodes[i].occupied;
			offset++;
		}
	}
	tbb_entry_sort (tbs[pid].buf, countn);

	total_num_edges += total_edges;
	distinct_edges += total_distinct_edges;
	total_num_vs += countn;
	double factor = (double)distinct_edges / total_num_vs;
	factor = my_round(factor, 3) + MSSG_ROUNDUP;
	if (factor > mssg_factor)
		mssg_factor = factor;
//	if (pid == np_node - 1)
	{
		printf ("partition %d: number of vertices: %u\n", pid, countn);
		printf ("MMMMMMMMMMMMMMMMMMMMMMMMMMMMM MSSG FACTOR REPORT::::::::::::::::\n");
		printf ("Number of distinct edges: %lu, mssg_factor = %.4f, mssg_roundup = %.4f\n", distinct_edges, mssg_factor, MSSG_ROUNDUP);
		printf ("MMMMMMMMMMMMMMMMMMMMMMMMMMMMM MSSG FACTOR REPORT::::::::::::::::\n");
		printf ("TTTTTTTTTTTTTTTTTTTTTT total number of vertices counted in output: %u\n", total_num_vs);
	}

	return countn;
}
