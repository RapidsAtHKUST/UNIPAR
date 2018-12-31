/*
 * utility.h
 *
 *  Created on: 2015-3-5
 *      Author: qiushuang
 */

/* TODO: Use existent predefined macros: change unit to uint_32t, ull to uint64_t */

#ifndef UTILITY_H_
#define UTILITY_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
#include <stdint.h>

#ifndef LITTLE_ENDIAN
#define LITTLE_ENDIAN
#endif

#define FILENAME_LENGTH 150

#if defined(__CUDACC__) // NVCC
   #define MY_ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
  #define MY_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
  #define MY_ALIGN(n) __declspec(align(n))
#else
  #error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif

#ifndef CHAR_BIT
#define CHAR_BIT 8
#endif

typedef unsigned long long ull;
typedef unsigned long long idtype;
typedef unsigned char uch;
typedef uint32_t uint;
typedef uint offset_t;

typedef struct timeval evaltime_t;

#define MIN(x, y) { (x) < (y) ? (x) : (y) }


/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

#define CHECK_PTR_RETURN(ptr, ...) {									\
	if (ptr == NULL) {												\
		printf (__VA_ARGS__);										\
		printf ("Error in returned value: NULL\n");					\
		exit (1);													\
	} }

#define write_int_arr(arr, size, buf, offset) {				\
	int i;														\
	for (i = 0; i < size; i++) {								\
		offset += sprintf (buf, "%d\t", arr[i]);				\
	} 															\
	offset += sprintf (buf, "\n");								\
}

#define write_str_arr (arr, size, buf, offset) {				\
	int i;														\
	for (i = 0; i < size; i++) {								\
		offset += sprintf (buf, "%c\t", arr[i]);				\
	} 															\
	offset += sprintf (buf, "\n");								\
}

#define debug(format, ...) \
		fprintf (stderr, format, ##__VA_ARGS__);

#define print_error(format, ...) {					\
		fprintf (stderr, format, ##__VA_ARGS__);	\
		exit (1); }

#define print_exec_time(start, end, ...) {	\
		printf (__VA_ARGS__);				\
		printf ("%f milliseconds\n", (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000); }

#define print_num_of_ms(msp_arr, num_of_reads) {						\
	int i, j;															\
	FILE * msparr;														\
	if ((msparr = fopen("msp_arr_file", "w")) == NULL) 					\
		{ printf("open msp array file fails\n");}						\
	for (i = 0; i < num_of_reads; i++)	{								\
		fprintf (msparr, "%d\t", msp_arr.nums[i]);						\
	}																	\
	fprintf (msparr, "\n\n");												\
	for (i = 0; i < num_of_reads; i++) {								\
		for (j = 0; j < 64; j++) {								\
			fprintf (msparr, "%d\t", msp_arr.poses[i * 64 + j]);\
		}																\
		fprintf (msparr, "\n");											\
	}																	\
	fprintf (msparr, "\n");												\
	for (i = 0; i < num_of_reads; i++) {								\
		for (j = 0; j < 64; j++) {								\
			fprintf (msparr, "%d\t", msp_arr.ids[i * 64 + j]);  \
		}																\
		fprintf (msparr, "\n");											\
	}																	\
	fprintf (msparr, "\n");												\
	fclose (msparr);													\
	}

#endif /* UTILITY_H_ */
