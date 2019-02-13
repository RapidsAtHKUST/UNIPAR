/*
 * node_visit.c
 *
 *  Created on: 2017-5-12
 *      Author: qiushuang
 */
#include <omp.h>
#include "tbb/parallel_sort.h"
#include "tbb/parallel_scan.h"
#include "tbb/blocked_range.h"
#include "../include/comm.h"

//******* Prefix sum for tbb parallel scan ************
template <typename T>
class Body {
    T sum;
    T* const y;
    const T* const z;
public:
    Body( T y_[], const T z_[] ) : sum(0), z(z_), y(y_) {}
    T get_sum() const { return sum; }

    template<typename Tag>
    void operator()( const tbb::blocked_range<size_t>& r, Tag ) {
        T temp = sum;
        for( int i=r.begin(); i<r.end(); ++i ) {
            temp = temp + z[i];
            if( Tag::is_final_scan() )
                y[i] = temp;
        }
        sum = temp;
    }
    Body( Body& b, tbb::split ) : z(b.z), y(b.y), sum(0) {}
    void reverse_join( Body& a ) { sum = a.sum + sum; }
    void assign( Body& b ) { sum = b.sum; }
};

template <typename T>
T TbbParallelScan( T y[], const T z[], size_t n ) {
    Body<T> body(y,z);
    tbb::parallel_scan( tbb::blocked_range<size_t>(0,n), body );
    return body.get_sum();
}

template <typename T>
void sequential_prefix_sum (T * array, int num)
{
	int i;
	for (i = 0; i < num - 1; i++)
	{
		array[i + 1] += array[i];
	}
}

extern "C"
{
void inclusive_prefix_sum_long (unsigned long long * array, int num)
{
	sequential_prefix_sum <unsigned long long> (array, num);
}

struct key_value_sort
{
	bool operator() (const kv_t kv1, const kv_t kv2) const
	{
		return (kv1.k < kv2.k); // careful!!!!!!!! CAN NOT USE kv1.k <= kv2.k
	}
};

void tbb_kv_sort (kv_t * kvs, uint size)
{
	tbb::parallel_sort (kvs, kvs + size, key_value_sort());
}

struct pair_sort
{
	bool operator() (const pair_t p1, const pair_t p2) const
	{
		return ((unsigned long long)(p1.kmer) < (unsigned long long)(p2.kmer));
	}
};

void tbb_pair_sort (pair_t * buf, uint size)
{
	tbb::parallel_sort (buf, buf+size, pair_sort());
}

struct vertex_sort
{
	bool operator() (const vertex_t & mssg1, const vertex_t & mssg2) const
	{
#ifdef LONG_KMER
		if (mssg1.kmer.x == mssg2.kmer.x)
		{
			if (mssg1.kmer.y == mssg2.kmer.y)
			{
				if (mssg1.kmer.z == mssg2.kmer.z)
				{
					return (mssg1.kmer.w < mssg2.kmer.w);
				}
				else return (mssg1.kmer.z < mssg2.kmer.z);
			}
			else return (mssg1.kmer.y < mssg2.kmer.y);
		}
		else return (mssg1.kmer.x < mssg2.kmer.x);
#else
		if (mssg1.kmer.x == mssg2.kmer.x)
			return (mssg1.kmer.y < mssg2.kmer.y);
		else
			return (mssg1.kmer.x < mssg2.kmer.x);

#endif
	}
};

struct kmer_vid_sort
{
	bool operator() (const kmer_vid_t mssg1, const kmer_vid_t mssg2) const
	{
#ifdef LONG_KMER
		if (mssg1.kmer.x == mssg2.kmer.x)
		{
			if (mssg1.kmer.y == mssg2.kmer.y)
			{
				if (mssg1.kmer.z == mssg2.kmer.z)
				{
					return (mssg1.kmer.w < mssg2.kmer.w);
				}
				else return (mssg1.kmer.z < mssg2.kmer.z);
			}
			else return (mssg1.kmer.y < mssg2.kmer.y);
		}
		else return (mssg1.kmer.x < mssg2.kmer.x);
#else
		if (mssg1.kmer.x == mssg2.kmer.x)
			return (mssg1.kmer.y < mssg2.kmer.y);
		else
			return (mssg1.kmer.x < mssg2.kmer.x);
#endif
	}
};

struct entry_sort
{
	bool operator() (const entry_t & mssg1, const entry_t & mssg2) const
	{
#ifdef LONG_KMER
		if (mssg1.kmer.x == mssg2.kmer.x)
		{
			if (mssg1.kmer.y == mssg2.kmer.y)
			{
				if (mssg1.kmer.z == mssg2.kmer.z)
				{
					return (mssg1.kmer.w < mssg2.kmer.w);
				}
				else return (mssg1.kmer.z < mssg2.kmer.z);
			}
			else return (mssg1.kmer.y < mssg2.kmer.y);
		}
		else return (mssg1.kmer.x < mssg2.kmer.x);
#else
		if (mssg1.kmer.x == mssg2.kmer.x)
			return (mssg1.kmer.y < mssg2.kmer.y);
		else
			return (mssg1.kmer.x < mssg2.kmer.x);
#endif
	}
};

void tbb_kmer_vid_sort (kmer_vid_t * buf, voff_t size)
{
	tbb::parallel_sort (buf, buf+size, kmer_vid_sort());
}

void tbb_vertex_sort (vertex_t * mssg_buf, uint size)
{
	tbb::parallel_sort (mssg_buf, mssg_buf + size, vertex_sort());
}

void tbb_entry_sort (entry_t * mssg_buf, uint size)
{
	tbb::parallel_sort (mssg_buf, mssg_buf + size, entry_sort());
}

void tbb_scan_long (size_t * input, size_t * output, size_t num)
{
	TbbParallelScan<size_t> (input, output, num);
}

void tbb_scan_uint (voff_t * input, voff_t * output, size_t num)
{
	TbbParallelScan<voff_t> (input, output, num);
}
}
