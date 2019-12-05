#include<stdio.h>

#include "gpu/kmer_count_kernels.h"
#include "gpu/bloom-gpu/fhash.h"
#include "gpu/bloom-gpu/nvbio/bloom_filter.h"
#include "gpu/bloom-gpu/nvbio/types.h"
#include "gpu/SlabHash/src/slab_hash.cuh"
#include "gpu/SlabHash/src/gpu_hash_table.cuh"

//converts char to a binary rappresentation (00 01 10 11) of the four bases 
void convCharToBin64(const char* array, vector<uint32_t> &h_kmers,vector<uint32_t> &h_index,int kmer_len){
	int conv_table[100]={0};
	conv_table[65] = 0x00;
	conv_table[67] = 0x01;
        conv_table[71] = 0x02;
	conv_table[84] = 0x03;
	uint32_t head = 0;
   	uint32_t tail = 0;
    	for(int i = 0; i<(kmer_len-16);i++){
		head = head<<2;
		head |= conv_table[(int)array[i]];
	}
    
   	for(int j = (kmer_len-16); j<kmer_len;j++){
            	tail = tail<<2;
            	tail |=conv_table[(int)array[j]];
    	}
    	h_index.push_back(head);
    	h_kmers.push_back(tail);
}

std::vector<uint32_t> HashTableGPU(uint32_t* h_kmers,uint32_t* h_index,int kmer_len,uint64_t num_kmers){
	uint32_t num_keys = num_kmers;
	assert(num_keys>0);
	float expected_chain = 1;
	uint32_t num_elements_per_unit = 15;
	uint32_t expected_elements_per_bucket =
			expected_chain * num_elements_per_unit;
	uint32_t num_buckets = (num_keys + expected_elements_per_bucket - 1) /
        			expected_elements_per_bucket;
//	num_buckets = 100000;	
	using KeyT = uint32_t;
	using ValueT = uint32_t;
    	std::vector<ValueT> h_result(num_kmers);
	
	gpu_hash_table<KeyT, ValueT, SlabHashTypeT::ConcurrentMap>
        	hash_table(num_kmers, num_buckets,pow(4, kmer_len-16),0, 1);

	printf("insert num kmers : %" PRId64 "\n",num_kmers);	
        float insert_time = hash_table.hash_insert_buffered(h_kmers, h_index,num_kmers);
                //hash_table.hash_insert(h_kmers_to_insert.data(),h_index.data(),num_kmers);
       		printf("finish insert");
	
	printf("insert time%f\n",insert_time);
	float search_time =
                hash_table.hash_search(h_kmers,h_index, h_result.data(),num_kmers);
	printf("insert time : %f, search time : %f\n",insert_time,search_time);
	return h_result;
}

