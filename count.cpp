
#include<stdio.h>

#include "gpu/kmer_count_kernels.h"
#include "gpu/bloom-gpu/fhash.h"
#include "gpu/bloom-gpu/nvbio/bloom_filter.h"
#include "gpu/bloom-gpu/nvbio/types.h"
#include "gpu/SlabHash/src/slab_hash.cuh"
#include "gpu/SlabHash/src/gpu_hash_table.cuh"


//handle all the GPU calls
std::vector<uint32_t> HashTableGPU(std::vector<uint32_t>& h_kmers_to_insert,vector<uint32_t> h_index,uint32_t*h_key_blocks, uint32_t *h_whitelist_blocks,uint64_t tot_key_blocks, uint64_t tot_whitelist_blocks,int kmer_len,uint64_t num_kmers){
        uint32_t num_keys = num_kmers;
        assert(num_keys>0);
        float expected_chain = 1;
        uint32_t num_elements_per_unit = 15;
        uint32_t expected_elements_per_bucket =
                        expected_chain * num_elements_per_unit;
        uint32_t num_buckets = (num_keys + expected_elements_per_bucket - 1) /
                                expected_elements_per_bucket;

        using KeyT = uint32_t;
        using ValueT = uint32_t;
        std::vector<ValueT> h_result(32*tot_whitelist_blocks);



	//init the hash table
        gpu_hash_table<KeyT, ValueT, SlabHashTypeT::ConcurrentMap>
                hash_table(tot_key_blocks, tot_whitelist_blocks,num_kmers, num_buckets, 0, 1);
        printf("h_kmers_to insert size : %d num kmers : %d\n",h_kmers_to_insert.size(),num_kmers);
       
	//performs the insert
	float insert_time =
                hash_table.hash_insert_on_reads(h_key_blocks,h_whitelist_blocks);
        //performs the search, this part must be change because is not meaningful given that is possible to get the full table from the gpu memory
	float search_time =
                hash_table.hash_search_on_reads( h_result.data());
        
	printf("insert time : %f, search time : %f\n",insert_time,search_time);
        return h_result;
}


//converts the reads from char to binary on blocks of 32 bit
void convCharToBinRead(const char* array, vector<uint32_t> &reads){
        int conv_table[100]={0};
        conv_table[65] = 0x00;
        conv_table[67] = 0x01;
        conv_table[71] = 0x02;
        conv_table[84] = 0x03;
        uint32_t block = 0;
	uint32_t empty = 0;
        int iterations = strlen(array)/16;
        for(int i = 0; i<iterations;i++){
                for(int j = 0; j<16;j++){
                        block = block<<2;
                        block |= conv_table[(int)array[i*16+j]];
                }
                reads.push_back(block);
                block = 0;
        }
        block = 0;
        for(int i = 0; i<strlen(array)%16; i++){
                block = block<<2;
                block |= conv_table[(int)array[iterations*16+i]];
        }
	
        //push block on the right side
        for(int i = 0; i<16-strlen(array)%16; i++){
                block = block<<2;
        }
	reads.push_back(block);
        if(iterations%2 == 0 && strlen(array)%16!=0){
		reads.push_back(empty);
        }
	
	
}

//generates the whitelist given the read
void genWhitelist(int size, vector<uint32_t> &h_whitelist, int kmer_len){
        uint32_t conv_table[32]={0x80000000,0xc0000000,0xe0000000,0xf0000000,
                                 0xf8000000,0xfc000000,0xfe000000,0xff000000,
                                 0xff800000,0xffc00000,0xffe00000,0xfff00000,
                                 0xfff80000,0xfffc0000,0xfffe0000,0xffff0000,
                                 0xffff8000,0xffffc000,0xffffe000,0xfffff000,
                                 0xfffff800,0xfffffc00,0xfffffe00,0xffffff00,
                                 0xffffff80,0xffffffc0,0xffffffe0,0xfffffff0,
                                 0xfffffff8,0xfffffffc,0xfffffffe,0xffffffff};

        //in order to create the white list

        uint32_t full = 0xFFFFFFFF;
	uint32_t empty = 0x0;
        size = size - kmer_len + 1; //number of kmers in a read
        for(int j = 0;j<size/32; j++){
                h_whitelist.push_back(full);
        }
        if(size !=0){
//              printf("size : %d, conv index: %d \n",size, (size%32)-1);
                h_whitelist.push_back(conv_table[(size%32)-1]);

        }
	if(size%32 > 16){
		h_whitelist.push_back(empty);
	}
}
//counts the number of blocks (32 bit) in order to store the keys and the indexes
void countBlocks(vector<vector<uint32_t>> &h_reads, vector<vector<uint32_t>> &h_whitelist, uint64_t &tot_key_blocks, uint64_t &tot_whitelist_blocks){
	uint64_t tot_keys_blocks = 0;
        uint64_t tot_index_blocks = 0;
	printf("MAXTHREADS: %d \n",MAXTHREADS);
        for(uint64_t i = 0; i<MAXTHREADS; i++){
                tot_keys_blocks += h_reads[i].size();
                tot_index_blocks += h_whitelist[i].size();
        }
}


