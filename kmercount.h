#ifndef BELLA_KMERCOUNT_H_
#define BELLA_KMERCOUNT_H_

#include <inttypes.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <istream>
#include <vector>
#include <string>
#include <stdlib.h>
#include <algorithm>
#include <utility>
#include <array>
#include <tuple>
#include <queue>
#include <memory>
#include <stack>
#include <numeric>
#include <functional>
#include <cstring>
#include <string.h>
#include <math.h>
#include <cassert>
#include <ios>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/sysctl.h>
#include <map>
#include <unordered_map>
#include <omp.h>
#include <chrono>

#include "libcuckoo/cuckoohash_map.hh"
#include "libbloom/bloom64.h"

#include "kmercode/hash_funcs.h"
#include "kmercode/Kmer.hpp"
#include "kmercode/Buffer.h"
#include "kmercode/common.h"
#include "kmercode/fq_reader.h"
#include "kmercode/ParallelFASTQ.h"
#include "kmercode/bound.hpp"
#include "kmercode/hyperloglog.hpp"
#include "mtspgemm2017/common.h"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <cuda_profiler_api.h>
#include "gpu/kmer_count_kernels.h"
#include "gpu/bloom-gpu/fhash.h"
#include "gpu/bloom-gpu/nvbio/bloom_filter.h"
#include "gpu/bloom-gpu/nvbio/types.h"
#include "gpu/SlabHash/src/slab_hash.cuh"
#include "gpu/SlabHash/src/gpu_hash_table.cuh"
typedef std::chrono::high_resolution_clock Clock;


using namespace std;
#define ASCIIBASE 33 // Pacbio quality score ASCII BASE
#ifndef PRINT
#define PRINT
#endif

typedef cuckoohash_map<Kmer, int> dictionary_t; // <k-mer && reverse-complement, #kmers>

struct filedata {

    char filename[MAX_FILE_PATH];
    size_t filesize;
};

/**
 * @brief GetFiles
 * @param filename
 * @return
 */
vector<filedata>  GetFiles(char *filename) {
    int64_t totalsize = 0;
    int numfiles = 0;
    std::vector<filedata> filesview;
    
    filedata fdata;
    ifstream allfiles(filename);
    if(!allfiles.is_open()) {
        cerr << "Could not open " << filename << endl;
        exit(1);
    }
    allfiles.getline(fdata.filename,MAX_FILE_PATH);
    while(!allfiles.eof())
    {
        struct stat st;
        stat(fdata.filename, &st);
        fdata.filesize = st.st_size;
        
        filesview.push_back(fdata);
        cout << filesview.back().filename << ": " << filesview.back().filesize / (1024*1024) << " MB" << endl;
        allfiles.getline(fdata.filename,MAX_FILE_PATH);
        totalsize += fdata.filesize;
        numfiles++;
    }
    return filesview;
}

/**
 * @brief JellyFishCount
 * @param kmer_file
 * @param countsreliable_jelly
 * @param lower
 * @param upper
 */
void JellyFishCount(char *kmer_file, dictionary_t & countsreliable_jelly, int lower, int upper) 
{
    ifstream filein(kmer_file);
    string line;
    int elem;
    string kmerstr;    
    Kmer kmerfromstr;
    
    // double kdict = omp_get_wtime();
    // Jellyfish file contains all the k-mers from fastq(s)
    // It is not filtered beforehand
    // A k-mer and its reverse complement are counted separately
    dictionary_t countsjelly;
    if(filein.is_open()) 
    { 
            while(getline(filein, line)) {
                if(line.length() == 0)
                    break;

                string substring = line.substr(1);
                elem = stoi(substring);
                getline(filein, kmerstr);   
                //kmerfromstr.set_kmer(kmerstr.c_str());

                auto updatecountjelly = [&elem](int &num) { num+=elem; };
                // If the number is already in the table, it will increment its count by the occurrence of the new element. 
                // Otherwise it will insert a new entry in the table with the corresponding k-mer occurrence.
                countsjelly.upsert(kmerfromstr.rep(), updatecountjelly, elem);      
            }
    } else std::cout << "Unable to open the input file\n";
    filein.close();
    //cout << "jellyfish file parsing took: " << omp_get_wtime()-kdict << "s" << endl;

    // Reliable k-mer filter on countsjelly
    int kmer_id = 0;
    auto lt = countsjelly.lock_table(); // our counting
    for (const auto &it : lt) 
        if (it.second >= lower && it.second <= upper)
        {
            countsreliable_jelly.insert(it.first,kmer_id);
            ++kmer_id;
        }
    lt.unlock(); // unlock the table
    // Print some information about the table
    cout << "Entries within reliable range Jellyfish: " << countsreliable_jelly.size() << std::endl;    
    //cout << "Bucket count Jellyfish: " << countsjelly.bucket_count() << std::endl;
    //cout << "Load factor Jellyfish: " << countsjelly.load_factor() << std::endl;
    countsjelly.clear(); // free 
}



/**
 * @brief DeNovoCount
 * @param allfiles
 * @param countsreliable_denovo
 * @param lower
 * @param upper
 * @param kmer_len
 * @param upperlimit
 */
void DeNovoCount_cpu(vector<filedata> & allfiles, dictionary_t & countsreliable_denovo, int & lower, int & upper, int kmer_len, int depth, double & erate, size_t upperlimit /* memory limit */, BELLApars & b_parameters)
{
   
    vector < vector<Kmer> > allkmers(MAXTHREADS);
    vector < vector<double> > allquals(MAXTHREADS);
    vector < HyperLogLog > hlls(MAXTHREADS, HyperLogLog(12));   // std::vector fill constructor

    double denovocount = omp_get_wtime();
    double cardinality;
    size_t totreads = 0;

    for(auto itr=allfiles.begin(); itr!=allfiles.end(); itr++) 
    {
        #pragma omp parallel
        {
            ParallelFASTQ *pfq = new ParallelFASTQ();
            pfq->open(itr->filename, false, itr->filesize);

            vector<string> seqs;
            vector<string> quals;
            vector<string> nametags;
            size_t tlreads = 0; // thread local reads

            size_t fillstatus = 1;
            while(fillstatus) 
            { 
                fillstatus = pfq->fill_block(nametags, seqs, quals, upperlimit);
                size_t nreads = seqs.size();

                for(int i=0; i<nreads; i++) 
                {
                    // remember that the last valid position is length()-1
                    int len = seqs[i].length();
                    double rerror = 0.0;

                    for(int j=0; j<=len-kmer_len; j++)  
                    {
                        std::string kmerstrfromfastq = seqs[i].substr(j, kmer_len);
			
                        Kmer mykmer(kmerstrfromfastq.c_str());
                        Kmer lexsmall = mykmer.rep();
                        allkmers[MYTHREAD].push_back(lexsmall);
                        hlls[MYTHREAD].add((const char*) lexsmall.getBytes(), lexsmall.getNumBytes());

            		if(b_parameters.skipEstimate == false)
            		{
                        	// accuracy
                       		int bqual = (int)quals[i][j] - ASCIIBASE;
                        	double berror = pow(10,-(double)bqual/10);
                        	rerror += berror;
            		}

                    }
		    if(b_parameters.skipEstimate == false)
		    {
                    	// remaining k qual position accuracy
                    	for(int j=len-kmer_len+1; j < len; j++)
                    	{
                        	int bqual = (int)quals[i][j] - ASCIIBASE;
                        	double berror = pow(10,-(double)bqual/10);
                        	rerror += berror;
                    	}
                    	rerror = rerror / len;
                    	allquals[MYTHREAD].push_back(rerror);
		    }
                } // for(int i=0; i<nreads; i++)
                tlreads += nreads;
            } //while(fillstatus) 
            delete pfq;

            #pragma omp critical
            totreads += tlreads;
        }
    }

    // Error estimation
    if(b_parameters.skipEstimate == false)
    {
        erate = 0.0; // reset to 0 here, otherwise it cointains default or user-defined values
        #pragma omp for reduction(+:erate)
        for (int i = 0; i < MAXTHREADS; i++) 
            {
                double temp = std::accumulate(allquals[i].begin(),allquals[i].end(), 0.0);
                erate += temp/(double)allquals[i].size();
            }
        erate = erate / (double)MAXTHREADS;
    }

    // HLL reduction (serial for now) to avoid double iteration
    for (int i = 1; i < MAXTHREADS; i++) 
    {
        std::transform(hlls[0].M.begin(), hlls[0].M.end(), hlls[i].M.begin(), hlls[0].M.begin(), [](uint8_t c1, uint8_t c2) -> uint8_t{ return std::max(c1, c2); });
    }
    cardinality = hlls[0].estimate();

    double load2kmers = omp_get_wtime(); 
    cout << "Initial parsing, error estimation, and k-mer loading took: " << load2kmers - denovocount << "s\n" << endl;

    const double desired_probability_of_false_positive = 0.05;
    struct bloom * bm = (struct bloom*) malloc(sizeof(struct bloom));
    bloom_init64(bm, cardinality * 1.1, desired_probability_of_false_positive);

#ifdef PRINT
    cout << "Cardinality estimate is " << cardinality << endl;
    cout << "Table size is: " << bm->bits << " bits, " << ((double)bm->bits)/8/1024/1024 << " MB" << endl;
    cout << "Optimal number of hash functions is: " << bm->hashes << endl;
#endif

    dictionary_t countsdenovo;


	uint64_t totkmers = 0;
	for (int i = 0; i < MAXTHREADS; ++i)
		totkmers += allkmers[i].size();

	auto t1 = Clock::now();
	#pragma omp parallel
    {       
    	for(auto v:allkmers[MYTHREAD])
    	{
        	bool inBloom = (bool) bloom_check_add(bm, v.getBytes(), v.getNumBytes(),1);
        	if(inBloom) countsdenovo.insert(v, 0);
    	}
    }


    double firstpass = omp_get_wtime();
    cout << "First pass of k-mer counting took: " << firstpass - load2kmers << "s" << endl;

    free(bm); // release bloom filter memory

    // in this pass, only use entries that already are in the hash table
    auto updatecount = [](int &num) { ++num; };
#pragma omp parallel
    {       
    	for(auto v:allkmers[MYTHREAD])
    	{
        	// does nothing if the entry doesn't exist in the table
        	countsdenovo.update_fn(v,updatecount);
    	}
    }
    cout << "Second pass of k-mer counting took: " << omp_get_wtime() - firstpass << "s\n" << endl;

		auto t2 = Clock::now();
	double duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
	duration = duration / 1e6;
	printf("bloom filter insert/query took %.2f milliseconds on CPU for %d kmers\n",
		   duration, totkmers);

    //cout << "countsdenovo.size() " << countsdenovo.size() << endl;
    // Reliable bounds computation using estimated error rate from phred quality score
    lower = computeLower(depth, erate, kmer_len);
    upper = computeUpper(depth, erate, kmer_len);

    // Reliable k-mer filter on countsdenovo
	uint32_t nkmersintable = 0;
    int kmer_id_denovo = 0;
    auto lt = countsdenovo.lock_table(); // our counting
    for (const auto &it : lt)
	{
		++nkmersintable;
        if (it.second >= lower && it.second <= upper)
        {
            countsreliable_denovo.insert(it.first,kmer_id_denovo);
            ++kmer_id_denovo;
        }
	}
    lt.unlock(); // unlock the table

	cout << "#kmers in table: " << nkmersintable << "\n";

    // Print some information about the table
    if (countsreliable_denovo.size() == 0)
    {
        cout << "BELLA terminated: 0 entries within reliable range (reduce k-mer length)\n" << endl;
        // exit(0);
    } 
    else 
    {
        cout << "Entries within reliable range: " << countsreliable_denovo.size() << endl;
    }
    //cout << "Bucket count: " << countsdenovo.bucket_count() << std::endl;
    //cout << "Load factor: " << countsdenovo.load_factor() << std::endl;
    countsdenovo.clear(); // free

}
void inputConverter(std::string& reads,char* filename){
	std::string line;
	std::string tot_line;
	ifstream myfile (filename);
	
  	if (myfile.is_open())
  	{
		bool flag = false;
    		while ( getline (myfile,line) )
    		{
                        if(line.compare(0,2,"+S")==0){
                                flag = false;
                        }
			
			if(flag){
                            reads= reads+line;
                        }		
	
			if(line.compare(0,2,"@S")==0){
				flag = true;
			}
    		}
    		myfile.close();
		//reads = &tot_line[0];
  	}
  	else cout << "Unable to open file"; 
	
}
void convCharToBin(const char*array, std::vector<uint32_t> &readsBin){
    int length = strlen(array);
    uint32_t conv = 0;
    int mult = length/16;
    int rest =  length%16;
    for(int i = 0; i< mult; i++){
        for(int j = 0; j<16;j++){
            conv = conv<<2;
            switch(array[i*16+j]){
                case 'A':
                    conv = conv | 0x00;
                    break;
                case 'C':
                    conv = conv | 0x01;
                    break;
                case 'G':
                    conv = conv | 0x02;
                    break;
                case 'T':
                    conv = conv | 0x03;
                    break;
            }
        }
        readsBin.push_back(conv);
        conv = 0;
    }

    for(int j = 0; j<rest;j++){
        conv = conv<<2;
        switch(array[16*mult+j]){
            case 'A':
                conv = conv | 0x00;
                break;
            case 'C':
                conv = conv | 0x01;
                break;
            case 'G':
                conv = conv | 0x02;
                break;
            case 'T':
                conv = conv | 0x03;
                break;
        }
    }

    for(int j = 0;j<16-rest;j++){
        conv = conv<<2;
    }
    readsBin.push_back(conv);
}


void printBin(char*toPrint){
        unsigned char *b = (unsigned char*) toPrint;
        unsigned char byte;
        int i, j;

        for (i=sizeof(uint32_t)-1;i>=0;i--)
        {
                for (j=7;j>=0;j--)
                {
                byte = (b[i] >> j) & 1;
                 printf("%u", byte);
                }
        }
        puts("");
}

void BloomFilterFunc(uint64_t totkmers,uint64_t *h_kmers){
    assert (N_LONGS == 4);

    cout<<" Copying kmers to gpu...\n";

    uint64_t *d_kmers = NULL;
    cudaMalloc((void**) &d_kmers, sizeof(*d_kmers)* totkmers * N_LONGS);
    
    cudaMemcpy(d_kmers, h_kmers, sizeof(*d_kmers) * totkmers * N_LONGS, cudaMemcpyHostToDevice);
    
    //definition of the bloom filter
    typedef nvbio::bloom_filter<5, RSHash<uint64_t *>, ElfHash<uint64_t *>, uint32_t *> bloom_filter_type;
    //bloom filter construction
    uint64_t nfilter_elems = totkmers/4;
    
    uint32_t *d_filter_storage = NULL;
        cout<<"finished the bloom filter, starting hashtable cpu..."<<'\n';
    
    cudaMalloc((void **)&d_filter_storage, nfilter_elems * sizeof(*d_filter_storage));
    cudaMemset(d_filter_storage, 0 , nfilter_elems *sizeof(*d_filter_storage));
    
    bloom_filter_type d_filter(nfilter_elems * 32, d_filter_storage);
    
    cout<<"totkmers: "<<totkmers<<'\n';

    cout << "number of bits " << nfilter_elems * 32<< " " << (nfilter_elems * 32) / ((1 << 20) * 8) << " mb " << endl;
    
    uint8_t *d_kmer_pass = NULL;
    uint64_t **d_kmer_ptrs = NULL;
    
    cudaMalloc((void **)&d_kmer_pass, totkmers * sizeof(*d_kmer_pass));
    cudaMemset(d_kmer_pass, 0, totkmers * sizeof(*d_kmer_pass));
    

    int nblocks = (totkmers + 1024)/1024;
    cudaMalloc((void **)&d_kmer_pass,
               totkmers * sizeof(*d_kmer_pass));
    cudaMemset(d_kmer_pass, 0,
               totkmers * sizeof(*d_kmer_pass));
    cudaMalloc((void **)&d_kmer_ptrs,
               totkmers * sizeof(*d_kmer_ptrs));

    
    populate_kernel<<<nblocks,1024>>>(totkmers, d_kmers, d_filter, d_kmer_pass, d_kmer_ptrs);
    
    cudaDeviceSynchronize();
    cudaProfilerStop();

    uint32_t *h_filter_storage = (uint32_t *) malloc(sizeof(*h_filter_storage)*nfilter_elems);
    
    cudaMemcpy(h_filter_storage, d_filter_storage, nfilter_elems * sizeof(*h_filter_storage), cudaMemcpyDeviceToHost);
    
    bloom_filter_type h_filter(nfilter_elems * 32, h_filter_storage);
    
}

dictionary_t accurateCount(double* duration_bella,vector < vector<Kmer> > allkmers){
    dictionary_t countsdenovo;
    cout<<"finished the bloom filter, starting hashtable cpu..."<<'\n';
    auto tbella1 = Clock::now();
    #pragma omp parallel
        {       
            for(auto v:allkmers[MYTHREAD])
            {
                countsdenovo.insert(v, 0);
            }
    }

    auto updatecount = [](int &num) { ++num; };
    #pragma omp parallel
        {          
            for(auto v:allkmers[MYTHREAD])
            {
                // does nothing if the entry doesn't exist in the table
                countsdenovo.update_fn(v,updatecount);
            }
    }
    auto tbella2 = Clock::now();
    *duration_bella = std::chrono::duration_cast<std::chrono::nanoseconds>(tbella2 - tbella1).count();
    *duration_bella = *duration_bella / 1e6;
    return countsdenovo;
}

void testHashThreads(uint32_t* d_key,uint32_t num_vec,int*num_kmers_read,int num_kmers, int num_of_reads){
int post  = 0;
int index_vector = 0;
int index_array = 0;
uint32_t myKey = 0;
int div = 0;
int prev = 0;
int start_index = 0;
int num_char =16;

for(int tid = 0; tid<num_kmers; tid++){
        for(int i=0;i<num_of_reads;i++){
                post += num_kmers_read[i];
		printf("------------------------------------------------------\n");
		printf("prev  : %d  post : %d  tid : %d\n", prev, post, tid);
                if(tid>=prev&&tid<post){
                        //calculate index of the block : tid is the id of the kmer, by subtracting prev it just start from zero
                        index_vector = start_index + (tid-prev)/16;
                        index_array = (tid-prev)%16;

                        myKey = ((d_key[index_vector]<<(2*index_array))|
                                (index_array!=0
                                        ?(d_key[index_vector+1]>>(2*(16-index_array)))
                                        :0x0));
			printf("index_vector : %d  index_array : %d \n",index_vector, index_array);
			for (int k = 31; 0 <= k; k--) {
				printf("%c", (myKey & (1 << k)) ? '1' : '0');
                	}
			printf("\n");
//                        myBucket = slab_hash.computeBucket(myKey);
//                        to_insert = 1;

                        break;
                }
        //calculate where the next block starts
                div = num_kmers_read[i]+num_char-1;
                start_index += div/num_char + (div % num_char != 0);
                prev+=num_kmers_read[i];
        }
        start_index = 0;
        prev = 0;
        post = 0;
}
}

std::vector<uint32_t> HashTableGPU(std::vector<uint32_t>& h_kmers_to_insert,int kmer_len,int num_kmers,vector<int> & h_num_kmers_read){
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
    	std::vector<ValueT> h_result(num_kmers);
	//printf("kmer_len: %d,  num_kmers : %d  \n",kmer_len , num_kmers);
//	testHashThreads(h_kmers_to_insert.data(), h_kmers_to_insert.size(), h_num_kmers_read.data(), num_kmers, h_num_kmers_read.size());

	gpu_hash_table<KeyT, ValueT, SlabHashTypeT::ConcurrentMap>
        	hash_table(num_kmers,h_num_kmers_read.size(),h_kmers_to_insert.size(), num_buckets, 0, 1);
		
        float insert_time =
                hash_table.hash_insert(h_kmers_to_insert.data(), h_kmers_to_insert.size(),h_num_kmers_read.data(), num_kmers);
        float search_time =
                hash_table.hash_search(h_kmers_to_insert.data(), h_result.data(), h_kmers_to_insert.size(),h_num_kmers_read.data(), num_kmers);
	
	printf("insert time : %f, search time : %f\n",insert_time,search_time);
	return h_result;
}

/**
 * @brief DeNovoCount
 * @param allfiles
 * @param countsreliable_denovo
 * @param lower
 * @param upper
 * @param kmer_len
 * @param upperlimit
 */
void
DeNovoCount_new(vector<filedata> & allfiles,
			dictionary_t & countsreliable_denovo,
			int & lower,
			int & upper,
			int kmer_len,
			int depth,
			double & erate,
			size_t upperlimit /* memory limit */,
			BELLApars & b_parameters)
{
    vector < vector<Kmer> > allkmers(MAXTHREADS);
    vector < vector<double> > allquals(MAXTHREADS);
    vector < HyperLogLog > hlls(MAXTHREADS, HyperLogLog(12));   // std::vector fill constructor
	kmer_len =16;
    double denovocount = omp_get_wtime();
    double cardinality;
    size_t totreads = 0;
    vector<vector<std::string>> sequenze(MAXTHREADS);
    for(auto itr=allfiles.begin(); itr!=allfiles.end(); itr++) 
    {
        #pragma omp parallel
        {
            ParallelFASTQ *pfq = new ParallelFASTQ();
            pfq->open(itr->filename, false, itr->filesize);

            vector<string> seqs;
            vector<string> quals;
            vector<string> nametags;
            size_t tlreads = 0; // thread local reads

            size_t fillstatus = 1;
            while(fillstatus) 
            { 
                fillstatus = pfq->fill_block(nametags, seqs, quals, upperlimit);
                size_t nreads = seqs.size();

                for(int i=0; i<nreads; i++) 
                {
                    // remember that the last valid position is length()-1
                    int len = seqs[i].length();
                    double rerror = 0.0;
		    std::string re = seqs[i];
	            char rea[re.length()];
		    
		    strcpy(rea,re.c_str());
		     sequenze[MYTHREAD].push_back(rea);
			
                    for(int j=0; j<=len-kmer_len; j++)  
                    {
                        std::string kmerstrfromfastq = seqs[i].substr(j, kmer_len);
                        Kmer mykmer(kmerstrfromfastq.c_str());
                        Kmer lexsmall = mykmer.rep();
                        allkmers[MYTHREAD].push_back(mykmer);
                        hlls[MYTHREAD].add((const char*) mykmer.getBytes(), mykmer.getNumBytes());

            		if(b_parameters.skipEstimate == false)
            		{
                        	// accuracy
                       		int bqual = (int)quals[i][j] - ASCIIBASE;
                        	double berror = pow(10,-(double)bqual/10);
                        	rerror += berror;
            		}

                    }
		    if(b_parameters.skipEstimate == false)
		    {
                    	// remaining k qual position accuracy
                    	for(int j=len-kmer_len+1; j < len; j++)
                    	{
                        	int bqual = (int)quals[i][j] - ASCIIBASE;
                        	double berror = pow(10,-(double)bqual/10);
                        	rerror += berror;
                    	}
                    	rerror = rerror / len;
                    	allquals[MYTHREAD].push_back(rerror);
		    }
                } // for(int i=0; i<nreads; i++)
                tlreads += nreads;
            } //while(fillstatus) 
            delete pfq;

            #pragma omp critical
            totreads += tlreads;
        }
    }

    // Error estimation
    if(b_parameters.skipEstimate == false)
    {
        erate = 0.0; // reset to 0 here, otherwise it cointains default or user-defined values
        #pragma omp for reduction(+:erate)
        for (int i = 0; i < MAXTHREADS; i++) 
            {
                double temp = std::accumulate(allquals[i].begin(),allquals[i].end(), 0.0);
                erate += temp/(double)allquals[i].size();
            }
        erate = erate / (double)MAXTHREADS;
    }

    // HLL reduction (serial for now) to avoid double iteration
    for (int i = 1; i < MAXTHREADS; i++) 
    {
        std::transform(hlls[0].M.begin(), hlls[0].M.end(), hlls[i].M.begin(), hlls[0].M.begin(), [](uint8_t c1, uint8_t c2) -> uint8_t{ return std::max(c1, c2); });
    }
    cardinality = hlls[0].estimate();

    double load2kmers = omp_get_wtime(); 
    cout << "Initial parsing, error estimation, and k-mer loading took: " << load2kmers - denovocount << "s\n" << endl;


////////////////////////////////////////////////////////////////////////////


	uint64_t totkmers = 0;
	for(int i = 0 ; i < MAXTHREADS; ++i){
		totkmers += allkmers[i].size();
	}
	
    uint64_t *h_kmers = (uint64_t *) malloc(sizeof(*h_kmers) * totkmers * N_LONGS);
    
    uint64_t tmp = 0;
    for(int i = 0; i< MAXTHREADS; i++){
        for( int j = 0; j< allkmers[i].size();++j){
            h_kmers[tmp++] = allkmers[i][j].getArray()[0];
            h_kmers[tmp++] = allkmers[i][j].getArray()[1];
            h_kmers[tmp++] = allkmers[i][j].getArray()[2];
            h_kmers[tmp++] = allkmers[i][j].getArray()[3];
        }

    }
    //BloomFilterFunc(totkmers,h_kmers);

	//allocating and transfering kmers from the host to the device


    	double duration_bella;
    	dictionary_t countsdenovo;
    	countsdenovo =  accurateCount(&duration_bella,allkmers);

	std::string h_kmer;
	int count = 0;
	int singleton = 0;
	int tot = 0;
	int errors = 0;
	printf("compressing... \n");	
///////////////////////////////
	vector<uint32_t> h_reads;
	vector<int> h_num_of_kmers_read;
        uint32_t conv;
	int kmer_size = 16;
	std::string reads;
	char* chunk = new char[kmer_size];
	int num_kmers=0;
        for(int i = 0; i<sequenze.size();i++){
                //printf("i :%d\n",i);
                for(int j = 0; j<sequenze[i].size();j++){
			const char *read = sequenze[i][j].c_str();
			convCharToBin(read,h_reads);
			//printf("read number : %d\n",j);
			//printf("%s\n",read);
			h_num_of_kmers_read.push_back(strlen(read)-kmer_size+1);
			num_kmers+=(strlen(read)-kmer_size+1);
                }
	}
	
	
///////////////////////////////

	
	//HASHTABLE
        //building hash table
	int bella_one =0 ;
	int bella_two = 0;
	int bella_three = 0;
	vector<int> vals;
	/*std::vector<uint32_t> h_query;
	 for (int k = 0; k < totkmers; ++k)
	 {
	 	int kmerid = k;
	 	{
	 		uint64_t *kmerptr = &(h_kmers[kmerid * N_LONGS]);
		
	 		size_t i,j,l;
	 		char *sx = (char *) malloc(1024);
	 		char *s = sx;
	 		memset(s, '\0', 1024);
	 	//	printf("kmerid %d\n", kmerid);
	 		for (i = 0; i < Kmer::k; i++)
	 		{
	 			j = i % 32;
	 			l = i / 32;

	 			switch(((kmerptr[l]) >> (2*(31-j)) ) & 0x03)
	 			{
	 			case 0x00: *s = 'A'; ++s; break;
	 			case 0x01: *s = 'C'; ++s; break;
	 			case 0x02: *s = 'G'; ++s; break;
	 			case 0x03: *s = 'T'; ++s; break;
	 			}
	 		}
			h_kmer = sx;
			uint32_t conv;
			//ToKmerBin(&conv, sx);
			h_query.push_back(conv);
                        Kmer mykmer(h_kmer.c_str());
                        //Kmer lexsmall = mykmer.rep();
 
			int val=0;
                        val = countsdenovo.find(mykmer);
       		//	printf("val : %d", val);                
	 	//	bool b = h_filter.has(&(h_kmers[kmerid*N_LONGS]));
		  //     	printf("%d\n",val);
			vals.push_back(val);
			if(val == 3){
				bella_three++;	
			}
			if (val == 2){
				bella_two++;
			}	
			if(val == 1){
				bella_one++;
			}	
		//	if(!b){
		//		errors++;
		//	}
		//	if(b){
		//		count++;
		//		tot += val;
		//	}
			
	 	}	
	 }*//*
	for(int i = 0; i<h_query.size();i++){
		printf("index : %d h_query : %"PRIu32"\n",i,h_query[i]);
	}
*/
	//testing the conversion
        uint32_t myKey;
        int index_vector = 0;
        int index_array = 0;
	uint32_t first_piece=0;
	uint32_t second_piece=0;

	printf("starting gpu hash");
	printf("\n\n\n\n");

	auto tgpu1 = Clock::now();
	vector<uint32_t> h_result;
	h_result = HashTableGPU(h_reads,16,num_kmers,h_num_of_kmers_read);
	auto tgpu2 = Clock::now();	
	int three = 0;
	int two = 0;
	int one = 0;
/*	for(int i=0; i<num_kmers;i++){
	//	if(h_result[i]==-1){
	//		printf("error\n");
	//	}	
	//	cout<<h_result[i]<<'\n';
		if(h_result[i] == 3){
			three++;
		}
		if(h_result[i] == 2){
			two++;
		}
		if(h_result[i] == 1){
			one++;
		}
	}*/
	printf("h_result size : %d, totkmers : %d \n",h_result.size(),totkmers);
	printf("one : %d,bella_one: %d, two: %d,bella_two : %d,three : %d, bella_three: %d \n"
	,one,bella_one, two,bella_two, three,bella_three);
	double duration_gpu = std::chrono::duration_cast<std::chrono::nanoseconds>(tgpu2 - tgpu1).count();
        duration_gpu = duration_gpu / 1e6;

	printf("totkmers: %d, tgpu: %.2f tbella: %.2f \n", totkmers,duration_gpu,duration_bella);
        
	/*
	A = 00
	C = 01
	G = 10
	T = 11
	*/
	

	exit(0); 

}

#endif
