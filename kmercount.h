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
//#include "gpu/kmer_count_kernels.h"
//#include "gpu/bloom-gpu/fhash.h"
//#include "gpu/bloom-gpu/nvbio/bloom_filter.h"
//#include "gpu/bloom-gpu/nvbio/types.h"
//#include "gpu/SlabHash/src/slab_hash.cuh"
//#include "gpu/SlabHash/src/gpu_hash_table.cuh"
#include "count.cpp"
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
    vector < vector<uint32_t> > convertedKmers_h_query(MAXTHREADS);
    vector < vector<uint32_t> > convertedKmers_h_index(MAXTHREADS);
    vector < vector<double> > allquals(MAXTHREADS);
    vector < HyperLogLog > hlls(MAXTHREADS, HyperLogLog(12));   // std::vector fill constructor
    double denovocount = omp_get_wtime();
    double cardinality;
    size_t totreads = 0;
    
    uint32_t *h_key_cudaHost, *h_index_cudaHost;
      	
   	
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
	//	    printf("length of the read: %d \n",len);
                    double rerror = 0.0;
		    std::string re = seqs[i];
		    
			
                    for(int j=0; j<=len-kmer_len; j++)  
                    {
                        std::string kmerstrfromfastq = seqs[i].substr(j, kmer_len);
			convCharToBin64(kmerstrfromfastq.c_str(),convertedKmers_h_query[MYTHREAD],convertedKmers_h_index[MYTHREAD],kmer_len);
                        Kmer mykmer(kmerstrfromfastq.c_str());
                        Kmer lexsmall = mykmer.rep();
                        //allkmers[MYTHREAD].push_back(mykmer);
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
        uint32_t conv_table[32]={0x80000000,0xc0000000,0xe0000000,0xf0000000,0xf8000000,0xfc000000,0xfe000000,0xff000000,
				 0xff800000,0xffc00000,0xffe00000,0xfff00000,0xfff80000,0xfffc0000,0xfffe0000,0xffff0000,
				 0xffff8000,0xffffc000,0xffffe000,0xfffff000,0xfffff800,0xfffffc00,0xfffffe00,0xffffff00,
				 0xffffff80,0xffffffc0,0xffffffe0,0xfffffff0,0xfffffff8,0xfffffffc,0xfffffffe,0xffffffff};
        
        

	vector<uint32_t> h_lookup_table;
	uint64_t totkmers = 0;
	uint32_t full = 0xFFFFFFFF;
	for(int i = 0 ; i < MAXTHREADS; ++i){
		int size = convertedKmers_h_query[i].size();
		totkmers += size;
//		for(int j = 0;j<size/32; j++){
//			h_lookup_table.push_back(full);
//		}
//		h_lookup_table.push_back(conv_table[(size%32)-1]);
	}
//	for(int i = 0; i<h_lookup_table.size();i++){
	
//		for (int k = 31; 0 <= k; k--) {
//                	printf("%c", (h_lookup_table[i] & (1 << k)) ? '1' : '0');
//                }
//              	printf("\n");

//	}
	printf("%" PRId64 "\n", totkmers);
	CHECK_CUDA_ERROR(cudaHostAlloc((void**) &h_key_cudaHost,totkmers*sizeof(uint32_t),cudaHostAllocDefault));
	CHECK_CUDA_ERROR(cudaHostAlloc((void**) &h_index_cudaHost,totkmers*sizeof(uint32_t),cudaHostAllocDefault));
	uint64_t t = 0;
	
	#pragma omp for
	for(uint64_t i = 0; i<MAXTHREADS; i++){
		int convSize = convertedKmers_h_query[i].size();
		uint32_t*h_query_list = convertedKmers_h_query[i].data();
		uint32_t*h_index_list = convertedKmers_h_index[i].data();
		for(uint32_t j = 0; j < convSize; j++){
			if(h_query_list[j] != 0xFFFFFFFF){
				h_key_cudaHost[t] = h_query_list[j];
				h_index_cudaHost[t] = h_index_list[j];
				t++;
			}
		}
	}
	
    	//uint64_t *h_kmers = (uint64_t *) malloc(sizeof(*h_kmers) * totkmers * N_LONGS);
   	/* 
    	uint64_t tmp = 0;
    	for(uint32_t i = 0; i< MAXTHREADS; i++){
        	for( int j = 0; j< allkmers[i].size();++j){
            		h_kmers[tmp++] = allkmers[i][j].getArray()[0];
            		h_kmers[tmp++] = allkmers[i][j].getArray()[1];
            		h_kmers[tmp++] = allkmers[i][j].getArray()[2];
	    		h_kmers[tmp++] = allkmers[i][j].getArray()[3];
        	}

    	}
	*/
    	//BloomFilterFunc(totkmers,h_kmers);

	//allocating and transfering kmers from the host to the device


    	double duration_bella;
    	dictionary_t countsdenovo;
    	countsdenovo = accurateCount(&duration_bella,allkmers);
	printf("tbella: %.2f \n", duration_bella);

	printf("conv...\n");
	//HASHTABLE
        //building hash table
	int bella_one =0;
	int bella_two = 0;
	int bella_three = 0;
	string h_kmer;
	vector<int> vals;
	vector<uint32_t> h_query;
	vector<uint32_t> h_index;
	/* for (uint32_t k = 0; k < totkmers; ++k)
	 {
	 	uint32_t kmerid = k;
	 	{
	 		uint64_t *kmerptr = &(h_kmers[kmerid * N_LONGS]);
		
	 		size_t i,j,l;
	 		char *sx = (char *) malloc(1024);
	 		char *s = sx;
	 		memset(s, '\0', 1024);
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
	//		printf("kmer : %s\n",sx);
			convCharToBin64(sx, h_query,h_index,kmer_len);
                        Kmer mykmer(h_kmer.c_str());
                        Kmer lexsmall = mykmer.rep();
			int val=0;
          */         /*     val = countsdenovo.find(mykmer);
			vals.push_back(val);
			if(val == 3){
				bella_three++;	
			}
			if (val == 2){
				bella_two++;
			}	
			if(val == 1){
				bella_one++;
			}*/
	// 	}	
	// }

	printf("starting gpu hash");
	printf("\n\n\n\n");

	auto tgpu1 = Clock::now();
	vector<uint32_t> h_result;
	//h_result = HashTableGPU(h_query,h_index,kmer_len,totkmers);
	h_result = HashTableGPU(h_key_cudaHost,h_index_cudaHost, kmer_len,totkmers);
	auto tgpu2 = Clock::now();	
	int three = 0;
	int two = 0;
	int one = 0;
	exit(0);
	int count_err = 0;
	#pragma omp for
	for(uint64_t i=0; i<totkmers;i++){
		if(h_result[i]==(-1)){
			//printf("error\n");
			count_err++;
		}
		if(h_result[i] == 3){
			three++;
		}
		if(h_result[i] == 2){
			two++;
		}
		if(h_result[i] == 1){
			one++;
		}
	}
	printf("h_result size : %d, totkmers : %d \n",h_result.size(),totkmers);
	printf("one : %d,bella_one: %d, two: %d,bella_two : %d,three : %d, bella_three: %d \n"
	,one,bella_one, two,bella_two, three,bella_three);
	double duration_gpu = std::chrono::duration_cast<std::chrono::nanoseconds>(tgpu2 - tgpu1).count();
        duration_gpu = duration_gpu / 1e6;
	printf("count err %d\n",count_err);
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
