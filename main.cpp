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

#include "libcuckoo/cuckoohash_map.hh"
#include "kmercount.h"

#include "kmercode/hash_funcs.h"
#include "kmercode/Kmer.hpp"
#include "kmercode/Buffer.h"
#include "kmercode/common.h"
#include "kmercode/fq_reader.h"
#include "kmercode/ParallelFASTQ.h"
#include "kmercode/bound.hpp"

#include "mtspgemm2017/utility.h"
#include "mtspgemm2017/CSC.h"
#include "mtspgemm2017/CSR.h"
#include "mtspgemm2017/global.h"
#include "mtspgemm2017/IO.h"
#include "mtspgemm2017/overlapping.h"
#include "mtspgemm2017/align.h"

#define LSIZE 16000
#define ITERS 10
#define PRINT
//#define JELLYFISH

double deltaChernoff = 0.1;            

using namespace std;

int main (int argc, char *argv[]) {

    //
    // Program name and purpose
    //
    cout << "\nBELLA - Long Read Aligner for De Novo Genome Assembly\n" << endl;
    //
    // Setup the input files
    //
    option_t *optList, *thisOpt;
    // Get list of command line options and their arguments 
    // Follow an option with a colon to indicate that it requires an argument.

    optList = NULL;
    optList = GetOptList(argc, argv, (char*)"f:i:o:d:hk:a:ze:p:w:vxc:m:");
   

    char *kmer_file = NULL;                 // Reliable k-mer file from Jellyfish
    char *all_inputs_fofn = NULL;           // List of fastqs (i)
    char *out_file = NULL;                  // output filename (o)
    int kmer_len = 17;                      // default k-mer length (k)
    int xdrop = 3;                          // default alignment x-drop factor (p)
    double erate = 0.15;                    // default error rate (e) 
    int depth = 0;                          // depth/coverage required (d)

    BELLApars b_parameters;

    if(optList == NULL)
    {
        cout << "BELLA execution terminated: not enough parameters or invalid option" << endl;
        cout << "Run with -h to print out the command line options\n" << endl;
        return 0;
    }

    while (optList!=NULL) {
        thisOpt = optList;
        optList = optList->next;
        switch (thisOpt->option) {
            case 'f': {
#ifdef JELLYFISH
                if(thisOpt->argument == NULL)
                {
                    cout << "BELLA execution terminated: -f requires an argument" << endl;
                    cout << "Run with -h to print out the command line options\n" << endl;
                    return 0;
                }
#endif
                kmer_file = strdup(thisOpt->argument);
                break;
            }
            case 'i': {
                if(thisOpt->argument == NULL)
                {
                    cout << "BELLA execution terminated: -i requires an argument" << endl;
                    cout << "Run with -h to print out the command line options\n" << endl;
                    return 0;
                }
                all_inputs_fofn = strdup(thisOpt->argument);
                break;
            }
            case 'o': {
                if(thisOpt->argument == NULL)
                {
                    cout << "BELLA execution terminated: -o requires an argument" << endl;
                    cout << "Run with -h to print out the command line options\n" << endl;
                    return 0;
                }
                char* line1 = strdup(thisOpt->argument);
                char* line2 = strdup(".out");
                size_t len1 = strlen(line1);
                size_t len2 = strlen(line2);

                out_file = (char*)malloc(len1 + len2 + 1);
                if (!out_file) abort();

                memcpy(out_file, line1, len1);
                memcpy(out_file + len1, line2, len2);
                out_file[len1 + len2] = '\0';

                delete line1;
                delete line2;

                break;
            }
            case 'd': {
                if(thisOpt->argument == NULL)
                {
                    cout << "BELLA execution terminated: -d requires an argument" << endl;
                    cout << "Run with -h to print out the command line options\n" << endl;
                    return 0;
                }
                depth = atoi(thisOpt->argument);  
                break;
            }
            case 'z': b_parameters.skipAlignment = true; break; 
            case 'v': {
                b_parameters.adapThr = true;
                xdrop = 7;
                break;
            }
            case 'x': b_parameters.alignEnd = true; break; 
            case 'k': {
                kmer_len = atoi(thisOpt->argument);
                break;
            }
            case 'e': {
                erate = strtod(thisOpt->argument, NULL);
                break;
            }
            case 'a': {
                b_parameters.defaultThr = atoi(thisOpt->argument);
                break;
            }
            case 'p': {
                xdrop = atoi(thisOpt->argument);
                break;
            } 
            case 'w': {
                b_parameters.relaxMargin = atoi(thisOpt->argument);
                break;
            }
        case 'm': {
        b_parameters.totalMemory = stod(thisOpt->argument);
        b_parameters.userDefMem = true;
        cout << "User defined memory set to " << b_parameters.totalMemory << " MB " << endl;
        break;
        }
            case 'c': {
                if(stod(thisOpt->argument) > 1.0 || stod(thisOpt->argument) < 0.0)
                {
                    cout << "BELLA execution terminated: -c requires a value in [0,1]" << endl;
                    cout << "Run with -h to print out the command line options\n" << endl;
                    return 0;
                }
                b_parameters.deltaChernoff = stod(thisOpt->argument);
                break;
            }
            case 'h': {
                cout << "Usage:\n" << endl;
                cout << " -f : k-mer list from Jellyfish (required if Jellyfish k-mer counting is used)" << endl; // Reliable k-mers are selected by BELLA
                cout << " -i : list of fastq(s) (required)" << endl;
                cout << " -o : output filename (required)" << endl;
                cout << " -d : depth/coverage (required)" << endl; // TO DO: add depth estimation
                cout << " -k : k-mer length [17]" << endl;
                cout << " -a : alignment score threshold [50]" << endl;
                cout << " -p : alignment x-drop factor [3]" << endl;
                cout << " -e : error rate [auto estimated from fastq]" << endl;
                cout << " -m : total RAM of the system in MB [auto estimated if possible or 8,000 if not]" << endl;
                cout << " -z : skip the pairwise alignment [false]" << endl;
                cout << " -w : relaxMargin parameter for alignment on edges [300]" << endl;
                cout << " -c : alignment score deviation from the mean [0.1]" << endl;
                cout << " -v : use adaptive alignment threshold [false]" << endl;
                cout << " -x : filter out alignment on edge [false]\n" << endl;

                FreeOptList(thisOpt); // Done with this list, free it
                return 0;
            }
        }
    }

#ifdef JELLYFISH
    if(kmer_file == NULL || all_inputs_fofn == NULL || out_file == NULL || depth == 0)
    {
        cout << "BELLA execution terminated: missing arguments" << endl;
        cout << "Run with -h to print out the command line options\n" << endl;
        return 0;
    }
#else
    if(all_inputs_fofn == NULL || out_file == NULL || depth == 0)
    {
        cout << "BELLA execution terminated: missing arguments" << endl;
        cout << "Run with -h to print out the command line options\n" << endl;
        return 0;
    }
#endif

    free(optList);
    free(thisOpt);
    //
    // Declarations 
    //
    vector<filedata> allfiles = GetFiles(all_inputs_fofn);
    FILE *fastafile;
    int lower, upper; // reliable range lower and upper bound
    double ratioPhi;
    char *buffer;
    Kmer::set_k(kmer_len);
    size_t upperlimit = 10000000; // in bytes
    Kmers kmervect;
    vector<string> seqs;
    vector<string> quals;
    vector<string> nametags;
    readVector_ reads;
    Kmers kmersfromreads;
    vector<tuple<size_t,size_t,size_t>> occurrences;
    vector<tuple<size_t,size_t,size_t>> transtuples;
    // 
    // File and setting used
    //
#ifdef PRINT
#ifdef JELLYFISH
    cout << "K-mer counting: Jellyfish" << endl;
    cout << "Input k-mer file: " << kmer_file << endl;
#endif
    cout << "K-mer counting: BELLA" << endl;
    cout << "Output filename: " << out_file << endl;
    cout << "K-mer length: " << kmer_len << endl;
    if(b_parameters.skipAlignment)
        cout << "Compute alignment: False" << endl;
    else cout << "Compute alignment: True" << endl;
    cout << "X-drop: " << xdrop << endl;
    cout << "Depth: " << depth << "X" << endl;
#endif

    //
    // Kmer file parsing, error estimation, reliable bounds computation, and k-mer dictionary creation
    //

    dictionary_t countsreliable;
#ifdef JELLYFISH
    // Reliable bounds computation for Jellyfish using default error rate
    double all = omp_get_wtime();
    lower = computeLower(depth,erate,kmer_len);
    upper = computeUpper(depth,erate,kmer_len);
    cout << "Error rate is " << erate << endl;
    cout << "Reliable lower bound: " << lower << endl;
    cout << "Reliable upper bound: " << upper << endl;
    JellyFishCount(kmer_file, countsreliable, lower, upper);

#else
    // Error estimation and reliabe bounds computation within denovo counting
    cout << "\nRunning with up to " << MAXTHREADS << " threads" << endl;
    double all = omp_get_wtime();
    DeNovoCount(allfiles, countsreliable, lower, upper, kmer_len, depth, erate, upperlimit);

#ifdef PRINT
    cout << "Error rate estimate is " << erate << endl;
    cout << "Reliable lower bound: " << lower << endl;
    cout << "Reliable upper bound: " << upper << endl;
if(b_parameters.adapThr)
{
    ratioPhi = adaptiveSlope(erate, (float)xdrop);
    cout << "Deviation from expected alignment score: " << b_parameters.deltaChernoff << endl;
    cout << "Constant of adaptive threshold: " << ratioPhi*(1-b_parameters.deltaChernoff)<< endl;
}
else cout << "Default alignment score threshold: " << b_parameters.defaultThr << endl;
if(b_parameters.alignEnd)
{
    cout << "Alignment on edge: True" << endl;
    cout << "Relax margin: " << b_parameters.relaxMargin << endl;
} else cout << "Alignment on edge: False" << endl;
#endif // PRINT
#endif // DENOVO COUNTING

    //
    // Fastq(s) parsing
    //

    double parsefastq = omp_get_wtime();
    size_t read_id = 0; // read_id needs to be global (not just per file)
    cout << "\nRunning with up to " << MAXTHREADS << " threads" << endl;

    vector < vector<tuple<int,int,int>> > alloccurrences(MAXTHREADS);   
    vector < vector<tuple<int,int,int>> > alltranstuples(MAXTHREADS);   
    vector < readVector_ > allreads(MAXTHREADS);

    double time1 = omp_get_wtime();
    for(auto itr=allfiles.begin(); itr!=allfiles.end(); itr++)
    {

        ParallelFASTQ *pfq = new ParallelFASTQ();
        pfq->open(itr->filename, false, itr->filesize);

        size_t fillstatus = 1;
        while(fillstatus)
        { 
            fillstatus = pfq->fill_block(nametags, seqs, quals, upperlimit);
            size_t nreads = seqs.size();

            #pragma omp parallel for
            for(int i=0; i<nreads; i++) 
            {
                // remember that the last valid position is length()-1
                int len = seqs[i].length();

                readType_ temp;
                temp.nametag = nametags[i];
                temp.seq = seqs[i]; // save reads for seeded alignment
                temp.readid = read_id+i;

                allreads[MYTHREAD].push_back(temp);
                        
                for(int j=0; j<=len-kmer_len; j++)  
                {
                    std::string kmerstrfromfastq = seqs[i].substr(j, kmer_len);
                    Kmer mykmer(kmerstrfromfastq.c_str());
                    // remember to use only ::rep() when building kmerdict as well
                    Kmer lexsmall = mykmer.rep();

                    int idx; // kmer_id
                    auto found = countsreliable.find(lexsmall,idx);
                    if(found)
                    {
                        alloccurrences[MYTHREAD].emplace_back(std::make_tuple(read_id+i,idx,j)); // vector<tuple<read_id,kmer_id,kmerpos>>
                        alltranstuples[MYTHREAD].emplace_back(std::make_tuple(idx,read_id+i,j)); // transtuples.push_back(col_id,row_id,kmerpos)
                    }
                }
            } // for(int i=0; i<nreads; i++)
            //cout << "total number of reads processed so far is " << read_id << endl;
            read_id += nreads;
        } //while(fillstatus) 
        delete pfq;
    } // for all files

    size_t readcount = 0;
    size_t tuplecount = 0;
    for(int t=0; t<MAXTHREADS; ++t)
    {
        readcount += allreads[t].size();
        tuplecount += alloccurrences[t].size();
    }
    reads.resize(readcount);
    occurrences.resize(tuplecount);
    transtuples.resize(tuplecount);

    size_t readssofar = 0;
    size_t tuplesofar = 0;
    for(int t=0; t<MAXTHREADS; ++t)
    {
        copy(allreads[t].begin(), allreads[t].end(), reads.begin()+readssofar);
        readssofar += allreads[t].size();

        copy(alloccurrences[t].begin(), alloccurrences[t].end(), occurrences.begin() + tuplesofar);
        copy(alltranstuples[t].begin(), alltranstuples[t].end(), transtuples.begin() + tuplesofar);
        tuplesofar += alloccurrences[t].size();
    }

    std::sort(reads.begin(), reads.end());   // bool operator in global.h: sort by readid
    std::vector<string>().swap(seqs);        // free memory of seqs  
    std::vector<string>().swap(quals);       // free memory of quals
    cout << "\nTotal number of reads: "<< read_id << endl;
    cout << "fastq(s) parsing fastq took: " << omp_get_wtime()-parsefastq << "s" << endl;
    double matcreat = omp_get_wtime();

    //
    // Sparse matrices construction
    //

    size_t nkmer = countsreliable.size();
    CSC<size_t,size_t> spmat(occurrences, read_id, nkmer, 
                            [] (size_t & p1, size_t & p2) 
                            {  
                                return p1;
                            });
    std::vector<tuple<size_t,size_t,size_t>>().swap(occurrences); // remove memory of occurences

    CSC<size_t,size_t> transpmat(transtuples, nkmer, read_id, 
                            [] (size_t & p1, size_t & p2) 
                            {  return p1;
                            });
    std::vector<tuple<size_t,size_t,size_t>>().swap(transtuples); // remove memory of transtuples

#ifdef PRINT
    cout << "spmat and spmat^T creation took: " << omp_get_wtime()-matcreat << "s" << endl;
#endif

    //
    // Overlap detection (sparse matrix multiplication) and seed-and-extend alignment
    //

    spmatPtr_ getvaluetype(make_shared<spmatType_>());
    HashSpGEMM(spmat, transpmat, 
            [] (size_t & pi, size_t & pj) // n-th k-mer positions on read i and on read j 
            {   spmatPtr_ value(make_shared<spmatType_>());
                value->count = 1;
                value->pos[0] = pi; // row
                value->pos[1] = pj; // col
                return value;
            }, 
            [] (spmatPtr_ & m1, spmatPtr_ & m2)
            {   m2->count = m1->count+m2->count;
                m2->pos[2] = m1->pos[0]; // row 
                m2->pos[3] = m1->pos[1]; // col
                return m2;
            }, reads, getvaluetype, kmer_len, xdrop, out_file, b_parameters, ratioPhi); 

    cout << "Total running time: " << omp_get_wtime()-all << "s\n" << endl;
    return 0;
} 