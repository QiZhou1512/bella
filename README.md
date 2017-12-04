# README #

Repository containing the source code of the k-mers analysis for long read assembly.

To compile: g++ -std=c++0x trie-tree-implementation.cpp -o trie
To run: ./trie mer_counts_dumps.fa (generated using jellyfish dump) my_genome.fna kmer_length (e.g. 15)

To compile the main code occurrence-matrix.cpp: make parse
To run: ./parse kmers-table.fa listoffastq.txt
In the folder the .axt has to be included.

In multiply.h: remember to define which OS you're using (OSX or LINUX) in order to get the memory usage.
