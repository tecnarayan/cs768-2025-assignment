## run citation_graph_genetator.py

output : 

tec@tec-82FG:~/Documents/Sem_6/Learning_With_Graphs/cs768-2025-assignment/cs768-2025-assignment$ python citation_graph_generator.py 
Loading paper data from: ./dataset_papers...
Loaded data for 6545 papers in 14.31 seconds.
Normalizing titles and building lookup table...
Built lookup table with 6545 unique normalized titles in 0.07 seconds.
Building citation graph...
Built full graph with 6545 nodes and 30762 edges in 10.94 seconds.

--- Full Graph Analysis ---
1. Number of edges: 30762
2. Number of isolated nodes: 441
3. Average In-Degree: 4.7001
   Average Out-Degree: 4.7001
   Average Total Degree: 9.4002
   Median Total Degree: 6
   Generating full degree histogram...
   Full degree histogram saved to: ./plots/degree_histogram.png
4. Diameter Calculation (Full Graph):
   Finding largest weakly connected component...
   Largest WCC size: 6055 nodes
   Calculating diameter of the largest WCC (treating as undirected)...
   Diameter of Largest Weakly Connected Component: 13

Saving full graph to citation_graph.adjlist...
Full graph saved successfully in 28.50 seconds.

--- Filtered Graph Analysis (Removing Bottom/Top 5% Degree Nodes) ---
   Degree Thresholds: Keep nodes with total degree >= 0.0 and <= 28.0
   Keeping 6232 of 6545 nodes.
   Filtered graph has 6232 nodes and 14013 edges.
   Filtered Average In-Degree: 2.2486
   Filtered Average Out-Degree: 2.2486
   Filtered Average Total Degree: 4.4971
   Filtered Median Total Degree: 3.0
   Generating filtered degree histogram...
   Filtered degree histogram saved to: ./plots/degree_histogram_filtered.png

Total execution time: 54.12 seconds.
tec@tec-82FG:~/Documents/Sem_6/Learning_With_Graphs/cs768-2025-assignment/cs768-2025-assignment$ 




#### then run precompute_data.py

Outputs are -->

# tfidf_vectorizer.pkl --> learned vocablaty and idf scofres .. baildy for each word .. it has that ifd .. which will be sued to caculatecd fpr vector .. 
# dataset_tfidf_vectors.npz  --> precconcomed vecctor for all id .. givn .. to avoid redundecny during test time 
# paper_id_order.txt --> order of id ... of preprocessed data ...

tec@tec-82FG:~/Documents/Sem_6/Learning_With_Graphs/cs768-2025-assignment/cs768-2025-assignment$ python precompute_data.py 
Loading paper text data from: ./dataset_papers...
Loaded text data for 6545 papers in 0.20 seconds.
Fitting TF-IDF Vectorizer (Unigrams) on combined title+abstract...
Vocabulary size: 24831
TF-IDF matrix shape: (6545, 24831)
TF-IDF fitting completed in 0.95 seconds.
Saving fitted vectorizer to tfidf_vectorizer.pkl...
Saving TF-IDF matrix to dataset_tfidf_vectors.npz...
Saving paper ID order to paper_id_order.txt...
Offline data saved successfully in 0.28 seconds.

Precomputation finished. Total time: 1.42 seconds.
tec@tec-82FG:~/Documents/Sem_6/Learning_With_Graphs/cs768-2025-assignment/cs768-2025-assignment$ 





##### Run in_degree.py
--> it gives in_degree of all nodes .. bacically in how many it got cited ... and 




##### then run simlalirty

tec@tec-82FG:~/Documents/Sem_6/Learning_With_Graphs/cs768-2025-assignment/cs768-2025-assignment$ python analysis/similarity_score.py 
Loading paper text data from: ./dataset_papers...
Loaded text data for 6545 papers in 1.04 seconds.
Loading citation graph from: ./citation_graph.adjlist...
Loaded graph with 6545 nodes and 30762 edges.
Analyzing 6545 nodes present in both graph and text data.
Fitting TF-IDF Vectorizer (Unigrams) on combined title+abstract...
Vocabulary size (unique words from titles+abstracts): 24831
TF-IDF fitting completed in 1.04 seconds.
Calculating cosine similarities between citing papers and their neighbors...
Similarity calculation completed in 2.39 seconds.

--- Neighbor Similarity Analysis ---
Calculated 30762 similarity scores between citing papers and cited neighbors.
Mean Similarity: 0.1046
Median Similarity: 0.0781
Mode Similarity (rounded to 0.01): 0.02
Generating similarity histogram...
Similarity histogram saved to: ./plots/neighbor_similarity_histogram.png

Total execution time: 4.96 seconds.
tec@tec-82FG:~/Documents/Sem_6/Learning_With_Graphs/cs768-2025-assignment/cs768-2025-assignment$ 