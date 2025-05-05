import os
import re
import argparse
import json
import time
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import statistics # For mean, median, mode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

known_joins = { # BY GPT
    # Core ML Areas
    'deeplearning': 'deep learning',
    'machinelearning': 'machine learning',
    'reinforcementlearning': 'reinforcement learning',
    'supervisedlearning': 'supervised learning',
    'unsupervisedlearning': 'unsupervised learning',
    'semisupervised': 'semi supervised', # Common prefix issue
    'selfsupervised': 'self supervised', # Common prefix issue
    'neuralnetwork': 'neural network',
    'neuralnet': 'neural network', # Common alternative
    'convolutionalneuralnetwork': 'convolutional neural network',
    # 'cnn': 'convolutional neural network', # Be CAREFUL with acronyms - only if frequently used without definition
    'recurrentneuralnetwork': 'recurrent neural network',
    # 'rnn': 'recurrent neural network', # Careful with acronyms
    'generativeadversarialnetwork': 'generative adversarial network',
    # 'gan': 'generative adversarial network', # Careful with acronyms
    'supportvectormachine': 'support vector machine',
    # 'svm': 'support vector machine', # Careful with acronyms
    'transformer': 'transformer', # Usually okay, but check for 'trans former' maybe? Less likely.
    'gradientdescent': 'gradient descent',
    'stochasticgradientdescent': 'stochastic gradient descent',
    'backpropagation': 'back propagation',
    'principalcomponentanalysis': 'principal component analysis',
    'naturallanguageprocessing': 'natural language processing',
    'computervision': 'computer vision',
    'transferlearning': 'transfer learning',
    'metalearning': 'meta learning',
    'fewshot': 'few shot',
    'zeroshot': 'zero shot',
    'dimensionalityreduction': 'dimensionality reduction',
    'featureengineering': 'feature engineering',
    'featureselection': 'feature selection',
    'crossvalidation': 'cross validation',
    'stateoftheart': 'state of the art', 
    'endtoend': 'end to end',
    'multitask': 'multi task',
    'multilabel': 'multi label',
    'multimodal': 'multi modal',
    'multiagent': 'multi agent',
    'longshortterm': 'long short term', 
}

def normalize_text(text):

    if not text or isinstance(text, (int, float)):
        return ""
    text = str(text).lower() 

    for joined, spaced in known_joins.items():
        text = text.replace(joined, spaced)

    text = re.sub(r'(?<=\w)-(?=\w)', ' ', text)

    text = re.sub(r'\{.*?\}', '', text)

    text = re.sub(r'[^\w\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()
    return text


def process_paper_folder_for_text(folder_path):
    """Processes a single paper folder to extract ID and combined title+abstract text."""
    if not os.path.isdir(folder_path):
        return None, None
    paper_id = os.path.basename(os.path.normpath(folder_path))
    paper_title = ""
    paper_abstract = ""
    title_file_path = os.path.join(folder_path, 'title.txt')
    try:
        with open(title_file_path, 'r', encoding='utf-8', errors='ignore') as f: paper_title = f.read().strip()
    except: pass
    abstract_file_path = os.path.join(folder_path, 'abstract.txt')
    try:
        with open(abstract_file_path, 'r', encoding='utf-8', errors='ignore') as f: paper_abstract = f.read().strip()
    except: pass

    # Combine title and abstract for TF-IDF representation
    combined_text = f"{paper_title} {paper_abstract}".strip() # Add space between them
    return paper_id, combined_text


def main():
    parser = argparse.ArgumentParser(description="Analyze content similarity between citing papers and their cited neighbors.")
    parser.add_argument("--dataset_path", type=str, default="./dataset_papers", help="Path to the root directory containing paper folders (e.g., ./dataset_papers).")
    parser.add_argument("--graph_file", type=str, default="./citation_graph.adjlist", help="Path to the citation graph file (e.g., citation_graph.adjlist).")
    parser.add_argument("--output_hist_file", type=str, default="./plots/neighbor_similarity_histogram.png", help="File path to save the similarity histogram plot.")

    args = parser.parse_args()
    start_time = time.time()


    print(f"Loading paper text data from: {args.dataset_path}...")
    paper_texts = {}
    paper_ids_from_folders = []
    paper_folders = [d for d in os.listdir(args.dataset_path) if os.path.isdir(os.path.join(args.dataset_path, d))]
    if not paper_folders: print(f"Error: No paper folders found in {args.dataset_path}"); return

    for folder_name in paper_folders:
        folder_path = os.path.join(args.dataset_path, folder_name)
        paper_id, combined_text = process_paper_folder_for_text(folder_path)
        if paper_id:
            paper_ids_from_folders.append(paper_id)
            paper_texts[paper_id] = combined_text # Store combined text

    if not paper_texts: print("Error: No paper text data could be loaded."); return
    load_text_time = time.time()
    print(f"Loaded text data for {len(paper_texts)} papers in {load_text_time - start_time:.2f} seconds.")

    # 2. Load the Citation Graph
    print(f"Loading citation graph from: {args.graph_file}...")
    try:
        G = nx.read_adjlist(args.graph_file, create_using=nx.DiGraph())
        print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    except FileNotFoundError:
        print(f"Error: Graph file not found at {args.graph_file}")
        return
    except Exception as e:
        print(f"Error loading graph: {e}")
        return

    # 3. Prepare for TF-IDF (using only nodes present in both graph and text data)
    nodes_in_graph = set(G.nodes())
    nodes_with_text = set(paper_texts.keys())
    valid_nodes = list(nodes_in_graph.intersection(nodes_with_text))
    print(f"Analyzing {len(valid_nodes)} nodes present in both graph and text data.")

    if not valid_nodes:
        print("Error: No common nodes between graph and loaded text data.")
        return

    # Get the text for valid nodes in a consistent order
    valid_texts = [paper_texts[node_id] for node_id in valid_nodes]
    # Map node ID to its index in the valid_texts list / TF-IDF matrix row
    node_id_to_tfidf_idx = {node_id: i for i, node_id in enumerate(valid_nodes)}

    # 4. Fit TF-IDF Vectorizer (Unigrams only)
    print("Fitting TF-IDF Vectorizer (Unigrams) on combined title+abstract...")
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 1), # Unigrams only
        preprocessor=normalize_text # Use the consistent normalizer
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(valid_texts)
        vocabulary = vectorizer.get_feature_names_out()
        vocab_size = len(vocabulary)
        print(f"Vocabulary size (unique words from titles+abstracts): {vocab_size}")
        # Ensure matrix shape matches number of valid nodes
        if tfidf_matrix.shape[0] != len(valid_nodes):
             print(f"Warning: TF-IDF matrix rows ({tfidf_matrix.shape[0]}) != valid nodes ({len(valid_nodes)})")
             # This shouldn't happen with default TF-IDF behavior but good to check
    except Exception as e:
        print(f"Error during TF-IDF vectorization: {e}")
        return

    fit_tfidf_time = time.time()
    print(f"TF-IDF fitting completed in {fit_tfidf_time - load_text_time:.2f} seconds.")


    # 5. Calculate Neighbor Similarities
    print("Calculating cosine similarities between citing papers and their neighbors...")
    all_neighbor_similarities = []
    calculation_errors = 0

    for source_node_id in valid_nodes: # Iterate only over nodes we have text for
        # Check if node is in graph (should be, but safe check) and has neighbors
        if source_node_id not in G: continue
        neighbors = list(G.successors(source_node_id)) # successors() gives nodes pointed to
        if not neighbors: continue

        try:
            source_idx = node_id_to_tfidf_idx[source_node_id]
            source_vector = tfidf_matrix[source_idx]

            # Get indices and vectors for valid neighbors (those also in valid_nodes)
            neighbor_indices = [node_id_to_tfidf_idx[n_id] for n_id in neighbors if n_id in node_id_to_tfidf_idx]
            if not neighbor_indices: continue # Skip if no valid neighbors have text data

            neighbor_vectors = tfidf_matrix[neighbor_indices]

            # Calculate similarities between the source vector and all its valid neighbor vectors
            # Result shape: (1, num_valid_neighbors)
            similarity_scores = cosine_similarity(source_vector, neighbor_vectors)

            # Add scores to the main list
            all_neighbor_similarities.extend(similarity_scores[0]) # similarity_scores[0] is the actual list of scores

        except KeyError as e:
            calculation_errors += 1
        except Exception as e:
            print(f"Error calculating similarity for node {source_node_id}: {e}")
            calculation_errors += 1

    calc_sim_time = time.time()
    print(f"Similarity calculation completed in {calc_sim_time - fit_tfidf_time:.2f} seconds.")
    if calculation_errors > 0:
        print(f"Encountered {calculation_errors} errors during similarity calculation (e.g., missing node data).")


    # 6. Analyze and Plot Similarity Scores
    print("\n--- Neighbor Similarity Analysis ---")
    num_similarities = len(all_neighbor_similarities)
    print(f"Calculated {num_similarities} similarity scores between citing papers and cited neighbors.")

    if num_similarities > 0:
        # Calculate Stats
        mean_sim = statistics.mean(all_neighbor_similarities)
        median_sim = statistics.median(all_neighbor_similarities)
        # Mode for (near-)continuous data is tricky. Calculate mode of rounded values.
        try:
            rounded_sims = [round(s, 2) for s in all_neighbor_similarities] # Round to 2 decimal places
            mode_sim = statistics.mode(rounded_sims)
            print(f"Mean Similarity: {mean_sim:.4f}")
            print(f"Median Similarity: {median_sim:.4f}")
            print(f"Mode Similarity (rounded to 0.01): {mode_sim:.2f}")
        except statistics.StatisticsError: # Handle case with no unique mode
            print(f"Mean Similarity: {mean_sim:.4f}")
            print(f"Median Similarity: {median_sim:.4f}")
            print("Mode Similarity: No unique mode found.")
        except Exception as e:
            print(f"Error calculating statistics: {e}")


        # Plot Histogram
        print(f"Generating similarity histogram...")
        plt.figure(figsize=(12, 7))
        # Define bins from 0 to 1 with step 0.05
        bins = np.arange(0, 1.05, 0.05)
        plt.hist(all_neighbor_similarities, bins=bins, alpha=0.75, edgecolor='black', label='Similarity Count')
        plt.title('Distribution of Content Similarity (TF-IDF Cosine) Between Citing Papers and Cited Neighbors')
        plt.xlabel('Cosine Similarity Score')
        plt.ylabel('Number of Citation Pairs')

        # Add vertical lines for mean and median
        plt.axvline(mean_sim, color='r', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_sim:.2f}')
        plt.axvline(median_sim, color='g', linestyle='dotted', linewidth=1.5, label=f'Median: {median_sim:.2f}')

        plt.legend()
        plt.grid(axis='y', alpha=0.5)
        plt.xticks(bins, rotation=45) # Show ticks for each bin edge
        plt.xlim(0, 1) # Ensure x-axis is strictly 0 to 1

        try:
            plt.tight_layout() # Adjust layout to prevent labels overlapping
            plt.savefig(args.output_hist_file)
            print(f"Similarity histogram saved to: {args.output_hist_file}")
        except Exception as e:
            print(f"   Error saving similarity histogram: {e}")
        plt.close()

    else:
        print("No neighbor similarity scores were calculated (perhaps no edges in graph or data issues).")

    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
