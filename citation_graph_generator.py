import os
import re
import argparse
import json
import time
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import statistics 


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

def extract_bbl_citations(file_path):
    citations = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        bib_items = content.split('\\bibitem')
        if len(bib_items) > 1:
            for item in bib_items[1:]:
                newblock_match = re.search(r'\\newblock(.*?)(?=(\\newblock|\Z))', item, re.DOTALL)
                if newblock_match:
                    citation_text = newblock_match.group(1).strip()
                    citation_text = re.sub(r'\s+', ' ', citation_text)
                    citation_text = citation_text.replace('{\\em ', '').replace(' em ', '').replace('}', '')
                    if citation_text:
                         citations.append(citation_text)
    except FileNotFoundError: pass
    except Exception as e: print(f"Error processing BBL {file_path}: {e}")
    return citations

def extract_bib_citations(file_path):
    citations = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        title_matches = re.findall(r'^\s*title\s*=\s*[\{"\']((?:[^{}"\']|\\[{}"\'])*?)\s*[\}"\']\s*,?\s*$',
                                   content, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        for title in title_matches:
            cleaned_title = re.sub(r'\s+', ' ', title.strip())
            cleaned_title = cleaned_title.replace('{', '').replace('}', '')
            if cleaned_title:
                citations.append(cleaned_title)
    except FileNotFoundError: pass
    except Exception as e: print(f"Error processing BIB {file_path}: {e}")
    return citations



def process_paper_folder(folder_path): # given folder path return node .. node has 4 attributes ... id , title , abstract , citations
    if not os.path.isdir(folder_path):
        return None
    paper_id = os.path.basename(os.path.normpath(folder_path))
    paper_title = "[TITLE NOT FOUND]"
    paper_abstract = "[ABSTRACT NOT FOUND]"
    all_citations = []
    title_file_path = os.path.join(folder_path, 'title.txt')
    try:
        with open(title_file_path, 'r', encoding='utf-8', errors='ignore') as f: paper_title = f.read().strip()
    except: pass
    abstract_file_path = os.path.join(folder_path, 'abstract.txt')
    try:
        with open(abstract_file_path, 'r', encoding='utf-8', errors='ignore') as f: paper_abstract = f.read().strip()
    except: pass
    try:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if filename.lower().endswith('.bbl'): all_citations.extend(extract_bbl_citations(file_path))
            elif filename.lower().endswith('.bib'): all_citations.extend(extract_bib_citations(file_path))
    except Exception as e: print(f"Error listing files in {folder_path}: {e}")
    node_data = {"id": paper_id, "title": paper_title, "abstract": paper_abstract, "citations": list(set(all_citations))}
    return node_data


def main():
    parser = argparse.ArgumentParser(description="Build and analyze the citation graph based on exact normalized title matching.")
    parser.add_argument("--dataset_path", type=str,  default="./dataset_papers", help="Path to the root directory containing paper folders (e.g., ./dataset_papers).")
    parser.add_argument("--output_graph_file", type=str, default="citation_graph.adjlist", help="File path to save the full graph (adjacency list format).")
    parser.add_argument("--output_hist_file", type=str, default="./plots/degree_histogram.png", help="File path to save the full degree histogram plot.")
    parser.add_argument("--output_hist_filtered_file", type=str, default="./plots/degree_histogram_filtered.png", help="File path to save the filtered degree histogram plot.")


    args = parser.parse_args()
    start_time = time.time()

    print(f"Loading paper data from: {args.dataset_path}...")
    all_papers_data = []
    paper_folders = [d for d in os.listdir(args.dataset_path) if os.path.isdir(os.path.join(args.dataset_path, d))]
    if not paper_folders: print(f"Error: No paper folders found in {args.dataset_path}"); return
    for folder_name in paper_folders:
        folder_path = os.path.join(args.dataset_path, folder_name)
        paper_info = process_paper_folder(folder_path)
        if paper_info: all_papers_data.append(paper_info)
    if not all_papers_data: print("Error: No paper data could be loaded."); return
    load_time = time.time()
    print(f"Loaded data for {len(all_papers_data)} papers in {load_time - start_time:.2f} seconds.")

    print("Normalizing titles and building lookup table...")
    normalized_title_to_ids = {}
    paper_ids = []
    for paper in all_papers_data:
        paper_ids.append(paper['id'])
        normalized_title = normalize_text(paper['title'])
        if normalized_title:
            if normalized_title not in normalized_title_to_ids: normalized_title_to_ids[normalized_title] = []
            normalized_title_to_ids[normalized_title].append(paper['id'])
    build_lookup_time = time.time()
    print(f"Built lookup table with {len(normalized_title_to_ids)} unique normalized titles in {build_lookup_time - load_time:.2f} seconds.")

    print("Building citation graph...")
    G = nx.DiGraph()
    G.add_nodes_from(paper_ids)
    edge_count = 0
    for source_paper in all_papers_data:
        source_id = source_paper['id']
        citations = source_paper.get('citations', [])
        if not citations: continue
        for citation_string in citations:
            normalized_citation = normalize_text(citation_string)
            if not normalized_citation: continue
            if normalized_citation in normalized_title_to_ids:
                matched_target_ids = normalized_title_to_ids[normalized_citation]
                for target_id in matched_target_ids:
                    if target_id in G and source_id != target_id:
                        if not G.has_edge(source_id, target_id):
                            G.add_edge(source_id, target_id)
                            edge_count +=1
    build_graph_time = time.time()
    print(f"Built full graph with {G.number_of_nodes()} nodes and {edge_count} edges in {build_graph_time - build_lookup_time:.2f} seconds.")
    if edge_count != G.number_of_edges():
        print(f"Warning: Edge count mismatch. Calculated: {edge_count}, NetworkX: {G.number_of_edges()}")
        edge_count = G.number_of_edges()

    print("\n--- Full Graph Analysis ---")
    num_nodes = G.number_of_nodes()
    print(f"1. Number of edges: {edge_count}")
    isolated_nodes = list(nx.isolates(G)); num_isolated = len(isolated_nodes)
    print(f"2. Number of isolated nodes: {num_isolated}")
    full_degrees = [] 
    avg_total_degree = 0
    median_total_degree = 0
    if num_nodes > 0:
        avg_in_degree = sum(d for n, d in G.in_degree()) / num_nodes
        avg_out_degree = sum(d for n, d in G.out_degree()) / num_nodes
        print(f"3. Average In-Degree: {avg_in_degree:.4f}")
        print(f"   Average Out-Degree: {avg_out_degree:.4f}")
        full_degrees = [d for n, d in G.degree()]
        if full_degrees:
             avg_total_degree = statistics.mean(full_degrees)
             median_total_degree = statistics.median(full_degrees)
             print(f"   Average Total Degree: {avg_total_degree:.4f}")
             print(f"   Median Total Degree: {median_total_degree}")
    else: print("3. Average Degree: Graph has no nodes.")
    # Plot Full Degree Histogram
    if num_nodes > 0 and full_degrees:
        print(f"   Generating full degree histogram...")
        max_degree = max(full_degrees)
        plt.figure(figsize=(12, 7)) # Slightly larger figure
        bins = min(50, max_degree + 1)
        plt.hist(full_degrees, bins=bins, alpha=0.75, label='Node Count')
        plt.title('Full Node Degree Distribution (Total Degree)')
        plt.xlabel('Degree')
        plt.ylabel('Number of Nodes')
        # Add vertical lines for mean and median
        plt.axvline(avg_total_degree, color='r', linestyle='dashed', linewidth=1.5, label=f'Mean: {avg_total_degree:.2f}')
        plt.axvline(median_total_degree, color='g', linestyle='dotted', linewidth=1.5, label=f'Median: {median_total_degree}')
        plt.legend() # Show labels for histogram and lines
        plt.grid(axis='y', alpha=0.5)
        plt.text(0.95, 0.95, f'Max Degree: {max_degree}', transform=plt.gca().transAxes, ha='right', va='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
        try:
            plt.savefig(args.output_hist_file)
            print(f"   Full degree histogram saved to: {args.output_hist_file}")
        except Exception as e: print(f"   Error saving full histogram: {e}")
        plt.close()
    else: print("   Skipping full histogram: Graph has no nodes or degrees.")
    # Diameter (Full graph - LWC)
    print("4. Diameter Calculation (Full Graph):")
    if num_nodes > 0 and edge_count > 0:
        print("   Finding largest weakly connected component...")
        try:
            weakly_connected_components = list(nx.weakly_connected_components(G))
            if weakly_connected_components:
                largest_wcc_nodes = max(weakly_connected_components, key=len)
                print(f"   Largest WCC size: {len(largest_wcc_nodes)} nodes")
                G_lwc = G.subgraph(largest_wcc_nodes)
                print("   Calculating diameter of the largest WCC (treating as undirected)...")
                if nx.is_connected(G_lwc.to_undirected()):
                     diameter = nx.diameter(G_lwc.to_undirected())
                     print(f"   Diameter of Largest Weakly Connected Component: {diameter}")
                else: print("   Warning: Largest WCC subgraph is not connected (unexpected).")
            else: print("   Graph has no connected components.")
        except Exception as e: print(f"   Could not compute diameter: {e}")
    else: print("   Skipping diameter calculation: Graph is empty or has no edges.")

    # Save Full Graph
    print(f"\nSaving full graph to {args.output_graph_file}...")
    try:
        nx.write_adjlist(G, args.output_graph_file)
        save_time = time.time()
        print(f"Full graph saved successfully in {save_time - build_graph_time:.2f} seconds.")
    except Exception as e: print(f"Error saving full graph: {e}")


    # --- Filtered Graph Analysis ---
    print("\n--- Filtered Graph Analysis (Removing Bottom/Top 5% Degree Nodes) ---")
    lower_perc_val = 0 # Initialize
    upper_perc_val = float('inf') # Initialize
    if num_nodes > 20 and full_degrees: # Need nodes and degrees to filter
        try:
            lower_perc_val = np.percentile(full_degrees, 5)
            upper_perc_val = np.percentile(full_degrees, 95)
            # Handle edge case where thresholds might be equal if data is sparse
            if lower_perc_val == upper_perc_val:
                 print(f"   Warning: 5th and 95th percentile degree values are the same ({lower_perc_val}). Filtering may remove many nodes.")
                 # Decide how to handle: maybe keep only this degree? Or adjust slightly?
                 # For now, let's proceed, but be aware.

            print(f"   Degree Thresholds: Keep nodes with total degree >= {lower_perc_val} and <= {upper_perc_val}")
            nodes_to_keep = [n for n, d in G.degree() if d >= lower_perc_val and d <= upper_perc_val]
            print(f"   Keeping {len(nodes_to_keep)} of {num_nodes} nodes.")

            if nodes_to_keep:
                G_filtered = G.subgraph(nodes_to_keep).copy()
                num_nodes_filtered = G_filtered.number_of_nodes()
                edge_count_filtered = G_filtered.number_of_edges()
                print(f"   Filtered graph has {num_nodes_filtered} nodes and {edge_count_filtered} edges.")

                filtered_degrees = []
                avg_total_degree_filtered = 0
                median_total_degree_filtered = 0
                if num_nodes_filtered > 0:
                    avg_in_degree_filtered = sum(d for n, d in G_filtered.in_degree()) / num_nodes_filtered
                    avg_out_degree_filtered = sum(d for n, d in G_filtered.out_degree()) / num_nodes_filtered
                    print(f"   Filtered Average In-Degree: {avg_in_degree_filtered:.4f}")
                    print(f"   Filtered Average Out-Degree: {avg_out_degree_filtered:.4f}")
                    filtered_degrees = [d for n, d in G_filtered.degree()]
                    if filtered_degrees:
                        avg_total_degree_filtered = statistics.mean(filtered_degrees)
                        median_total_degree_filtered = statistics.median(filtered_degrees)
                        print(f"   Filtered Average Total Degree: {avg_total_degree_filtered:.4f}")
                        print(f"   Filtered Median Total Degree: {median_total_degree_filtered}")
                else: print("   Filtered Average Degree: Filtered graph has no nodes.")

                # Plot Filtered Degree Histogram
                if num_nodes_filtered > 0 and filtered_degrees:
                    print(f"   Generating filtered degree histogram...")
                    max_degree_filtered = max(filtered_degrees)
                    plt.figure(figsize=(12, 7))
                    bins_filtered = min(50, max_degree_filtered + 1)
                    # Use density=True if comparing shapes, False for raw counts
                    plt.hist(filtered_degrees, bins=bins_filtered, alpha=0.75, label='Node Count')
                    plt.title('Filtered Node Degree Distribution (Total Degree - Excl. Bottom/Top 5%)')
                    plt.xlabel('Degree')
                    plt.ylabel('Number of Nodes')
                    # Add vertical lines for filtered mean and median
                    plt.axvline(avg_total_degree_filtered, color='r', linestyle='dashed', linewidth=1.5, label=f'Mean (Filt): {avg_total_degree_filtered:.2f}')
                    plt.axvline(median_total_degree_filtered, color='g', linestyle='dotted', linewidth=1.5, label=f'Median (Filt): {median_total_degree_filtered}')
                    # Add lines for the original percentile thresholds used for filtering
                    plt.axvline(lower_perc_val, color='k', linestyle='dashdot', linewidth=1, label=f'5th Perc: {lower_perc_val:.2f}')
                    plt.axvline(upper_perc_val, color='k', linestyle='dashdot', linewidth=1, label=f'95th Perc: {upper_perc_val:.2f}')
                    plt.legend(fontsize='small') # Adjust legend size if needed
                    plt.grid(axis='y', alpha=0.5)
                    plt.text(0.95, 0.95, f'Max Degree: {max_degree_filtered}', transform=plt.gca().transAxes, ha='right', va='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
                    try:
                         plt.savefig(args.output_hist_filtered_file)
                         print(f"   Filtered degree histogram saved to: {args.output_hist_filtered_file}")
                    except Exception as e: print(f"   Error saving filtered histogram: {e}")
                    plt.close()
                else: print("   Skipping filtered histogram: Filtered graph has no nodes or degrees.")
            else: print("   No nodes left after filtering based on degree percentiles.")
        except Exception as e: print(f"   Error during percentile calculation or filtering: {e}")
    else: print("   Skipping filtered analysis: Not enough nodes or degrees in the full graph.")

    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()