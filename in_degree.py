import os
import argparse
import time
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import statistics # For mean, median
import re

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




def main():
    parser = argparse.ArgumentParser(description="Analyze In-Degrees of a citation graph.")
    parser.add_argument("--graph_file", type=str, default="./citation_graph.adjlist",help="Path to the citation graph file (e.g., citation_graph.adjlist).")
    parser.add_argument("--output_degree_file", type=str, default="in_degrees.txt", help="File path to save node in-degrees.")
    parser.add_argument("--output_hist_file", type=str, default="./plots/in_degree_histogram.png", help="File path to save the in-degree histogram plot.")
    parser.add_argument("--separator", type=str, default="\t", help="Separator for the output degree file (e.g., '\\t' for tab, ',' for CSV).")


    args = parser.parse_args()
    start_time = time.time()

    print(f"Loading citation graph from: {args.graph_file}...")
    try:
        G = nx.read_adjlist(args.graph_file, create_using=nx.DiGraph())
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        print(f"Loaded graph with {num_nodes} nodes and {num_edges} edges.")
    except FileNotFoundError: print(f"Error: Graph file not found at {args.graph_file}"); return
    except Exception as e: print(f"Error loading graph: {e}"); return

    if num_nodes == 0: print("Graph is empty. No analysis possible."); return

    print("Calculating in-degrees...")
    node_in_degrees = {}
    in_degrees_list = []
    for node, degree in G.in_degree():
        node_in_degrees[node] = degree
        in_degrees_list.append(degree)

    calc_time = time.time()
    print(f"Calculated in-degrees for {len(node_in_degrees)} nodes in {calc_time - start_time:.2f} seconds.")

    # 3. Save In-Degree Data
    print(f"Saving in-degrees to {args.output_degree_file}...")
    try:
        with open(args.output_degree_file, 'w', encoding='utf-8') as f:
            for node, degree in node_in_degrees.items():
                f.write(f"{node}{args.separator}{degree}\n")
        save_data_time = time.time()
        print(f"In-degree data saved successfully in {save_data_time - calc_time:.2f} seconds.")
    except Exception as e: print(f"Error saving in-degree data: {e}")

    print("\n--- In-Degree Analysis ---")

    if not in_degrees_list: print("No in-degrees calculated."); return

    try:
        mean_in_degree = statistics.mean(in_degrees_list)
        median_in_degree = statistics.median(in_degrees_list)
        max_in_degree = max(in_degrees_list) if in_degrees_list else 0
        print(f"Mean In-Degree: {mean_in_degree:.4f}")
        print(f"Median In-Degree: {median_in_degree}")
        print(f"Max In-Degree: {max_in_degree}")

        # --- Find and Print Top 5 Most Cited ---
        print("\nTop 5 Most Cited Papers (Highest In-Degree):")
        if node_in_degrees:
             sorted_degrees = sorted(node_in_degrees.items(), key=lambda item: item[1], reverse=True)
             top_5_cited = sorted_degrees[:5]
             for i, (node, degree) in enumerate(top_5_cited):
                 print(f"  {i+1}. Paper ID: {node}, In-Degree: {degree}")
        else:
            print("  No nodes with in-degrees found.")

        # Plot Histogram
        print(f"\nGenerating in-degree histogram...")
        plt.figure(figsize=(12, 7))
        bins = min(50, max_in_degree + 1)
        plt.hist(in_degrees_list, bins=bins, alpha=0.75, label='Node Count')
        plt.title('Node In-Degree Distribution')
        plt.xlabel('In-Degree (Number of citations received)')
        plt.ylabel('Number of Papers')
        plt.axvline(mean_in_degree, color='r', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_in_degree:.2f}')
        plt.axvline(median_in_degree, color='g', linestyle='dotted', linewidth=1.5, label=f'Median: {median_in_degree}')
        plt.legend()
        plt.grid(axis='y', alpha=0.5)
        plt.text(0.95, 0.95, f'Max In-Degree: {max_in_degree}', transform=plt.gca().transAxes, ha='right', va='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

        try:
            plt.tight_layout()
            plt.savefig(args.output_hist_file)
            print(f"In-degree histogram saved to: {args.output_hist_file}")
        except Exception as e: print(f"   Error saving in-degree histogram: {e}")
        plt.close()

    except statistics.StatisticsError: print("Could not calculate statistics.")
    except Exception as e: print(f"An error occurred during analysis/plotting: {e}")


    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
