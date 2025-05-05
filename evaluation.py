import argparse
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import load_npz # For loading sparse matrix
import os
import re
import networkx as nx # Needed for loading graph and finding neighbors



######### MY CODE START HERE ################### NOTE I AM NOT VERY SURE THIS IS ALLOWD BUT ASSUMING THIS IS OHKK


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


ALPHA = 0.7  # soon ill do some analysis for best alpha value
MAX_IN_DEGREE = 50 # deeping it 50 .. cuz .. max value is around 600 .. that would always almost make freaction 0 ...
SEED_COUNT = 5  # SEED ... for which i check neighbours ... 
TFIDF_VECTORIZER_PATH = "tfidf_vectorizer.pkl" #  learned vocablaty and idf scofres .. baildy for each word .. it has that ifd .. which will be sued to caculatecd fpr vector .. 
DATASET_VECTORS_PATH = "dataset_tfidf_vectors.npz" # precconcomed vecctor for all id .. givn .. to avoid redundecny during test time 
IN_DEGREES_PATH = "in_degrees.txt" # Path to file from in_degree.py # indegree for all ids ... for bacially caluation its importance level 
PAPER_ID_ORDER_PATH = "paper_id_order.txt" # ordre for that .npz .. matrix
GRAPH_PATH = "citation_graph.adjlist" # Path to the graph file
BETA = 0.9 # for sorting remaining

# ---  loaded data (load only once) ---
LOADED_DATA = None

def load_offline_data():
    global LOADED_DATA
    if LOADED_DATA is not None:
        return LOADED_DATA

    try:
        with open(TFIDF_VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)

        dataset_vectors = load_npz(DATASET_VECTORS_PATH) # Load sparse matrix

        in_degrees = {} # Initialize the dictionary to store loaded degrees
        # Load the actual degrees from the file
        try:
            with open(IN_DEGREES_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split() # Split on whitespace
                    if len(parts) == 2:
                        try:
                           node_id = parts[0]
                           degree = int(parts[1])
                           in_degrees[node_id] = degree # Populate the dictionary
                        except ValueError:
                           continue # Skip lines with non-integer degrees
        except FileNotFoundError:
            pass


        effective_max_in_degree = MAX_IN_DEGREE

        paper_ids = []
        with open(PAPER_ID_ORDER_PATH, 'r', encoding='utf-8') as f:
            paper_ids = [line.strip() for line in f if line.strip()]

        graph = nx.read_adjlist(GRAPH_PATH, create_using=nx.DiGraph())

        LOADED_DATA = (vectorizer, dataset_vectors, in_degrees, paper_ids, effective_max_in_degree, graph)

        return LOADED_DATA

    except Exception as e:
        exit(1) # Exit if essential data cannot be loaded
        

def print_err(*args, **kwargs): # just in case 
    """Helper to print to stderr"""
    import sys
    print(*args, file=sys.stderr, **kwargs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-paper-title", type=str, required=True)
    parser.add_argument("--test-paper-abstract", type=str, required=True)
    args = parser.parse_args()

    ################################################
    #               YOUR CODE START                #
    ################################################

    # Load pre-computed models and data
    vectorizer, dataset_vectors, in_degrees, paper_ids, max_in_degree, graph = load_offline_data()

    #  Prepare the input text
    input_text = f"{args.test_paper_title} {args.test_paper_abstract}".strip()
    normalized_input = normalize_text(input_text)

    # Get TF-IDF vector for the input paper
    try:
        input_vector = vectorizer.transform([normalized_input])
    except Exception as e:
        input_vector = None

    # Calculate content similarity scores
    content_scores = {}
    if input_vector is not None and input_vector.nnz > 0 :
        similarities = cosine_similarity(input_vector, dataset_vectors)[0]
        content_scores = {pid: score for pid, score in zip(paper_ids, similarities)}
    else:
        content_scores = {pid: 0.0 for pid in paper_ids}

    # get tok-5 Seed Score ....
    sorted_by_content = sorted(content_scores.keys(), key=lambda pid: content_scores[pid], reverse=True)
    seed_nodes = set(sorted_by_content[SEED_COUNT])

    #  Expand to Candidate Set M (Seed Nodes + Neighbors)
    candidate_set_M = set(seed_nodes) # Start with seeds
    nodes_in_graph = set(graph.nodes()) # For quick checking
    for seed_id in seed_nodes:
        if seed_id in graph: # Check if seed is in the loaded graph
            neighbors = set(graph.successors(seed_id))
            candidate_set_M.update(neighbors)

    candidate_set_M = candidate_set_M.intersection(set(paper_ids))

    final_scores = {}
    for paper_id in candidate_set_M:
        tf_idf_score = content_scores.get(paper_id, 0.0)
        paper_in_degree = in_degrees.get(paper_id, 0)
        normalized_in_degree_score = paper_in_degree / max_in_degree

        #print(normalized_in_degree_score)

        final_score = ALPHA * tf_idf_score + (1 - ALPHA) * normalized_in_degree_score
        final_scores[paper_id] = final_score

    # Sort candidates in M by final score (descending)
    ranked_candidates = sorted(final_scores.keys(), key=lambda pid: final_scores[pid], reverse=True)

    # prepare a ranked list of papers like this:
    result = ranked_candidates # The list of paper IDs from set M, ranked

    # Find papers not yet included in the main result
    remaining_paper_ids_set = set(sorted_by_content) - candidate_set_M
    
    remaining_paper_ids_list = [pid for pid in sorted_by_content if pid in remaining_paper_ids_set]

    # 2. Calculate scores for these remaining papers using BETA
    remaining_scores = {}
    for paper_id in remaining_paper_ids_list:
        tf_idf_score = content_scores.get(paper_id, 0.0)
        paper_in_degree = in_degrees.get(paper_id, 0)
        normalized_in_degree_score = paper_in_degree / max_in_degree

        final_score_remaining = BETA * tf_idf_score + (1 - BETA) * normalized_in_degree_score
        remaining_scores[paper_id] = final_score_remaining

    ranked_remaining = sorted(remaining_paper_ids_list, key=lambda pid: remaining_scores.get(pid, -1.0), reverse=True)

    result = ranked_candidates + ranked_remaining


    ################################################
    #               YOUR CODE END                  #
    ################################################


    ################################################
    #               DO NOT CHANGE                  #
    ################################################
    print('\n'.join(result))

if __name__ == "__main__":
    main()