import os
import re
import argparse
import time
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz # For saving sparse matrix

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


def process_paper_folder_for_text(folder_path): # given path .. expract conntexxt .. wchich is paper_title + paper_abstract ... 
    """Processes a single paper folder to extract ID and combined title+abstract text."""
    if not os.path.isdir(folder_path): return None, None
    paper_id = os.path.basename(os.path.normpath(folder_path)); paper_title = ""; paper_abstract = ""
    title_file_path = os.path.join(folder_path, 'title.txt')
    try:
        with open(title_file_path, 'r', encoding='utf-8', errors='ignore') as f: paper_title = f.read().strip()
    except: pass
    abstract_file_path = os.path.join(folder_path, 'abstract.txt')
    try:
        with open(abstract_file_path, 'r', encoding='utf-8', errors='ignore') as f: paper_abstract = f.read().strip()
    except: pass
    combined_text = f"{paper_title} {paper_abstract}".strip(); return paper_id, combined_text



def main():
    parser = argparse.ArgumentParser(description="Precompute TF-IDF model, vectors, and ID order for citation prediction.")
    parser.add_argument("--dataset_path", type=str, default= "./dataset_papers", help="Path to the root directory containing paper folders.")
    parser.add_argument("--vec_out", type=str, default="tfidf_vectorizer.pkl", help="Output path for the fitted TfidfVectorizer.")
    parser.add_argument("--mat_out", type=str, default="dataset_tfidf_vectors.npz", help="Output path for the computed TF-IDF sparse matrix.")
    parser.add_argument("--ids_out", type=str, default="paper_id_order.txt", help="Output path for the list of paper IDs in matrix order.")

    args = parser.parse_args()
    start_time = time.time()

    #  Load Paper Texts and Maintain Order
    print(f"Loading paper text data from: {args.dataset_path}...")
    paper_texts_list = [] # List to store text in order
    paper_id_order = []   # List to store IDs in the same order
    paper_folders = [d for d in os.listdir(args.dataset_path) if os.path.isdir(os.path.join(args.dataset_path, d))]
    if not paper_folders: print(f"Error: No paper folders found in {args.dataset_path}"); return

    for folder_name in sorted(paper_folders): # Sort for consistent order # bacically id in order 
        folder_path = os.path.join(args.dataset_path, folder_name)
        paper_id, combined_text = process_paper_folder_for_text(folder_path)
        if paper_id:
            paper_id_order.append(paper_id)
            paper_texts_list.append(combined_text if combined_text else "") # Use empty string if no text

    if not paper_id_order: print("Error: No paper data could be loaded."); return
    load_text_time = time.time()
    print(f"Loaded text data for {len(paper_id_order)} papers in {load_text_time - start_time:.2f} seconds.")

    #  Fit TF-IDF Vectorizer
    print("Fitting TF-IDF Vectorizer (Unigrams) on combined title+abstract...")
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 1), # Unigrams only
        preprocessor=normalize_text # Use the consistent normalizer
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(paper_texts_list)
        vocabulary = vectorizer.get_feature_names_out()
        vocab_size = len(vocabulary)
        print(f"Vocabulary size: {vocab_size}")
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}") # Should be (num_papers, vocab_size)
    except Exception as e:
        print(f"Error during TF-IDF vectorization: {e}")
        return

    fit_tfidf_time = time.time()
    print(f"TF-IDF fitting completed in {fit_tfidf_time - load_text_time:.2f} seconds.")

    print(f"Saving fitted vectorizer to {args.vec_out}...")
    try:
        with open(args.vec_out, 'wb') as f_vec:
            pickle.dump(vectorizer, f_vec)
    except Exception as e: print(f"Error saving vectorizer: {e}"); return

    print(f"Saving TF-IDF matrix to {args.mat_out}...")
    try:
        save_npz(args.mat_out, tfidf_matrix)
    except Exception as e: print(f"Error saving matrix: {e}"); return

    print(f"Saving paper ID order to {args.ids_out}...")
    try:
        with open(args.ids_out, 'w', encoding='utf-8') as f_ids:
            for pid in paper_id_order:
                f_ids.write(f"{pid}\n")
    except Exception as e: print(f"Error saving ID order: {e}"); return

    save_time = time.time()
    print(f"Offline data saved successfully in {save_time - fit_tfidf_time:.2f} seconds.")
    print(f"\nPrecomputation finished. Total time: {save_time - start_time:.2f} seconds.")
    
if __name__ == "__main__":
    main()

