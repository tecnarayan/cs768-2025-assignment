# CS768 Assignment: Citation Network Analysis and Recommendation

## Overview

This project addresses the CS768 assignment, focusing on graph construction, analysis, and link prediction within a citation network context. We use a dataset of ~6500 research papers from NeurIPS and ICML to:
1.  **Task 1:** Build and analyze a directed citation graph based on references found within the papers' bibliography files.
2.  **Task 2:** Develop a machine learning model (`evaluation.py`) to recommend papers from the dataset that a new, unseen paper might cite.

## Dataset

*   **Source:** Provided dataset containing ~6500 folders, each representing a research paper from NeurIPS/ICML.
*   **Contents:** Each folder (`<paper_id>`) contains `title.txt`, `abstract.txt`, and bibliography files (`.bbl` and/or `.bib`).
*   **Note:** The dataset was manually curated and may contain noisy entries.

## Setup and Dependencies

1.  **Clone the repository:**
    ```bash
    git clone <your_repo_url>
    cd <your_repo_directory>
    ```
2.  **Dataset:** Ensure the `dataset_papers` directory (extracted from `dataset_papers.tar.gz` or downloaded) is present in the root project directory.
3.  **Python Environment:** A Python 3 environment is required.
4.  **Install Dependencies:**
    ```bash
    pip install numpy scipy scikit-learn networkx matplotlib
    ```
5.  **Plots Directory:** Create a directory for output plots:
    ```bash
    mkdir -p plots
    ```
    (The scripts `citation_graph_generator.py` and `in_degree.py` save plots here by default).

## Code Structure

*   `dataset_papers/`: Directory containing the paper data folders.
*   `citation_graph_generator.py`: Script for Task 1 - builds the citation graph using exact normalized title matching and performs graph analysis (degree distribution, diameter, etc.). Saves the graph and plots.
*   `precompute_data.py`: Script to perform offline precomputation for Task 2 - fits TF-IDF vectorizer, computes and saves TF-IDF vectors for all papers, saves paper ID order.
*   `in_degree.py`: Calculates and analyzes the in-degree of each node in the generated graph. Saves in-degrees to a file and plots the distribution.
*   `evaluation.py`: Script for Task 2 - takes a test paper's title and abstract, loads precomputed data, and prints a ranked list of predicted citations using a hybrid TF-IDF + In-Degree approach. **This is the file submitted for evaluation.**
*   `autograder.py`: A script to simulate the evaluation process locally using the ground truth graph and calculating Average Recall@K.
*   `analysis/similarity_score.py`: (Optional) Script to analyze the content similarity between citing and cited papers in the graph.
*   `plots/`: Directory where generated histograms are saved.
*   `tfidf_vectorizer.pkl`: Saved TF-IDF vectorizer object (Output of `precompute_data.py`).
*   `dataset_tfidf_vectors.npz`: Saved sparse matrix of TF-IDF vectors for the dataset (Output of `precompute_data.py`).
*   `paper_id_order.txt`: List of paper IDs corresponding to the rows of the TF-IDF matrix (Output of `precompute_data.py`).
*   `citation_graph.adjlist`: The generated citation graph (Output of `citation_graph_generator.py`).
*   `in_degrees.txt`: List of paper IDs and their corresponding in-degrees (Output of `in_degree.py`).
*   `README.md`: This file.

## How to Run

**Important:** Ensure the text normalization function (`normalize_text`) and the `known_joins` dictionary are *identical* across all relevant scripts (`citation_graph_generator.py`, `precompute_data.py`, `in_degree.py`, `evaluation.py`, `analysis/similarity_score.py`) before running. The less aggressive version (without venue/year/page removal) should be used consistently.

1.  **Task 1: Build Graph & Analyze Properties:**
    ```bash
    python citation_graph_generator.py
    ```
    *   *Input:* Reads from `./dataset_papers`.
    *   *Outputs:* `citation_graph.adjlist`, `./plots/degree_histogram.png`, `./plots/degree_histogram_filtered.png`. Prints analysis results to console.

2.  **Precompute Data for Task 2:**
    ```bash
    python precompute_data.py
    ```
    *   *Input:* Reads from `./dataset_papers`.
    *   *Outputs:* `tfidf_vectorizer.pkl`, `dataset_tfidf_vectors.npz`, `paper_id_order.txt`.

3.  **Calculate In-Degrees:**
    ```bash
    python in_degree.py
    ```
    *   *Input:* Reads `citation_graph.adjlist`.
    *   *Outputs:* `in_degrees.txt` (tab-separated by default), `./plots/in_degree_histogram.png`. Prints analysis and top 5 cited papers to console.
    *   **Action Required:** Note the `Max In-Degree` value printed. You **must** update the `MAX_IN_DEGREE` constant inside `evaluation.py` with this value (or decide on your normalization strategy - see Task 2 Methodology).

4.  **Optional: Analyze Neighbor Similarity:**
    ```bash
    python analysis/similarity_score.py ./dataset_papers/ citation_graph.adjlist
    ```
    *   *Input:* Reads from `./dataset_papers` and `citation_graph.adjlist`.
    *   *Outputs:* `./plots/neighbor_similarity_histogram.png`. Prints similarity statistics.

5.  **Simulate Task 2 Evaluation:**
    ```bash
    python autograder.py ./dataset_papers/ evaluation.py citation_graph.adjlist -n 100 -k 10
    ```
    *   *Input:* Reads from `./dataset_papers`, uses `evaluation.py` for predictions, `citation_graph.adjlist` as ground truth.
    *   *Outputs:* Average Recall@10 for 100 randomly sampled papers.

## Methodology

### Task 1: Graph Construction

A directed graph $G=(V, E)$ was built where $V$ is the set of paper IDs and $(u, v) \in E$ if paper $u$ cites paper $v$. Edges were determined based on an **exact match** between normalized citation strings from $u$'s bibliography and normalized titles of papers $v$ in the dataset.

*   **Normalization (`normalize_text`):** Applied consistently to titles and extracted citation strings. Steps include: lowercasing, expanding known joined terms (e.g., `machinelearning` -> `machine learning`), replacing hyphens between words with spaces, removing LaTeX braces (`{...}`), removing all punctuation except whitespace, and standardizing whitespace. The *less aggressive* version was used in the final run.
*   **Matching:** A lookup table mapping normalized titles to paper IDs was created. Each normalized citation from a source paper was checked against this table to find matches and add corresponding edges (excluding self-citations).

### Task 2: Citation Recommendation (`evaluation.py`)

A hybrid approach combining content similarity and graph-based importance was used, incorporating a candidate selection strategy.

*   **Offline Steps (`precompute_data.py`):**
    *   Concatenated title and abstract for all dataset papers.
    *   Trained a `TfidfVectorizer` (unigrams, English stopwords, consistent `normalize_text`) on this corpus. Saved the vectorizer.
    *   Computed and saved the TF-IDF vectors (sparse matrix `.npz`) and the corresponding paper ID order (`.txt`).
    *   Calculated and saved in-degrees for all nodes (`in_degree.py` -> `in_degrees.txt`). Noted the maximum in-degree (MaxID) for normalization.
*   **Online Prediction (`evaluation.py`):**
    1.  Load precomputed data (vectorizer, vectors, IDs, in-degrees, graph, MaxID).
    2.  Process input title/abstract: Concatenate, normalize.
    3.  Generate input TF-IDF vector using the loaded vectorizer.
    4.  Calculate cosine similarity between the input vector and all dataset vectors (`content_scores`).
    5.  Select top `SEED_COUNT` (e.g., 30) papers by content score, excluding the top match (assumed self), as `seed_nodes`.
    6.  Create `candidate_set_M` = `seed_nodes` $\cup$ {successors of `seed_nodes` in graph}.
    7.  Rank papers $p \in M$ using `Score = alpha * TFIDF(p) + (1-alpha) * (InDegree(p) / MaxID_eff)`. (Used `alpha=0.7`, `MaxID_eff` currently hardcoded to 50.0). Let this be `ranked_candidates`.
    8.  Rank remaining papers $p \notin M$ using `Score = beta * TFIDF(p) + (1-beta) * (InDegree(p) / MaxID_eff)`. (Used `beta=0.5`). Let this be `ranked_remaining`.
    9.  Final result is `ranked_candidates + ranked_remaining`.
    10. Print the final ranked list of paper IDs.

## Results

### Task 1: Graph Analysis

Based on the run using the *less aggressive* normalization:
*   **Nodes:** 6545
*   **Edges:** 30762
*   **Isolated Nodes:** 441
*   **Average In/Out Degree:** 4.7001
*   **Average Total Degree:** 9.4002
*   **Median Total Degree:** 6
*   **LWCC Diameter:** 13 (on 6055 nodes)
*   **Max In-Degree:** [Insert Final Max In-Degree value from `in_degree.py` output here]
*   **Top 5 Cited Papers (In-Degree):**
    1.  [Insert ID, Degree from final `in_degree.py` output]
    2.  [Insert ID, Degree from final `in_degree.py` output]
    3.  [Insert ID, Degree from final `in_degree.py` output]
    4.  [Insert ID, Degree from final `in_degree.py` output]
    5.  [Insert ID, Degree from final `in_degree.py` output]
*   **Degree Histograms:** See `plots/degree_histogram.png` and `plots/degree_histogram_filtered.png`. Both show highly skewed distributions typical of citation networks.
*   **In-Degree Histogram:** See `plots/in_degree_histogram.png`. Median in-degree is 1.

### Task 2: Link Prediction Evaluation

*   **Method:** Hybrid TF-IDF + In-Degree with Seed/Neighbor Candidate Selection.
*   **Metric:** Average Recall@K (simulated via `autograder.py`).
*   **Result:** [Report the Average Recall@K value obtained from running `autograder.py` here, e.g., "Average Recall@10 on 100 sampled papers: 0.XXXX"].
*   **Note:** The `evaluation.py` script returned an average of [Report value, e.g., ~88] predictions per paper, covering the full candidate list. The autograder only considers the top K for scoring.

### Content Similarity Analysis

*   Analysis of TF-IDF cosine similarity between citing papers and the papers they cite (30762 pairs) showed low average similarity.
*   **Mean Similarity:** 0.1046
*   **Median Similarity:** 0.0781
*   **Histogram:** See `plots/neighbor_similarity_histogram.png`. This suggests simple content overlap is often low, motivating hybrid or semantic approaches.

## Discussion

The implemented approach successfully builds the citation graph and provides a baseline hybrid recommendation system. The exact match strategy for graph building is efficient but likely underestimates connectivity. The Task 2 hybrid model leverages both content and graph structure. The use of `MaxID_eff = 50` in `evaluation.py` needs careful consideration, as it non-linearly boosts papers with in-degrees above 50 relative to the TF-IDF score. Future work could involve exploring fuzzy matching for graph building, using semantic embeddings (like Sentence-BERT), employing GNNs, and rigorously tuning hyperparameters like `alpha`, `beta`, `SEED_COUNT`, and the in-degree normalization factor.