import os
import re
import argparse
import subprocess
import random
import time
import networkx as nx
import sys
import numpy as np # For calculating average recall

# --- Helper function to read title/abstract (Unchanged) ---
def read_paper_data(dataset_path, paper_id):
    """Reads title and abstract for a given paper ID."""
    folder_path = os.path.join(dataset_path, paper_id)
    title = ""
    abstract = ""
    title_file = os.path.join(folder_path, 'title.txt')
    abstract_file = os.path.join(folder_path, 'abstract.txt')
    try:
        with open(title_file, 'r', encoding='utf-8', errors='ignore') as f:
            title = f.read().strip()
    except FileNotFoundError: pass
    except Exception as e: print(f"Warning: Error reading title for {paper_id}: {e}", file=sys.stderr)
    try:
        with open(abstract_file, 'r', encoding='utf-8', errors='ignore') as f:
            abstract = f.read().strip()
    except FileNotFoundError: pass
    except Exception as e: print(f"Warning: Error reading abstract for {paper_id}: {e}", file=sys.stderr)
    return title, abstract

# --- Helper function to run the evaluation script (Unchanged) ---
def run_evaluation_script(script_path, title, abstract):
    """Runs the student's evaluation script and returns the output list."""
    try:
        if not script_path.endswith(".py"): print(f"Error: Eval script path '{script_path}' not py.", file=sys.stderr); return []
        if not os.path.exists(script_path): print(f"Error: Eval script '{script_path}' not found.", file=sys.stderr); return []
        python_executable = sys.executable
        completed_process = subprocess.run(
            [python_executable, script_path, "--test-paper-title", str(title), "--test-paper-abstract", str(abstract)],
            capture_output=True, text=True, timeout=60 )
        if completed_process.returncode != 0:
            # Limit printing potentially long titles/abstracts in error messages
            title_preview = title[:50] + '...' if len(title) > 50 else title
            print(f"\nWarning: Eval script for '{title_preview}' exit code {completed_process.returncode}.", file=sys.stderr)
            print(f"Stderr:\n{completed_process.stderr}", file=sys.stderr); return []
        result = completed_process.stdout.strip().split('\n'); result = [line for line in result if line]
        return result
    except subprocess.TimeoutExpired:
        title_preview = title[:50] + '...' if len(title) > 50 else title
        print(f"\nWarning: Eval script timed out for '{title_preview}'", file=sys.stderr); return []
    except Exception as e: print(f"Error running subprocess for {script_path}: {e}", file=sys.stderr); return []


# --- Main Autograder Logic ---
def main():
    parser = argparse.ArgumentParser(description="Autograder for CS768 Task 2 evaluation (Average Recall@K).") # Updated description
    parser.add_argument("dataset_path", type=str, help="Path to the root dataset directory (e.g., ./dataset_papers).")
    parser.add_argument("evaluation_script", type=str, help="Path to the student's evaluation script (e.g., evaluation.py).")
    parser.add_argument("graph_file", type=str, help="Path to the ground truth citation graph file (e.g., citation_graph.adjlist).")
    parser.add_argument("-n", "--num_papers", type=int, default=100, help="Number of random papers to evaluate.")
    parser.add_argument("-k", "--recall_k", type=int, default=10, help="Value of K for Recall@K calculation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for paper selection reproducibility.")

    args = parser.parse_args()

    print(f"--- Autograder Starting (Average Recall@K Mode) ---") # Updated mode
    print(f"Dataset Path: {args.dataset_path}")
    print(f"Evaluation Script: {args.evaluation_script}")
    print(f"Ground Truth Graph: {args.graph_file}")
    print(f"Number of Papers to Test (N): {args.num_papers}")
    print(f"Recall@K Value (K): {args.recall_k}")
    print(f"Random Seed: {args.seed}")
    print(f"---------------------------")

    start_time = time.time()
    random.seed(args.seed) # Set random seed

    # 1. Load Ground Truth Graph (No change)
    print("Loading ground truth graph...")
    try:
        G_truth = nx.read_adjlist(args.graph_file, create_using=nx.DiGraph())
        print(f"Loaded ground truth graph with {G_truth.number_of_nodes()} nodes.")
    except FileNotFoundError: print(f"FATAL: Ground truth graph file not found at {args.graph_file}"); return
    except Exception as e: print(f"FATAL: Error loading ground truth graph: {e}"); return

    # 2. Get list of all paper IDs (No change)
    try:
        all_paper_ids = [d for d in os.listdir(args.dataset_path) if os.path.isdir(os.path.join(args.dataset_path, d))]
        if not all_paper_ids: print(f"FATAL: No paper folders found in {args.dataset_path}"); return
        print(f"Found {len(all_paper_ids)} total papers in dataset.")
    except FileNotFoundError: print(f"FATAL: Dataset path not found at {args.dataset_path}"); return
    except Exception as e: print(f"FATAL: Error listing dataset papers: {e}"); return

    # 3. Select random subset (No change)
    if args.num_papers >= len(all_paper_ids):
        print("Warning: Requested number of papers >= total papers. Testing all.")
        test_paper_ids = all_paper_ids
        args.num_papers = len(all_paper_ids)
    else:
        test_paper_ids = random.sample(all_paper_ids, args.num_papers)
    print(f"Selected {len(test_paper_ids)} papers for evaluation.")

    # --- MODIFIED SECTION 4: Run evaluation loop - Calculate individual Recall@K ---
    individual_recalls = [] # Store Recall@K for each paper
    papers_evaluated = 0
    total_predictions_made = 0
    papers_with_citations = 0 # Count papers that actually had citations to average over

    for i, paper_id in enumerate(test_paper_ids):
        print(f"Evaluating paper {i+1}/{args.num_papers}: {paper_id} ...", end='\r', flush=True)

        # Check if paper exists in graph
        if paper_id not in G_truth:
            print(f"\nWarning: Skipping {paper_id} - not found in ground truth graph.", file=sys.stderr)
            continue

        # Get paper title and abstract
        title, abstract = read_paper_data(args.dataset_path, paper_id)
        if not title and not abstract:
             # Check if abstract exists even if title is missing (common for some entries)
             if abstract:
                 print(f"\nWarning: Using only abstract for {paper_id} as title is missing.", file=sys.stderr)
             else:
                 print(f"\nWarning: Skipping {paper_id} - could not read title or abstract.", file=sys.stderr)
                 continue

        # Get actual citations (successors), excluding self-loops if any exist
        actual_citations = set(G_truth.successors(paper_id))
        # Remove the paper itself from its potential citations (no self-loops for recall)
        actual_citations.discard(paper_id)
        num_actual = len(actual_citations)

        # Run evaluation script
        predicted_ids = run_evaluation_script(args.evaluation_script, title, abstract)
        total_predictions_made += len(predicted_ids)

        # Calculate hits@K and Recall@K for this paper
        hits_this_paper = 0
        recall_this_paper = 0.0 # Default

        if predicted_ids:
            # Exclude the test paper itself from its predictions
            top_k_predictions = set(predicted_ids[:args.recall_k])
            top_k_predictions.discard(paper_id) # Ensure self not counted

            # Find intersection
            intersection = top_k_predictions.intersection(actual_citations)
            hits_this_paper = len(intersection)

        # Calculate Recall@K for this paper
        if num_actual > 0:
            # Standard case: calculate fraction of actual citations found
            recall_this_paper = hits_this_paper / num_actual
            papers_with_citations += 1
            individual_recalls.append(recall_this_paper)
        #elif num_actual == 0 and hits_this_paper == 0: # Optional: Handle no actual citations case
            # If there were no citations to find, and none were found (in top K), consider it perfect recall?
            # Or just exclude these papers from the average? Let's exclude for now.
            # individual_recalls.append(1.0) # Option 1: Score 1.0
            # pass # Option 2: Exclude (default by not adding to list)
        #else: # num_actual == 0 but hits_this_paper > 0 (predicted ghost citations?)
            # This case shouldn't happen if prediction list is filtered by known paper IDs
            # We simply don't add it to the list if num_actual is 0

        papers_evaluated += 1
    # --- END MODIFIED SECTION 4 ---

    print("\nEvaluation Loop Finished.")

    # --- MODIFIED SECTION 5: Calculate and Print Average Recall@K ---
    if papers_evaluated > 0:
        # Calculate Average Recall@K only over papers that had citations
        if papers_with_citations > 0:
             average_recall_at_k = np.mean(individual_recalls) if individual_recalls else 0.0
        else:
             average_recall_at_k = 0.0 # Or undefined/NaN if preferred

        print(f"\n--- Results ---")
        print(f"Papers Evaluated: {papers_evaluated}")
        print(f"Papers With >0 Actual Citations (used for Avg Recall): {papers_with_citations}")
        print(f"Average Recall@{args.recall_k}: {average_recall_at_k:.4f}") # Main metric
        avg_preds = total_predictions_made / papers_evaluated if papers_evaluated > 0 else 0
        print(f"Average number of predictions returned per paper: {avg_preds:.1f}")
    else:
        print("\n--- Results ---")
        print("No papers were successfully evaluated.")
    # --- END MODIFIED SECTION 5 ---

    end_time = time.time()
    print(f"---------------------------")
    print(f"Autograder finished in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    # Need numpy for the average calculation
    import numpy as np
    main()