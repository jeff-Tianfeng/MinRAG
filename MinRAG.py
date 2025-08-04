import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List
import os
import pandas as pd
from tqdm import tqdm
from collections import Counter

class KnowledgePointEntropyAnalyzer:
    """
    Analyze the entropy of knowledge points in a binary message matrix.
    """
    
    def __init__(self, alpha: float = 1e-6):
        """
        Args:
            alpha: Laplace smoothing factor to avoid zero probabilities
        """
        self.alpha = alpha
    
    def add_background(self, B: np.ndarray) -> np.ndarray:
        """
        Add small background noise to avoid zero probabilities.
        
        Args:
            B: N x M binary matrix (messages x knowledge points)
        
        Returns:
            B_prime: smoothed matrix
        """
        n, M = B.shape
        background = self.alpha / (n * M)
        B_prime = B + background
        return B_prime
    
    def normalize_to_probability(self, B_prime: np.ndarray) -> np.ndarray:
        """
        Normalize the matrix to a probability distribution.
        """
        S = np.sum(B_prime)
        P = B_prime / S
        return P

    def calculate_type2_entropy(self, P: np.ndarray) -> float:
        """
        Calculate Shannon entropy of the flattened probability distribution.
        
        Args:
            P: Probability matrix
        
        Returns:
            H_element: Shannon entropy value
        """
        P_flat = P.flatten()
        P_nonzero = P_flat[P_flat > 0]  # avoid log(0)
        H_element = -np.sum(P_nonzero * np.log2(P_nonzero))
        return H_element
    
    def analyze(self, B: np.ndarray) -> Dict:
        """
        Analyze the entropy for a given sample matrix.
        
        Args:
            B: binary matrix of shape N x M
        
        Returns:
            Dictionary containing processed matrices and entropy values
        """
        B_prime = self.add_background(B)
        P = self.normalize_to_probability(B_prime)
        H_element = self.calculate_type2_entropy(P)
        
        return {
            'B': B,
            'B_prime': B_prime,
            'P': P,
            'type2': H_element,
            'n_messages': B.shape[0],
            'n_knowledge_points': B.shape[1]
        }

def run_sampling_entropy(matrix: np.ndarray,
                         sample_sizes: List[int],
                         n_trials: int,
                         alpha: float,
                         method: str = "random") -> pd.DataFrame:
    """
    Run entropy experiments under different sampling strategies.

    Args:
        matrix: Original binary matrix (N x M)
        sample_sizes: List of sample sizes
        n_trials: Number of trials per sample size
        alpha: Laplace smoothing factor
        method: "random" or "greedy"
    
    Returns:
        DataFrame of entropy results
    """
    analyzer = KnowledgePointEntropyAnalyzer(alpha=alpha)
    records = []

    for size in tqdm(sample_sizes, desc=f"{method} sampling"):
        for trial in range(n_trials):
            if method == "random":
                # Random sampling with replacement
                indices = np.random.choice(matrix.shape[0], size=size, replace=True)
                sampled = matrix[indices]
            elif method == "greedy":
                # Greedy sampling prioritizing high-entropy knowledge points
                sampled = greedy_entropy_sampling(matrix, n_select=size)
            else:
                raise ValueError(f"Unsupported sampling method: {method}")

            result = analyzer.analyze(sampled)
            log_n = np.log2(size)
            records.append({
                "method": method,
                "sample_size": size,
                "trial": trial,
                "log_n": log_n,
                "H_element": result['type2'],
                "H_element_norm": result['type2'] / log_n
            })

    return pd.DataFrame(records)

def greedy_entropy_sampling(matrix: np.ndarray, n_select: int) -> np.ndarray:
    """
    Greedy sampling: select message rows that cover high-entropy knowledge points first.
    (贪心采样：优先选择包含高熵知识点的消息行)
    
    Args:
        matrix: Original N x M binary knowledge point matrix
        n_select: Number of messages to select

    Returns:
        Submatrix of size n_select x M
    """
    n, m = matrix.shape
    B = matrix.copy()

    # Step 1: Calculate marginal entropy for each knowledge point
    def binary_entropy(p):
        if p == 0 or p == 1:
            return 0
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    p_j = np.mean(B, axis=0)
    H_j = np.array([binary_entropy(p) for p in p_j])
    sorted_col_indices = np.argsort(-H_j)  # sort by entropy descending

    selected_rows = set()
    covered_cols = set()

    for col in sorted_col_indices:
        # Step 2: Find rows containing this knowledge point
        rows_with_col = set(np.where(B[:, col] == 1)[0])
        candidate_rows = rows_with_col - selected_rows

        for row in candidate_rows:
            selected_rows.add(row)
            covered_cols.add(col)
            if len(selected_rows) >= n_select:
                break
        if len(selected_rows) >= n_select:
            break

    # Step 3: If not enough rows, fill randomly
    if len(selected_rows) < n_select:
        remaining = list(set(range(n)) - selected_rows)
        supplement = np.random.choice(remaining, size=n_select - len(selected_rows), replace=False)
        selected_rows.update(supplement)

    selected_rows = sorted(list(selected_rows))
    return B[selected_rows]


def plot(df_all: pd.DataFrame):
    """
    Plot the average entropy curves for different sampling methods.
    """
    df_avg = df_all.groupby(['sample_size', 'method']).agg({
        'H_element': 'mean',
        'H_element_norm': 'mean'
    }).reset_index()

    df_avg.to_csv("type2_entropy_averaged.csv", index=False)
    print("✅ Averaged entropy results saved to type2_entropy_averaged.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot raw Shannon entropy
    sns.lineplot(
        data=df_avg, x="sample_size", y="H_element",
        hue="method", ax=axes[0], linewidth=2
    )
    axes[0].set_title("Shannon Entropy")
    axes[0].set_xlabel("Sample Size")
    axes[0].set_ylabel("Entropy Value")
    axes[0].legend(title="Method")
    axes[0].grid(False)

    # Plot normalized entropy
    sns.lineplot(
        data=df_avg, x="sample_size", y="H_element_norm",
        hue="method", ax=axes[1], linewidth=2
    )
    axes[1].set_title("Shannon Entropy / log2(n)")
    axes[1].set_xlabel("Sample Size")
    axes[1].set_ylabel("Unit entropy")
    axes[1].legend(title="Method")
    axes[1].grid(False)

    plt.tight_layout()
    plt.savefig("type2_entropy_comparison_smooth.png", dpi=300)
    plt.show()
    print("✅ Smoothed type-2 entropy plot saved as type2_entropy_comparison_smooth.png")

if __name__ == "__main__":
    path = "build_matrix/matrix.npy"
    matrix = np.load(path)
    sample_sizes = list(range(50, 30000, 50))  # Sampling sizes to evaluate
    n_trials = 10  # Number of repeated trials for each sample size
    alpha = 1e-6   # Laplace smoothing factor

    # Run random sampling entropy analysis
    df_random = run_sampling_entropy(matrix, sample_sizes, n_trials, alpha, method="random")
    # Run greedy sampling entropy analysis
    df_greedy = run_sampling_entropy(matrix, sample_sizes, n_trials, alpha, method="greedy")

    df_all = pd.concat([df_random, df_greedy], ignore_index=True)
    plot(df_all)
