#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on October 30 16:04 2025
@author: @jhambale
example run 1: python 16_network_rep.py -i ../../minibinders_orthorep_data/minibinders_orthorep_outputs/jh_008/neutral_drift_library/*scores*.csv -r ../../minibinders_orthorep_data/ngs_raw/jh_008/references/pmaa23_nbonly.fasta -c 50
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
import umap
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from Bio.Align import substitution_matrices
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
import pandas as pd

import argparse

def compute_sequence_similarity_matrix(sequences, method='blosum62'):
    """
    Compute pairwise similarity matrix for amino acid sequences.
    
    Parameters:
    -----------
    sequences : list of str
        List of amino acid sequences
    method : str
        'blosum62': Use BLOSUM62 substitution matrix for alignment
        'identity': Simple identity-based similarity
        'levenshtein': Edit distance based similarity
    """
    n = len(sequences)
    similarity_matrix = np.zeros((n, n))
    
    if method == 'blosum62':
        # Use BLOSUM62 matrix for biologically meaningful similarity
        blosum62 = substitution_matrices.load("BLOSUM62")
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # Perform global alignment
                    alignments = pairwise2.align.globalds(
                        sequences[i], 
                        sequences[j],
                        blosum62, 
                        -10,  # gap open penalty
                        -0.5  # gap extend penalty
                    )
                    if alignments:
                        # Normalize by the maximum possible score
                        score = alignments[0].score
                        max_score = max(
                            sum(blosum62[aa, aa] for aa in sequences[i]),
                            sum(blosum62[aa, aa] for aa in sequences[j])
                        )
                        normalized_score = score / max_score if max_score > 0 else 0
                        similarity_matrix[i, j] = normalized_score
                        similarity_matrix[j, i] = normalized_score
    
    elif method == 'identity':
        # Simple identity-based similarity
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # Pad sequences to same length
                    max_len = max(len(sequences[i]), len(sequences[j]))
                    seq1 = sequences[i].ljust(max_len, '-')
                    seq2 = sequences[j].ljust(max_len, '-')
                    
                    # Calculate identity
                    matches = sum(a == b for a, b in zip(seq1, seq2))
                    similarity = matches / max_len
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
    
    elif method == 'levenshtein':
        # Levenshtein distance based similarity
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            if len(s2) == 0:
                return len(s1)
            
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    distance = levenshtein_distance(sequences[i], sequences[j])
                    max_len = max(len(sequences[i]), len(sequences[j]))
                    similarity = 1 - (distance / max_len)
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
    
    return similarity_matrix

def reduce_dimensions(similarity_matrix, method='tsne', random_state=42):
    """
    Reduce high-dimensional similarity matrix to 2D coordinates.
    
    Parameters:
    -----------
    similarity_matrix : np.array
        Pairwise similarity matrix
    method : str
        'tsne': t-SNE dimensionality reduction
        'mds': Multidimensional scaling
        'pca': Principal component analysis (on distance matrix)
        'umap': UMAP dimensionality reduction
    """
    # Convert similarity to distance
    distance_matrix = 1 - similarity_matrix
    np.fill_diagonal(distance_matrix, 0)
    
    if method == 'tsne':
        # t-SNE works better with perplexity adjusted to dataset size
        n_samples = len(similarity_matrix)
        perplexity = min(30, n_samples - 1)  # Adjust perplexity for small datasets
        
        tsne = TSNE(n_components=2, metric='precomputed', init='random',
                    perplexity=perplexity, random_state=random_state)
        coords = tsne.fit_transform(distance_matrix)
    
    elif method == 'mds':
        mds = MDS(n_components=2, dissimilarity='precomputed', 
                  random_state=random_state)
        coords = mds.fit_transform(distance_matrix)
    
    elif method == 'pca':
        # For PCA, we need to work with the similarity matrix directly
        pca = PCA(n_components=2, random_state=random_state)
        coords = pca.fit_transform(similarity_matrix)
    
    elif method == 'umap':
        # UMAP can work with distance matrices
        reducer = umap.UMAP(n_components=2, metric='precomputed', 
                           random_state=random_state)
        coords = reducer.fit_transform(distance_matrix)
    
    return coords

def plot_sequence_landscape(sequences, function_scores, 
                           similarity_method='blosum62',
                           reduction_method='tsne',
                           figsize=(10, 8),
                           cmap='coolwarm',
                           point_size=100,
                           show_labels=False):
    """
    Create a 2D visualization of sequences colored by function scores.
    
    Parameters:
    -----------
    sequences : list of str
        List of amino acid sequences
    function_scores : list or np.array
        Function scores (0-1) for each sequence
    similarity_method : str
        Method for computing sequence similarity
    reduction_method : str
        Method for dimensionality reduction
    figsize : tuple
        Figure size
    cmap : str
        Colormap for function scores
    point_size : int
        Size of points in scatter plot
    show_labels : bool
        Whether to show sequence labels
    """
    # Compute similarity matrix
    print(f"Computing sequence similarity using {similarity_method}...")
    similarity_matrix = compute_sequence_similarity_matrix(sequences, similarity_method)
    
    # Reduce to 2D
    print(f"Reducing dimensions using {reduction_method}...")
    coords = reduce_dimensions(similarity_matrix, reduction_method)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot with color based on function scores
    scatter = ax.scatter(coords[:, 0], coords[:, 1], 
                        c=function_scores, 
                        cmap=cmap,
                        s=point_size,
                        alpha=0.7,
                        edgecolors='black',
                        linewidth=0.5,
                        vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Function Score', rotation=270, labelpad=20)
    
    # Add labels if requested
    if show_labels:
        for i, seq in enumerate(sequences):
            # Show truncated sequence if too long
            label = seq if len(seq) <= 10 else f"{seq[:7]}..."
            ax.annotate(label, (coords[i, 0], coords[i, 1]), 
                       fontsize=8, alpha=0.7)
    
    # Styling
    ax.set_xlabel(f'{reduction_method.upper()} Component 1')
    ax.set_ylabel(f'{reduction_method.upper()} Component 2')
    ax.set_title(f'Sequence Landscape Colored by Function Score ({len(sequences)} sequences)\n'
                f'(Similarity: {similarity_method}, Reduction: {reduction_method})')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax, coords

# Example usage with sample data
def generate_example_data():
    """Generate example amino acid sequences and function scores."""
    # Example sequences (you would replace with your actual data)
    sequences = [
        "MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKF",
        "MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKF",
        "MVLSEGEWQLVLHVWAKVEADIAGHGQDILIRLFKSHPETLEKF",
        "MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKF",
        "MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKF",
        "MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKF",
        "MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKF",
        "MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKF",
        "MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKF",
        "MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKF",
    ]
    
    # Example function scores (normalized 0-1)
    function_scores = np.array([0.95, 0.88, 0.72, 0.65, 0.43, 
                                0.31, 0.25, 0.18, 0.12, 0.05])
    
    return sequences, function_scores

# Main execution
def main():

    # add arguments via command line

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', nargs='+', help='input mutation analysis csv files')
    parser.add_argument('-r', help='Path to reference fasta')
    parser.add_argument('-c', help='counts per million cutoff used for generation of input csvs')

    args = parser.parse_args()
    
    # initialize variables
    
    all_score_paths = args.i
    cpm = args.c

    keyword = f'thresh{cpm}'
    score_seq_path = [path for path in all_score_paths if isinstance(path, str) and keyword in path]

    print(score_seq_path)
    
    expt_id = score_seq_path[0].split('/')[-3] # experiment id is the entry past the 4th slash
    print(f'experiment id: {expt_id}')
    
    score_seq_df = pd.read_csv(score_seq_path[0], index_col=0)

    sequences = score_seq_df['aa_sequence']

    score_seq_df['average_score'] = score_seq_df[['replicate_1', 'replicate_2']].mean(axis=1)
    
    function_scores = score_seq_df['average_score']

    # print(function_scores)

    sequence_test = sequences[:110]
    function_scores_test = function_scores[:110]
    
    # Create visualization
    fig, ax, coords = plot_sequence_landscape(
        sequences, 
        function_scores,
        similarity_method='levenshtein',  # or 'identity', 'levenshtein'
        reduction_method='tsne',  # or 'mds', 'pca', 'umap'
        figsize=(10, 8),
        cmap='coolwarm',  # or 'viridis', 'plasma', 'RdYlBu_r', etc.
        point_size=100,
        show_labels=False  # Set to True to show sequence labels
    )
    
    # plt.show()

    out_dir = f'../../minibinders_orthorep_data/minibinders_orthorep_outputs/{expt_id}/neutral_drift_library/plots/'

    
    # Optional: Save the figure
    fig.savefig(out_dir + f'sequence_landscape_{expt_id}.png', dpi=300, bbox_inches='tight')
    
    # Optional: Save coordinates for further analysis
    # df = pd.DataFrame({
    #     'sequence': sequences,
    #     'function_score': function_scores,
    #     'x': coords[:, 0],
    #     'y': coords[:, 1]
    # })
    # df.to_csv('sequence_coordinates.csv', index=False)

if __name__ == "__main__":
    main()