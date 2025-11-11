#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on November 10 16:04 2025
@author: @jhambale
example run 1: python 17_network_rep.py -i ../../minibinders_orthorep_data/minibinders_orthorep_outputs/jh_008/neutral_drift_library/*scores*.csv -r ../../minibinders_orthorep_data/ngs_raw/jh_008/references/pmaa23_nbonly.fasta -c 50
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from utils_nd import *

import argparse

def build_sequence_network(sequences: List[str], scores: List[float]) -> Tuple[nx.Graph, Dict]:
    """
    Build a network where nodes are sequences and edges connect sequences
    that differ by exactly 1 amino acid.
    
    Args:
        sequences: List of amino acid sequences
        scores: List of function scores corresponding to each sequence
    
    Returns:
        G: NetworkX graph
        node_data: Dictionary mapping sequences to their scores
    """
    # Create graph
    G = nx.Graph()
    
    # Create node data dictionary
    node_data = {seq: score for seq, score in zip(sequences, scores)}
    
    # Add nodes with their scores as attributes
    for seq, score in node_data.items():
        G.add_node(seq, score=score)
    
    # Add edges between sequences that differ by exactly 1 amino acid
    for i, seq1 in enumerate(sequences):
        for j, seq2 in enumerate(sequences[i+1:], start=i+1):
            if hamming_distance(seq1, seq2) == 1:
                G.add_edge(seq1, seq2)
    
    return G, node_data

def visualize_network(G: nx.Graph, node_data: Dict, expt_id='',
                     layout='spring', figsize=(12, 10),
                     cmap='viridis', node_size=100,
                     save_path=None):
    """
    Visualize the sequence network with nodes colored by function score.
    
    Args:
        G: NetworkX graph
        node_data: Dictionary mapping sequences to scores
        layout: Layout algorithm ('spring', 'circular', 'kamada_kawai', 'spectral')
        figsize: Figure size tuple
        cmap: Colormap name for heatmap coloring
        node_size: Size of nodes
        save_path: Optional path to save the figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Choose layout
    if layout == 'spring':
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'spectral':
        pos = nx.spectral_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Get scores for coloring
    scores = [node_data[node] for node in G.nodes()]
    
    # Normalize scores for colormap
    norm = mcolors.Normalize(vmin=min(scores), vmax=max(scores))
    cmap_obj = plt.get_cmap(cmap)
    
    # Draw network
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=1.5, edge_color='gray', ax=ax)
    
    # Draw nodes with color based on score
    nodes = nx.draw_networkx_nodes(G, pos, 
                                   node_color=scores,
                                   node_size=node_size,
                                   cmap=cmap_obj,
                                   vmin=min(scores),
                                   vmax=max(scores),
                                   ax=ax)
    
    # Add labels (optional - may be cluttered for large networks)
    # Uncomment the next line if you want to show sequence labels
    # nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    
    # Add colorbar
    cbar = plt.colorbar(nodes, ax=ax)
    cbar.set_label('Function Score', rotation=270, labelpad=20, fontsize=12)
    
    # Set title and remove axes
    ax.set_title(f'Amino Acid Sequence Network for library {expt_id}\n(Edges connect sequences differing by 1 amino acid)', 
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # plt.show()

def analyze_network(G: nx.Graph, node_data: Dict):
    """
    Print basic network statistics.
    """
    print("=" * 50)
    print("NETWORK STATISTICS")
    print("=" * 50)
    print(f"Number of sequences (nodes): {G.number_of_nodes()}")
    print(f"Number of connections (edges): {G.number_of_edges()}")
    print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    print(f"Network density: {nx.density(G):.4f}")
    print(f"Number of connected components: {nx.number_connected_components(G)}")
    
    # Score statistics
    scores = list(node_data.values())
    print(f"\nFunction Score Statistics:")
    print(f"  Min: {min(scores):.4f}")
    print(f"  Max: {max(scores):.4f}")
    print(f"  Mean: {np.mean(scores):.4f}")
    print(f"  Std: {np.std(scores):.4f}")
    
    # Find most connected sequences
    degrees = dict(G.degree())
    top_connected = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\nTop 5 most connected sequences:")
    for seq, degree in top_connected:
        print(f"  {seq}: {degree} connections (score: {node_data[seq]:.4f})")
    
    print("=" * 50)

# ============================================
# EXAMPLE USAGE
# ============================================

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

    dir_prefix =  '/'.join(all_score_paths[0].split('/')[:-1]) + '/plots/'
    print(f'output directory: {dir_prefix}')

    keyword = f'thresh{cpm}'
    score_seq_path = [path for path in all_score_paths if isinstance(path, str) and keyword in path]

    print(score_seq_path)
    
    expt_id = score_seq_path[0].split('/')[-3] # experiment id is the entry past the 4th slash
    print(f'experiment id: {expt_id}')
    
    score_seq_df = pd.read_csv(score_seq_path[0], index_col=0)

    sequences = score_seq_df['aa_sequence']

    score_seq_df['average_score'] = score_seq_df[['replicate_1', 'replicate_2']].mean(axis=1)
    
    function_scores = score_seq_df['average_score']

     # Build network
    print("Building sequence network...")
    G, node_data = build_sequence_network(sequences, function_scores)
    
    # Analyze network
    analyze_network(G, node_data)
    
    # Visualize network
    print("\nVisualizing network...")
    visualize_network(G, node_data, expt_id, 
                     layout='spring',  # Try 'circular', 'kamada_kawai', 'spectral'
                     cmap='RdYlGn',    # Try 'viridis', 'plasma', 'coolwarm', 'RdYlGn'
                     node_size=100,
                     save_path = dir_prefix + f'{expt_id}_sequence_network.png')  # Set to None to not save

if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     # Example data - replace with your actual data
#     sequences = [
#         "ACDEFG",
#         "ACDEFH",  # 1 diff from ACDEFG
#         "ACDEFI",  # 1 diff from ACDEFG
#         "BCDEFG",  # 1 diff from ACDEFG
#         "ACDEFF",  # 1 diff from ACDEFG
#         "ACDEFJ",  # 1 diff from ACDEFG
#         "ACDEFK",  # 1 diff from ACDEFG
#         "BCDEFF",  # 1 diff from BCDEFG and ACDEFF
#         "BCDEFH",  # 1 diff from BCDEFG and ACDEFH
#     ]
    
#     # Example function scores - replace with your actual scores
#     scores = [0.5, 0.7, 0.3, 0.8, 0.6, 0.9, 0.4, 0.75, 0.85]
    
#     # Build network
#     print("Building sequence network...")
#     G, node_data = build_sequence_network(sequences, scores)
    
#     # Analyze network
#     analyze_network(G, node_data)
#     # Visualize network
#     print("\nVisualizing network...")
#     visualize_network(G, node_data, 
#                      layout='spring',  # Try 'circular', 'kamada_kawai', 'spectral'
#                      cmap='RdYlGn',    # Try 'viridis', 'plasma', 'coolwarm', 'RdYlGn'
#                      node_size=800,
#                      save_path='sequence_network.png')  # Set to None to not save