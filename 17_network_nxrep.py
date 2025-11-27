#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on November 10 16:04 2025
@author: @jhambale
example run 1: python 17_network_rep.py -i ../../minibinders_orthorep_data/minibinders_orthorep_outputs/jh_008/neutral_drift_library/*scores*.csv -r ../../minibinders_orthorep_data/ngs_raw/jh_008/references/pmaa23_nbonly.fasta -c 50
"""

import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import json
import ast
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

def build_sequence_network_with_frequency(sequences: List[str], 
                                         scores: List[float],
                                         frequencies: List[int]) -> Tuple[nx.Graph, Dict, Dict]:
    """
    Build a network with sequence, score, and frequency data.
    
    Args:
        sequences: List of amino acid sequences (unique)
        scores: List of function scores corresponding to each sequence
        frequencies: List of observation counts for each sequence
    
    Returns:
        G: NetworkX graph
        node_data: Dictionary mapping sequences to their scores
        frequency_data: Dictionary mapping sequences to their frequencies
    """
    # Create dictionaries
    node_data = {seq: score for seq, score in zip(sequences, scores)}
    frequency_data = {seq: freq for seq, freq in zip(sequences, frequencies)}
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes with their scores and frequencies as attributes
    for seq, score, freq in zip(sequences, scores, frequencies):
        G.add_node(seq, score=score, frequency=freq)
    
    # Add edges between sequences that differ by exactly 1 amino acid
    print(f"Building edges for {len(sequences)} sequences...")
    edge_count = 0
    for i, seq1 in enumerate(sequences):
        if (i + 1) % 100 == 0:  # Progress indicator
            print(f"  Processed {i + 1}/{len(sequences)} sequences...")
        for j, seq2 in enumerate(sequences[i+1:], start=i+1):
            if hamming_distance(seq1, seq2) == 1:
                G.add_edge(seq1, seq2)
                edge_count += 1
    
    print(f"Created {edge_count} edges")
    
    return G, node_data, frequency_data

def visualize_network(G: nx.Graph, node_data: Dict, expt_id='',
                     layout='spring', figsize=(12, 10),
                     cmap='viridis', 
                     size_by='frequency',  # NEW PARAMETER
                     min_node_size=100,    # NEW PARAMETER
                     max_node_size=1000,   # NEW PARAMETER
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
    
    # Get sizes based on frequency or other attribute
    if size_by == 'frequency':
        sizes = [G.nodes[node].get('frequency', 1) for node in G.nodes()]
    elif size_by == 'degree':
        sizes = [G.degree(node) for node in G.nodes()]
    else:
        sizes = [1] * len(G.nodes())
    
    # Normalize sizes to the specified range
    if max(sizes) > min(sizes):
        sizes_normalized = [
            min_node_size + (s - min(sizes)) / (max(sizes) - min(sizes)) * (max_node_size - min_node_size)
            for s in sizes
        ]
    else:
        sizes_normalized = [min_node_size] * len(sizes)
    
    # Normalize scores for colormap
    norm = mcolors.Normalize(vmin=min(scores), vmax=max(scores))
    cmap_obj = cm.get_cmap(cmap)
    
    # Draw network
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=1.5, edge_color='gray', ax=ax)
    
    # Draw nodes with color based on score and size based on frequency
    nodes = nx.draw_networkx_nodes(G, pos, 
                                   node_color=scores,
                                   node_size=sizes_normalized,
                                   cmap=cmap_obj,
                                   vmin=min(scores),
                                   vmax=max(scores),
                                   alpha=0.8,
                                   ax=ax)
    
    # Add colorbar for scores
    cbar = plt.colorbar(nodes, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Function Score', rotation=270, labelpad=20, fontsize=12)
    
    # Add size legend
    if size_by:
        # Create legend for node sizes
        legend_sizes = [min(sizes), np.median(sizes), max(sizes)]
        legend_labels = [f'{int(s)}' for s in legend_sizes]
        legend_handles = []
        
        for size, label in zip(legend_sizes, legend_labels):
            size_normalized = min_node_size + (size - min(sizes)) / (max(sizes) - min(sizes)) * (max_node_size - min_node_size) if max(sizes) > min(sizes) else min_node_size
            handle = plt.scatter([], [], s=size_normalized, c='gray', alpha=0.6, edgecolors='black', linewidths=1)
            legend_handles.append(handle)
        
        legend = ax.legend(legend_handles, legend_labels, 
                          scatterpoints=1, 
                          title=f'{size_by.capitalize()}',
                          loc='upper left',
                          frameon=True,
                          fontsize=10)
        legend.get_title().set_fontsize(12)
    
    # Set title and remove axes
    title = 'Amino Acid Sequence Network\n(Edges connect sequences differing by 1 amino acid)'
    if size_by:
        title += f'\nNode size = {size_by}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()

def analyze_network(G: nx.Graph, node_data: Dict, output_file=None):
    """
    Print basic network statistics and optionally save to file.
    
    Args:
        G: NetworkX graph
        node_data: Dictionary mapping sequences to scores
        output_file: Optional path to save the analysis output
    """
    # Collect all output in a list
    output_lines = []
    
    output_lines.append("=" * 50)
    output_lines.append("NETWORK STATISTICS")
    output_lines.append("=" * 50)
    output_lines.append(f"Number of sequences (nodes): {G.number_of_nodes()}")
    output_lines.append(f"Number of connections (edges): {G.number_of_edges()}")
    output_lines.append(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    output_lines.append(f"Network density: {nx.density(G):.4f}")
    output_lines.append(f"Number of connected components: {nx.number_connected_components(G)}")
    
    # Score statistics
    scores = list(node_data.values())
    output_lines.append(f"\nFunction Score Statistics:")
    output_lines.append(f"  Min: {min(scores):.4f}")
    output_lines.append(f"  Max: {max(scores):.4f}")
    output_lines.append(f"  Mean: {np.mean(scores):.4f}")
    output_lines.append(f"  Std: {np.std(scores):.4f}")
    
    # Find most connected sequences
    degrees = dict(G.degree())
    top_connected = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    output_lines.append(f"\nTop 5 most connected sequences:")
    for seq, degree in top_connected:
        output_lines.append(f"  {seq}: {degree} connections (score: {node_data[seq]:.4f})")
    
    output_lines.append("=" * 50)
    
    # Print to terminal
    for line in output_lines:
        print(line)
    
    # Save to file if path provided
    if output_file:
        with open(output_file, 'w') as f:
            f.write('\n'.join(output_lines))
        print(f"\nAnalysis saved to: {output_file}")

def export_for_cytoscape(G: nx.Graph, node_data: Dict, expt_id: str, output_prefix: str):
    """
    Export network data in multiple formats compatible with Cytoscape.
    
    Args:
        G: NetworkX graph
        node_data: Dictionary mapping sequences to scores
        output_prefix: Prefix for output files (e.g., 'jh_019_network')
    """
    base_dir = os.path.dirname(output_prefix)
    base_name = os.path.basename(output_prefix)
    
    # 1. Export as GraphML (RECOMMENDED - preserves all attributes)
    graphml_file = os.path.join(base_dir, f"{base_name}.graphml")
    nx.write_graphml(G, graphml_file)
    print(f"GraphML file saved: {graphml_file}")
    
    # 2. Export as GML
    gml_file = os.path.join(base_dir, f"{base_name}.gml")
    nx.write_gml(G, gml_file)
    print(f"GML file saved: {gml_file}")
    
    # 3. Export node table (CSV)
    node_table = []
    degrees = dict(G.degree())
    for node in G.nodes():
        node_table.append({
            'sequence': node,
            'function_score': node_data[node],
            'degree': degrees[node],
            'sequence_length': len(node)
        })
    
    node_df = pd.DataFrame(node_table)
    node_csv = os.path.join(base_dir, f"{base_name}_nodes.csv")
    node_df.to_csv(node_csv, index=False)
    print(f"Node table saved: {node_csv}")
    
    # 4. Export edge table (CSV)
    edge_table = []
    for edge in G.edges():
        seq1, seq2 = edge
        # Find the position where they differ
        diff_positions = [i for i, (c1, c2) in enumerate(zip(seq1, seq2)) if c1 != c2]
        
        edge_table.append({
            'source': seq1,
            'target': seq2,
            'interaction': 'one_mutation',
            'mutation_position': diff_positions[0] if diff_positions else -1,
            'source_score': node_data[seq1],
            'target_score': node_data[seq2],
            'score_difference': abs(node_data[seq1] - node_data[seq2])
        })
    
    edge_df = pd.DataFrame(edge_table)
    edge_csv = os.path.join(base_dir, f"{base_name}_edges.csv")
    edge_df.to_csv(edge_csv, index=False)
    print(f"Edge table saved: {edge_csv}")
    
    # 5. Export as Cytoscape.js JSON
    cyjs_data = nx.cytoscape_data(G)
    cyjs_file = os.path.join(base_dir, f"{base_name}.cyjs")
    with open(cyjs_file, 'w') as f:
        json.dump(cyjs_data, f, indent=2)
    print(f"Cytoscape.js JSON saved: {cyjs_file}")
    
    # 6. Export as adjacency list
    adjlist_file = os.path.join(base_dir, f"{base_name}.adjlist")
    nx.write_adjlist(G, adjlist_file)
    print(f"Adjacency list saved: {adjlist_file}")
    
    print("\n" + "="*50)
    print("CYTOSCAPE IMPORT INSTRUCTIONS")
    print("="*50)
    print("\nOption 1 (RECOMMENDED): Import GraphML")
    print(f"  1. Open Cytoscape")
    print(f"  2. File → Import → Network from File")
    print(f"  3. Select: {graphml_file}")
    print(f"  4. All node attributes (including scores) will be imported automatically")
    
    print("\nOption 2: Import from CSV tables")
    print(f"  1. Open Cytoscape")
    print(f"  2. File → Import → Network from File")
    print(f"  3. Select: {edge_csv}")
    print(f"  4. Set 'source' as Source Node and 'target' as Target Node")
    print(f"  5. File → Import → Table from File")
    print(f"  6. Select: {node_csv}")
    print(f"  7. Import as Node Table, key column: 'sequence'")
    
    print("\nOption 3: Import Cytoscape.js JSON")
    print(f"  1. Open Cytoscape")
    print(f"  2. File → Import → Network from File")
    print(f"  3. Select: {cyjs_file}")
    print("="*50)

def export_node_attributes_detailed(G: nx.Graph, node_data: Dict, output_file: str):
    """
    Export detailed node attributes table for Cytoscape import.
    Includes additional network metrics.
    
    Args:
        G: NetworkX graph
        node_data: Dictionary mapping sequences to scores
        output_file: Path to save the CSV file
    """
    
    # Calculate various centrality measures
    degree_cent = nx.degree_centrality(G)
    
    # Only calculate these if graph is connected, otherwise do per component
    if nx.is_connected(G):
        betweenness_cent = nx.betweenness_centrality(G)
        closeness_cent = nx.closeness_centrality(G)
    else:
        betweenness_cent = {node: 0 for node in G.nodes()}
        closeness_cent = {node: 0 for node in G.nodes()}
    
    # Clustering coefficient
    clustering = nx.clustering(G)
    
    # Build detailed table
    node_table = []
    for node in G.nodes():
        # Get neighbors
        neighbors = list(G.neighbors(node))
        neighbor_scores = [node_data[n] for n in neighbors]
        
        node_table.append({
            'sequence': node,
            'function_score': node_data[node],
            'degree': G.degree(node),
            'degree_centrality': degree_cent[node],
            'betweenness_centrality': betweenness_cent[node],
            'closeness_centrality': closeness_cent[node],
            'clustering_coefficient': clustering[node],
            'num_neighbors': len(neighbors),
            'avg_neighbor_score': np.mean(neighbor_scores) if neighbor_scores else 0,
            'max_neighbor_score': max(neighbor_scores) if neighbor_scores else 0,
            'min_neighbor_score': min(neighbor_scores) if neighbor_scores else 0,
            'sequence_length': len(node)
        })
    
    df = pd.DataFrame(node_table)
    df.to_csv(output_file, index=False)
    print(f"Detailed node attributes saved: {output_file}")
    
    return df


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

    read_freq_cols = ['bin_0', 'bin_1', 'bin_2', 'bin_3']

    frequencies = [
        sum(sum(ast.literal_eval(row[col]) if isinstance(row[col], str) else row[col]) 
            for col in read_freq_cols)
        for idx, row in score_seq_df.iterrows()
    ]    

    # print(frequencies[:10])

     # Build network
    print("Building sequence network...")
    # G, node_data = build_sequence_network(sequences, function_scores)

    G, node_data, frequency_data = build_sequence_network_with_frequency(
    sequences, function_scores, frequencies
    )
    
    network_txt = dir_prefix + f'{expt_id}_network_stats.txt'
    
    # Analyze network
    analyze_network(G, node_data, network_txt)

    # Export for Cytoscape
    print("\nExporting network data for Cytoscape...")
    export_for_cytoscape(G, node_data, expt_id,
                        output_prefix=dir_prefix + f'{expt_id}_network')
    
    # Export detailed node attributes
    export_node_attributes_detailed(G, node_data,
                                    output_file=dir_prefix + f'{expt_id}_network_nodes_detailed.csv')

    layouts = ['spring', 'circular', 'kamada_kawai', 'spectral']

    # Visualize network in each network style

    for style in layouts:
        print(f"\nVisualizing network ({style})...")
        visualize_network(G, node_data, expt_id, 
                          layout=str(style),  # Try 'circular', 'kamada_kawai', 'spectral'
                          cmap='RdYlGn',    # Try 'viridis', 'plasma', 'coolwarm', 'RdYlGn'
                          size_by='frequency',  # This tells it to use frequency for sizing
                          min_node_size=100,
                          max_node_size=1000,
                          save_path = dir_prefix + f'{expt_id}_sequence_network_{style}.png')  # Set to None to not save
        

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