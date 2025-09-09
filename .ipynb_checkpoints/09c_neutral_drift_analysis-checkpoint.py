#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on January 29 08:24:51 2025
@author: alcantar
modifed by J. Hambalek on 18 July 2025
example run 1: python 09_neutral_drift_analysis.py -r ../../minibinders_orthorep_data/ngs_raw/jh_001/references/pjh3_nbonly.fasta -c 50
"""
# activate virtual enviroment before running script
# source activate minibinders

import pandas as pd
import tqdm
import math
from matplotlib import pyplot as plt
import scipy.stats as stats
from matplotlib.lines import Line2D
import numpy as np
from sklearn.cluster import KMeans
from matplotlib.ticker import MaxNLocator
from itertools import combinations
import seaborn as sns

from Bio import SeqIO
import argparse

from utils_nd import *

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', help='Path to reference fasta')
    parser.add_argument('-c', help='counts per million cutoff used for generation of input dataframes')

    args = parser.parse_args()

    ref_fasta_path = args.r #'../../minibinders_data/ngs_raw/jh_001/references/pjh3_nbonly.fasta' # input
    expt_id = ref_fasta_path.split('/')[-3] #experiment id is the entry past the 4th slash
    print(f'experiment id: {expt_id}')

    cpm = args.c
    
    # define parent / wt sequence
    ref_fasta = SeqIO.read(ref_fasta_path, "fasta")
    parent_dna_seq = ref_fasta.seq
    parent_dna_id = ref_fasta.id
    # define parent / wt sequence

    # read in relevant dataframes (manually typed in here)
    med_rep1_path = f'../../minibinders_orthorep_data/minibinders_orthorep_outputs/{expt_id}/mutation_dfs/19mdbindcpm{cpm}_mutation_analysis.csv'
    med_rep2_path = f'../../minibinders_orthorep_data/minibinders_orthorep_outputs/{expt_id}/mutation_dfs/22mdbindcpm{cpm}_mutation_analysis.csv'

    low_rep1_path = f'../../minibinders_orthorep_data/minibinders_orthorep_outputs/{expt_id}/mutation_dfs/19lobindcpm{cpm}_mutation_analysis.csv'
    low_rep2_path = f'../../minibinders_orthorep_data/minibinders_orthorep_outputs/{expt_id}/mutation_dfs/22lobindcpm{cpm}_mutation_analysis.csv'

    hi_rep1_path = f'../../minibinders_orthorep_data/minibinders_orthorep_outputs/{expt_id}/mutation_dfs/19hibindcpm{cpm}_mutation_analysis.csv'
    hi_rep2_path = f'../../minibinders_orthorep_data/minibinders_orthorep_outputs/{expt_id}/mutation_dfs/22hibindcpm{cpm}_mutation_analysis.csv'

    no_rep1_path = f'../../minibinders_orthorep_data/minibinders_orthorep_outputs/{expt_id}/mutation_dfs/19nobindcpm{cpm}_mutation_analysis.csv'
    no_rep2_path = f'../../minibinders_orthorep_data/minibinders_orthorep_outputs/{expt_id}/mutation_dfs/22nobindcpm{cpm}_mutation_analysis.csv'

    # display_rep1_path = '../../minibinders_orthorep_data/minibinders_orthorep_outputs/maa_003/mutation_dfs/mb376-drift-displaypos-rep1_mutation_analysis.csv'
    # display_rep2_path = '../../minibinders_orthorep_data/minibinders_orthorep_outputs/maa_003/mutation_dfs/mb376-drift-displaypos-rep2_mutation_analysis.csv'

    out_dir = f'../../minibinders_orthorep_data/minibinders_orthorep_outputs/{expt_id}/neutral_drift_library/'
    make_dir(out_dir)
    out_dir_figs = out_dir+'plots/'
    make_dir(out_dir_figs)

    med_rep1_df = pd.read_csv(med_rep1_path, index_col=0)
    med_rep2_df = pd.read_csv(med_rep2_path, index_col=0)

    low_rep1_df = pd.read_csv(low_rep1_path, index_col=0)
    low_rep2_df = pd.read_csv(low_rep2_path, index_col=0)

    hi_rep1_df = pd.read_csv(hi_rep1_path, index_col=0)
    hi_rep2_df = pd.read_csv(hi_rep2_path, index_col=0)

    no_rep1_df = pd.read_csv(no_rep1_path, index_col=0)
    no_rep2_df = pd.read_csv(no_rep2_path, index_col=0)

    # display_rep1_df = pd.read_csv(display_rep1_path, index_col=0)
    # display_rep2_df = pd.read_csv(display_rep2_path, index_col=0)

    # consilidate by amino acid sequence
    med_rep1_df_agg = aggregate_by_aa_sequence(med_rep1_df)
    med_rep2_df_agg = aggregate_by_aa_sequence(med_rep2_df)

    low_rep1_df_agg = aggregate_by_aa_sequence(low_rep1_df)
    low_rep2_df_agg = aggregate_by_aa_sequence(low_rep2_df)

    hi_rep1_df_agg = aggregate_by_aa_sequence(hi_rep1_df)
    hi_rep2_df_agg = aggregate_by_aa_sequence(hi_rep2_df)

    no_rep1_df_agg = aggregate_by_aa_sequence(no_rep1_df)
    no_rep2_df_agg = aggregate_by_aa_sequence(no_rep2_df)

    # display_rep1_df = aggregate_by_aa_sequence(display_rep1_df)
    # display_rep2_df = aggregate_by_aa_sequence(display_rep2_df)

    # create master dataframe with all sequences that will be considered
    # these sequences will be filtered using a read count threshold
    master_rep1_df = pd.concat([med_rep1_df_agg,
                               low_rep1_df_agg,
                               hi_rep1_df_agg,
                               no_rep1_df_agg]).reset_index(drop=True).drop(['read_count'], axis=1)

    master_rep2_df = pd.concat([med_rep2_df_agg,
                               low_rep2_df_agg,
                               hi_rep2_df_agg,
                               no_rep2_df_agg]).reset_index(drop=True).drop(['read_count'], axis=1)

    # drop duplicate rows with the same amino acid sequence
    master_rep1_df = master_rep1_df.drop_duplicates(subset=['aa_sequence'],).reset_index(drop=True)
    master_rep2_df = master_rep2_df.drop_duplicates(['aa_sequence']).reset_index(drop=True)


    # populate master_dfs with the read counts and also apply read count threshold filter
    count_threshold = 0

    master_rep1_df = master_rep1_df.assign(bin_1=0, bin_2=0, bin_3=0, bin_4=0)
    master_rep2_df = master_rep2_df.assign(bin_1=0, bin_2=0, bin_3=0, bin_4=0)

    no_rep1_counts_dict = dict(zip(no_rep1_df_agg['aa_sequence'],no_rep1_df_agg['read_count']))
    no_rep2_counts_dict = dict(zip(no_rep2_df_agg['aa_sequence'],no_rep2_df_agg['read_count']))

    low_rep1_counts_dict = dict(zip(low_rep1_df_agg['aa_sequence'],low_rep1_df_agg['read_count']))
    low_rep2_counts_dict = dict(zip(low_rep2_df_agg['aa_sequence'],low_rep2_df_agg['read_count']))

    med_rep1_counts_dict = dict(zip(med_rep1_df_agg['aa_sequence'],med_rep1_df_agg['read_count']))
    med_rep2_counts_dict = dict(zip(med_rep2_df_agg['aa_sequence'],med_rep2_df_agg['read_count']))

    hi_rep1_counts_dict = dict(zip(hi_rep1_df_agg['aa_sequence'],hi_rep1_df_agg['read_count']))
    hi_rep2_counts_dict = dict(zip(hi_rep2_df_agg['aa_sequence'],hi_rep2_df_agg['read_count']))

    raw_counts_rep1_df = retrieve_filter_counts(master_rep1_df,
                               no_rep1_counts_dict,
                               low_rep1_counts_dict,
                               med_rep1_counts_dict,
                               hi_rep1_counts_dict,
                               count_threshold=count_threshold)

    raw_counts_rep2_df = retrieve_filter_counts(master_rep2_df,
                               no_rep2_counts_dict,
                               low_rep2_counts_dict,
                               med_rep2_counts_dict,
                               hi_rep2_counts_dict,
                               count_threshold=count_threshold)


    # calculate cell fractions in each bin
    rep1_cell_counts = [114000, 27000, 46000, 299000]
    rep1_cell_fracs = [count/sum(rep1_cell_counts) for count in rep1_cell_counts]

    rep2_cell_counts = [62813, 20928, 30941, 162571]
    rep2_cell_fracs = [count/sum(rep2_cell_counts) for count in rep2_cell_counts]

    # create new dataframes that will contain normalized scores
    first_n_columns = 6
    norm_counts_rep1_df = raw_counts_rep1_df.copy().iloc[:, :first_n_columns]
    norm_counts_rep2_df = raw_counts_rep2_df.copy().iloc[:, :first_n_columns]

    # create dataframes which only contain read counts and will be used to calculate normalized scores
    raw_counts_only_rep1_df = raw_counts_rep1_df.copy().iloc[:,-4:]
    raw_counts_only_rep2_df = raw_counts_rep2_df.copy().iloc[:,-4:]

    # apply sort-seq-esque normalization to both replicates

    # compute total number of reads per bin (per concentration)
    column_sums_rep1 = raw_counts_only_rep1_df.sum().tolist()
    for bn, bin_no in enumerate(list(raw_counts_only_rep1_df.columns)):
        # normalize by number of reads and fraction of cells in each bin
        norm_counts_rep1_df[bin_no] = raw_counts_only_rep1_df[bin_no] * rep1_cell_fracs[bn]/ column_sums_rep1[bn]
    # normalize each row by the sum of each row
    norm_counts_rep1_df.iloc[:, -4:] = (
    norm_counts_rep1_df.iloc[:, -4:]
    .div(norm_counts_rep1_df.iloc[:, -4:].sum(axis=1), axis=0)
    )

    norm_counts_rep1_df['normalized_score_rep1'] = (norm_counts_rep1_df['bin_1']*0 \
                                                                           + norm_counts_rep1_df['bin_2']*(1/3) \
                                                                           + norm_counts_rep1_df['bin_3']*(2/3) \
                                                                           + norm_counts_rep1_df['bin_4']*(3/3))

    # compute total number of reads per bin (per concentration)
    column_sums_rep2 = raw_counts_only_rep2_df.sum().tolist()
    for bn, bin_no in enumerate(list(raw_counts_only_rep2_df.columns)):
        # normalize by number of reads and fraction of cells in each bin
        norm_counts_rep2_df[bin_no] = raw_counts_only_rep2_df[bin_no] * rep2_cell_fracs[bn]/ column_sums_rep2[bn]
    # normalize each row by the sum of each row
    norm_counts_rep2_df.iloc[:, -4:] = (
    norm_counts_rep2_df.iloc[:, -4:]
    .div(norm_counts_rep2_df.iloc[:, -4:].sum(axis=1), axis=0)
    )

    norm_counts_rep2_df['normalized_score_rep2'] = (norm_counts_rep2_df['bin_1']*0 \
                                                                           + norm_counts_rep2_df['bin_2']*(1/3) \
                                                                           + norm_counts_rep2_df['bin_3']*(2/3) \
                                                                           + norm_counts_rep2_df['bin_4']*(3/3))

    # merge both replicates for amino acid sequences that appear in both replicates
    # Get the last column of norm_counts_rep2_df
    last_column = norm_counts_rep2_df[["aa_sequence", "normalized_score_rep2"]]

    # Merge norm_counts_rep1_df with the last column from norm_counts_rep2_df based on dna_sequence
    norm_counts_df = norm_counts_rep1_df.copy().merge(last_column, on="aa_sequence", how="inner")



    # remove highly variant sequences differing by 1/3 a.u. -- note that
    # differing by 1/3 a.u. means that, on average, the bins
    # were sorted into different bins
    for idx, row in tqdm.tqdm(norm_counts_df.iterrows(), total = norm_counts_df.shape[0]):
        replicate_difference = abs(row['normalized_score_rep1'] - row['normalized_score_rep2'])
        if replicate_difference >= (1/3):
            norm_counts_df = norm_counts_df.drop(idx)

    norm_counts_annotated_df = norm_counts_df.copy()

    # Prepare feature matrix (normalized scores)
    X = norm_counts_annotated_df[['normalized_score_rep1', 'normalized_score_rep2']].values

    # Use KMeans clustering to group points into 3 categories
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    kmeans_labels = kmeans.fit_predict(X)

    # Map cluster labels to descriptive categories (assign based on mean locations)
    cluster_centers = kmeans.cluster_centers_
    sorted_indices = np.argsort(cluster_centers[:, 0])  # Sort clusters by normalized_score_rep1
    cluster_map = {sorted_indices[0]: 'Inactive',
                   sorted_indices[1]: 'Active',
                   sorted_indices[2]: 'WT-like'}

    # Assign labels to dataframe
    norm_counts_annotated_df['predicted_category'] = [cluster_map[label] for label in kmeans_labels]

    # Map categories to colors for visualization
    category_colors = {
        'Inactive': 'black',
        'Active': '#3B71B2',
        'WT-like': 'orange'
    }
    norm_counts_annotated_df['color'] = norm_counts_annotated_df['predicted_category'].map(category_colors)

    # Plot the results
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot with predicted colors
    ax.scatter(
        norm_counts_annotated_df['normalized_score_rep1'],
        norm_counts_annotated_df['normalized_score_rep2'],
        c=norm_counts_annotated_df['color'],
        edgecolors='black',
        s=100,
        alpha=0.75,
        label="Predicted categories"
    )

    # Highlight parent DNA sequence
    # parent_mask = norm_counts_annotated_df['dna_sequence'] == parent_dna_seq
    parent_mask = norm_counts_annotated_df['dna_sequence'].apply(lambda seq_list: parent_dna_seq in seq_list)

    ax.scatter(
        norm_counts_annotated_df['normalized_score_rep1'][parent_mask],
        norm_counts_annotated_df['normalized_score_rep2'][parent_mask],
        c='orange',
        edgecolors='black',
        s=750,
        linewidth=2,
        label="Parent DNA sequence"
    )

    # Add labels, legend, and title
    ax.set_xlabel('Normalized Score Replicate 1', fontsize=16)
    ax.set_ylabel('Normalized Score Replicate 2', fontsize=16)
    ax.set_title(f'K-means clustering of FluA (F9) nanobody activity library {expt_id}', fontsize=16)
    ax.legend(fontsize=12, loc="best")

    rep1_scores = norm_counts_annotated_df['normalized_score_rep1']
    rep2_scores = norm_counts_annotated_df['normalized_score_rep2']
    correlation_coefficient, p_value = stats.pearsonr(rep1_scores, rep2_scores)
    correlation_coefficient_spear, p_value_spear = stats.spearmanr(rep1_scores, rep2_scores)

    textstr = (f"Pearson's r: {correlation_coefficient:.2f}, p = {p_value:.2e}\n"
               f"Spearman's r: {correlation_coefficient_spear:.2f}, p = {p_value_spear:.2e}")
    plt.gca().text(0.01, 0.8, textstr, transform=plt.gca().transAxes, fontsize=16, verticalalignment='top')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    out_name = out_dir + f'library_scores_{expt_id}_c.csv'
    out_name_fig = out_dir_figs + f'library_scores_{expt_id}'

    # save plot
    plt.savefig(out_name_fig+'.png', dpi=400)
    plt.savefig(out_name_fig+'.pdf', dpi=400)
    # plt.savefig(out_name_fig+'.svg', dpi=400)
    plt.close()
    # save dataframe
    norm_counts_annotated_df.drop('color', axis=1).to_csv(out_name)

    total_seqs = norm_counts_annotated_df.shape[0]
    num_wt_like_seqs = (norm_counts_annotated_df['predicted_category'] == 'WT-like').sum()
    num_active_seqs = (norm_counts_annotated_df['predicted_category'] == 'Active').sum()
    active_wt_seqs = num_wt_like_seqs + num_active_seqs
    num_unactive_sequences = (norm_counts_annotated_df['predicted_category'] == 'Inactive').sum()

    percent_wt = num_wt_like_seqs/total_seqs*100
    percent_active = num_active_seqs/total_seqs*100

    percent_wt_active = (num_wt_like_seqs + num_active_seqs) / total_seqs * 100

    percent_inactive = num_unactive_sequences / total_seqs * 100

    text_file_name = out_dir + 'results_summary.txt'
    with open(text_file_name, "w") as file:
        file.write(f"{percent_wt:.2f}% of sequences ({num_wt_like_seqs}/{total_seqs}) are WT-like.\n")
        file.write(f"{percent_wt_active:.2f}% of sequences ({num_wt_like_seqs + num_active_seqs}/{total_seqs}) are active.\n")
        file.write(f"{percent_inactive:.2f}% of sequences ({num_unactive_sequences}/{total_seqs}) are inactive.")

    # plot hamming distances
    # plot all hamming
    plot_hamming_distances(annotated_counts_df=norm_counts_annotated_df,
                           out_name_1='hamming_vs_parent_all' ,
                           out_name_2='pairwise_hamming_all',
                           expt_id=expt_id)

    wt_annotated_counts_df = norm_counts_annotated_df.copy()[norm_counts_annotated_df['predicted_category']=="WT-like"]
    active_annotated_counts_df = norm_counts_annotated_df.copy()[norm_counts_annotated_df['predicted_category']=="Active"]
    inactive_annotated_counts_df = norm_counts_annotated_df.copy()[norm_counts_annotated_df['predicted_category']=="Inactive"]


    plot_hamming_distances(annotated_counts_df=wt_annotated_counts_df,
                           out_name_1='hamming_vs_parent_WT-like' ,
                           out_name_2='pairwise_hamming_WT-like',
                           expt_id=expt_id)
    plot_hamming_distances(annotated_counts_df=active_annotated_counts_df,
                           out_name_1='hamming_vs_parent_active' ,
                           out_name_2='pairwise_hamming_active',
                           expt_id=expt_id)
    plot_hamming_distances(annotated_counts_df=inactive_annotated_counts_df,
                           out_name_1='hamming_vs_parent_inactive' ,
                           out_name_2='pairwise_hamming_inactive',
                           expt_id=expt_id)
    # plot versus parent
    num_sequences = norm_counts_annotated_df.shape[0]
    # Set the Seaborn style and context
    sns.set(style="white")

    # Set global font size and family
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'Arial'
    })
    # Set the Seaborn style
    sns.set_style("ticks")

    wt_mean = np.mean(wt_annotated_counts_df['number_aa_mutations'])
    wt_median = np.median(wt_annotated_counts_df['number_aa_mutations'])

    active_mean = np.mean(active_annotated_counts_df['number_aa_mutations'])
    active_median = np.median(active_annotated_counts_df['number_aa_mutations'])

    inactive_mean = np.mean(inactive_annotated_counts_df['number_aa_mutations'])
    inactive_median = np.median(inactive_annotated_counts_df['number_aa_mutations'])

    g = sns.histplot(data=wt_annotated_counts_df, x="number_aa_mutations",
                         binwidth=1,stat='probability',
                         edgecolor='black', color='orange', linewidth=0.5,
                         alpha=0.5,
                         label=f"WT-like (n={wt_annotated_counts_df.shape[0]} [{wt_mean:.1f}|{wt_median:.1f}])")
    sns.histplot(data=active_annotated_counts_df, x="number_aa_mutations",
                         binwidth=1,stat='probability',
                         edgecolor='black',color='#3B71B2', linewidth=0.5,
                         alpha=0.5,
                         label=f"Active (n={active_annotated_counts_df.shape[0]} [{active_mean:.1f}|{active_median:.1f}])")
    sns.histplot(data=inactive_annotated_counts_df, x="number_aa_mutations",
                         binwidth=1,stat='probability',
                         edgecolor='black', color='grey',linewidth=0.5,
                         alpha=0.5,
                         label=f"Inactive (n={inactive_annotated_counts_df.shape[0]}[{inactive_mean:.1f}|{inactive_median:.1f}])")
    ax = plt.gca()
    # Adjust the ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ticks = g.get_xticks()
    new_ticks = ticks[:-1] + 0.5
    g.set_xticks(new_ticks)
    g.set_xticklabels([str(int(t)) for t in new_ticks])

    g.set_xlabel('amino acid mutations')
    g.set_ylabel(f'frequency (n={num_sequences} sequences)')
    g.spines['right'].set_visible(False)
    g.spines['top'].set_visible(False)
    for spine in ['left', 'bottom']:
        g.spines[spine].set_linewidth(0.5)
    g.tick_params(width=0.5)
    g.set_xlim([new_ticks[0]+0.5, new_ticks[-1]-0.5])
    plt.legend(loc='upper left')
    outname_hamming_dir = f'../../minibinders_orthorep_data/minibinders_orthorep_outputs/{expt_id}/neutral_drift_library/plots/'
    outname_histogram_all_classes =outname_hamming_dir + 'hamming_all_classes'
    plt.savefig(outname_histogram_all_classes + '.png', dpi=400)
    plt.savefig(outname_histogram_all_classes + '.pdf', dpi=400)
    # plt.savefig(outname_histogram_all_classes + '.svg', dpi=400)
    plt.close()
if __name__ == "__main__":
    main()
