#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on January 29 08:24:51 2025
@author: alcantar
modifed by J. Hambalek on 18 July 2025
imported changes from jupyter notebook '09nb_...' 01 August 2025
example run 1: python 14_neutral_drift_analysis_input.py -i ../../minibinders_orthorep_data/minibinders_orthorep_outputs/jh_008/mutation_dfs/*mutation_analysis.csv -m ../../minibinders_orthorep_data/ngs_raw/jh_008/demultiplex/jh_008_metadata.txt -r ../../minibinders_orthorep_data/ngs_raw/jh_008/references/pmaa23_nbonly.fasta -c 25 -t 4 -f False
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
import re
import json

from Bio import SeqIO
import argparse

from utils_nd import *

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', nargs='+', help='input mutation analysis csv files')
    parser.add_argument('-m', help='path to metadata for declaration of csvs to bins')
    parser.add_argument('-r', help='Path to reference fasta')
    parser.add_argument('-c', help='counts per million cutoff used for generation of input csvs')
    parser.add_argument('-t', help='count threshold for filtering')
    parser.add_argument('-f', help='boolean value for applying score dissimilarity removal')
    parser.add_argument('-s', default = 'False', help='boolean value for choosing high or low replicate correlation for threshold stringency')

    args = parser.parse_args()

    ref_fasta_path = args.r #'../../minibinders_data/ngs_raw/jh_001/references/pjh3_nbonly.fasta' # input
    expt_id = ref_fasta_path.split('/')[-3] #experiment id is the entry past the 4th slash
    nanobody_id = ref_fasta_path.split('/')[-1].replace('_nbonly.fasta', '')
    print(f'experiment id: {expt_id}')

    out_dir = f'../../minibinders_orthorep_data/minibinders_orthorep_outputs/{expt_id}/neutral_drift_library/'
    make_dir(out_dir)
    out_dir_figs = out_dir+'plots/'
    make_dir(out_dir_figs)
    out_dir_reps = out_dir+'rep_scores/'
    make_dir(out_dir_reps)
    
    # define parent / best sequence
    ref_fasta = SeqIO.read(ref_fasta_path, "fasta")
    parent_dna_seq = ref_fasta.seq
    parent_dna_id = ref_fasta.id
    # define parent / best sequence

    cpm = args.c

    metadata_path = args.m
    metadata_df = pd.read_csv(metadata_path, sep='\t', index_col=None)
    # print(metadata_df)
    rep_total = max(metadata_df['replicate'])
    bin_total = max(metadata_df['bin'])
    print(f'{str(rep_total)} total replicates with {str(bin_total+1)} bins each')

    all_bin_seqs = args.i
    keyword = f'cpm{cpm}'
    # bin_seq_paths = list(filter(lambda x: x == f'*cpm{cpm}*', all_bin_seqs))
    bin_seq_paths = [path for path in all_bin_seqs if isinstance(path, str) and keyword in path]
    # print(bin_seq_paths)

    replicate_declare_dict = {}

    for rep in list(range(rep_total+1))[1:]:
        try:
            metadata_replicate = metadata_df[metadata_df['replicate'] == rep].copy()
            # print(f'replicate {rep}')
            # print(metadata_replicate)
            rep_bin_list = []
            # print(list(range(bin_total+1)))
            for binny in list(range(bin_total+1)):
                metadata_slice = metadata_replicate[metadata_replicate['bin'] == binny].copy()
                sample_name = metadata_slice['binder_name'].tolist()[0]
                # print(f'replicate {rep} bin {binny} has name {sample_name}')
                rep_bin_samp = [path2 for path2 in bin_seq_paths if isinstance(path2, str) and sample_name in path2]
                rep_bin_list.append(rep_bin_samp[0])
            replicate_declare_dict[rep] = rep_bin_list
        except Exception as e:
            print(f'replicate {rep} and bin {binny} not found in metadata! \n{e}')

    # print(replicate_declare_dict)

    count_threshold = int(args.t)
    reps_dict = {}
    # cell_counts = []
    for replicate in replicate_declare_dict.keys():

        # derive cell fractions by finding all cell counts associated with the replicate
        metadata_rep = metadata_df.copy()[metadata_df['replicate'] == replicate]
        cell_counts_rep = metadata_rep['cell_count'].tolist()
        sum_cell_counts = np.sum(cell_counts_rep)
        # print(cell_counts_rep)
        # print(sum_cell_counts)

        score_init = []
        
        bin_list = replicate_declare_dict[replicate]
        # print(bin_list)
        for scorenum, sample in enumerate(bin_list):
            bin_df = pd.read_csv(sample,index_col=0)
            bin_df_agg = aggregate_by_aa_sequence(bin_df)
            counts_dict = dict(zip(bin_df_agg['aa_sequence'],bin_df_agg['read_count']))
            sample_name = sample.split('/')[-1].replace(f'cpm{cpm}_mutation_analysis.csv','') #dependent on naming match from script 02
            # print(sample_name)
            metadata_samp = metadata_df.copy()[metadata_df['binder_name'].str.contains(sample_name)]
            # print(metadata_samp['cell_count'])
            bin_value = int(metadata_samp['bin'].item())
            cell_count_value = int(metadata_samp['cell_count'].item())
            cell_frac = cell_count_value/sum_cell_counts
            # print(cell_frac)
            # cell_counts.append(cell_count_value)
            # bin_df_agg = bin_df_agg.rename(columns={'read_count': f'bin_{bin_value}'},inplace=True)
            bin_df_agg[f'bin_{str(bin_value)}'] = bin_df_agg['aa_sequence'].map(counts_dict).fillna(0).astype(int)
            
            # debug module for determining mapping
            # debug_seq = []
            # for seq in bin_df_agg['aa_sequence']:
            #     if bin_df_agg[bin_df_agg['aa_sequence'] == seq][f'bin_{str(bin_value)}'].item() != bin_df_agg[bin_df_agg['aa_sequence'] == seq]['read_count'].item():
            #         debug_seq.append(bin_df_agg[seq].index)
            # print(f'{len(debug_seq)} sequences mapped incorrectly')
            
            # print(bin_df_agg[bin_df_agg[f'bin_{str(bin_value)}'] >= count_threshold])
            
            # test subset of sequences to figure out the issue
            idxs = list(range(len(bin_df_agg)))
            # idxs = list(range(8))
            
            # print(idxs[:3])
            for idx in idxs:
                if bin_df_agg.loc[idx, f'bin_{str(bin_value)}'] >= count_threshold:
                    score_init.append({
                        "aa_sequence": bin_df_agg.loc[idx, 'aa_sequence']
                    , "dna_sequence": bin_df_agg.loc[idx, 'dna_sequence']
                    , "aa_mutations": bin_df_agg.loc[idx, 'aa_mutations']
                    , "dna_mutations": bin_df_agg.loc[idx, 'dna_mutations']
                    , "number_aa_mutations": bin_df_agg.loc[idx, 'number_aa_mutations']
                    , "number_aa_mutations": bin_df_agg.loc[idx, 'number_aa_mutations']
                    , f"bin_{str(bin_value)}": bin_df_agg.loc[idx, f'bin_{str(bin_value)}']
                    , f"bin_{str(bin_value)}_norm": (bin_df_agg.loc[idx, f'bin_{str(bin_value)}']*cell_frac)})
        
                # print(score_init)

        rep_df = pd.DataFrame(score_init).fillna(0)
        # print(rep_df[:5])
        # Identify bin columns robustly
        bin_cols = [c for c in rep_df.columns if re.match(r'^bin_?\d+(?:_norm)?$', str(c))]
        # print(bin_cols)
        
        # Ensure numeric, treat NaN as 0
        rep_df[bin_cols] = rep_df[bin_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

        # Clean sequence key (avoid hidden dupes from whitespace/case)
        rep_df['aa_sequence'] = rep_df['aa_sequence'].astype(str).str.strip()

        # everything else (except the key)
        meta_cols = [c for c in rep_df.columns if c not in ['aa_sequence'] + bin_cols]

        agg_map = {**{c: 'sum' for c in bin_cols}, **{c: 'first' for c in meta_cols}}

        # Collapse: sum all bin columns per sequence
        rep_df_merged = rep_df.groupby('aa_sequence', as_index=False).agg(agg_map)

        bin_cols_norm = [c for c in rep_df.columns if re.match(r'^bin_?\d+(?:_norm)$', str(c))]
        
        # ensure numeric and drop sequences not read enough
        rep_df_merged[bin_cols] = rep_df_merged[bin_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

        bin_cols_raw = [c for c in bin_cols if c not in bin_cols_norm]

        read_total = rep_df_merged[bin_cols_raw].sum(axis=1)
        print(f'read total: {len(read_total)}')
        

        # Drop rows below threshold
        rep_df_merged = rep_df_merged.loc[read_total >= count_threshold].reset_index(drop=True)
        for binny in list(range(bin_total+1)):
            # rep_df_merged[f'bin_{binny}_norm'] = rep_df_merged[f'bin_{binny}_norm']/np.sum(cell_counts) # cell fraction normalization
            # print(int(np.sum(rep_df_merged[f'bin_{binny}'])))
            rep_df_merged[f'bin_{binny}_norm'] = rep_df_merged[f'bin_{binny}_norm']/int(np.sum(rep_df_merged[f'bin_{binny}'])) # read count normalization

        # row wise normalization


        rep_df_merged[bin_cols_norm] = rep_df_merged[bin_cols_norm].div(rep_df_merged[bin_cols_norm].sum(axis=1), axis=0)

        # add normqlized score
        # norm_counts = rep_df_merged[bin_cols_norm]
        bin_weights = [i/bin_total for i in range(bin_total+1)]
        # print(bin_weights)
        rep_df_merged['norm_score'] = rep_df_merged[bin_cols_norm].dot(np.asarray(bin_weights, dtype=float))

        # print(rep_df_merged)
        # print(len(rep_df), len(rep_df_merged))
        # print(rep_df_merged[rep_df_merged['aa_mutations'] == '[]'])
        reps_dict[f'replicate_{replicate}'] = rep_df_merged

        rep_out_name = out_dir_reps + f'replicate_{replicate}_{expt_id}_countthresh{count_threshold}' + '.csv'

        rep_df_merged.to_csv(rep_out_name)
    # print(reps_dict)

    # bringing all dfs together into one dataframe
    
    scored_df_long = pd.concat(
    [df.assign(rep_key=key) for key, df in reps_dict.items()],
    ignore_index=True
    )
    # print(scored_df_long[:5])
    scored_df_agg = scored_df_long.groupby('aa_sequence')[bin_cols].agg(list).reset_index()

    # check duplicates
    scored_df_dupes = scored_df_long.duplicated(subset=['aa_sequence', 'rep_key'], keep=False)

    # build dataframes compiling scores for a given sequence into one row by replicate
    scored_df_wide = scored_df_long.pivot(index = 'aa_sequence', columns='rep_key', values = 'norm_score').reset_index()

    scored_df_wide.fillna('na').to_csv(out_dir_reps + f'{expt_id}_allscores_countthresh{count_threshold}' + '.csv')
    
    scored_df_wide.columns.name = '.'

    # print(scored_df_reps[:4])
    rep_cols = list(reps_dict.keys())
    mask_all_present = scored_df_wide[rep_cols].notna().all(axis=1)  # keep only rows with no NaNs in rep_cols
    scored_df_reps = scored_df_wide.loc[mask_all_present].reset_index(drop=True)
    # print(scored_df_reps[:4])

    # add back the rest of the metadata
    scored_meta_cols = [col for col in scored_df_long.columns if col not in ['norm_score','rep_key'] + rep_cols + bin_cols]
    scored_meta = scored_df_long[scored_meta_cols].drop_duplicates('aa_sequence')
    scored_bins = scored_df_agg[['aa_sequence'] + bin_cols] \
        .drop_duplicates(subset='aa_sequence', keep='first') \
        .reset_index(drop=True)

    scored_df_all = scored_df_reps \
        .merge(scored_meta, on='aa_sequence', how ='inner') \
        .merge(scored_bins, on='aa_sequence', how ='inner')
    # print(scored_df_all[:5])

    # identify replicates with drastically different scores
    diff_reps = []
    diff_val = math.ceil(1/3*100)/100
    for k, row in tqdm.tqdm(scored_df_all.iterrows(), total = scored_df_all.shape[0]):
        replicate_diff = row[rep_cols].std() * 2
        if replicate_diff > diff_val:
            diff_reps.append(k)

    scored_df_uniform = scored_df_all.copy()
    apply_filt = args.f
    # print(apply_filt)
    if apply_filt.lower() == 'true':
        print("applying filtering")
        for k2 in diff_reps:
            scored_df_uniform = scored_df_uniform.drop(k2)
        filtered = True
    else:
        filtered = False
    print(f"identified {len(diff_reps)} sequences with dissimilar replicate scores (|diff| > {diff_val})")

    X = scored_df_uniform[rep_cols].values
    
    # Use KMeans clustering to group points into 3 categories
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    kmeans_labels = kmeans.fit_predict(X)
    
    # Map cluster labels to descriptive categories (assign based on mean locations)
    cluster_centers = kmeans.cluster_centers_
    sorted_indices = np.argsort(cluster_centers[:, 0])  # Sort clusters by normalized_score_rep1
    cluster_map = {sorted_indices[0]: 'Inactive',
                   sorted_indices[1]: 'Active',
                   sorted_indices[2]: 'Best Activity'}

    # Assign to dataframe
    scored_df_uniform['predicted_category'] = [cluster_map[label] for label in kmeans_labels]

    # Map categories to colors for visualization
    category_colors = {
        'Inactive': 'black',
        'Active': '#3B71B2',
        'Best Activity': 'orange'
    }
    scored_df_uniform['color'] = scored_df_uniform['predicted_category'].map(category_colors)

    # print(scored_df_uniform)
    out_name = out_dir + f'dyn_library_scores_{expt_id}_countthresh{count_threshold}' + ('filtered' if filtered==True else '') + '.csv'

    scored_df_uniform['dna_sequence'] = scored_df_uniform['dna_sequence'].apply(json.dumps)

    # # Read back and restore to list (if needed to process downstream load json
    # df2 = pd.read_csv('out.csv', converters={'seq_list': json.loads})
    # print(scored_df_uniform[scored_df_uniform['aa_mutations'] == '[]'])
    scored_df_uniform.drop('color', axis=1).to_csv(out_name)

    # initialize figure generation
    fig, ax = plt.subplots(figsize=(10, 8))
    comp_cols = [c for c in rep_cols if pd.api.types.is_numeric_dtype(scored_df_uniform[c])]
    corr = scored_df_uniform[comp_cols].corr(method='pearson')
    # ignore diagonal and duplicate pairs by masking upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    corr_masked = corr.mask(mask)

    stringency = args.s
    
    if stringency.lower() == 'true':
        # option to use min value for read count stringency
        corr_idx = corr_masked.stack().idxmin()
        print(corr_idx)
    else:
        # otherwise take find max value of correlations (default)
        corr_idx = corr_masked.stack().idxmax()
    
    best_i, best_j = corr_idx
    best_score = corr_masked.loc[best_i, best_j] 

    if stringency.lower() == 'true':
        # read out the lowest correlation (could be bad)
        print(f"Worst pair: ({best_i},{best_j}) (|pearson|={best_score:.3f})")
    else:
        # standard best print statement
        print(f"Best pair: ({best_i},{best_j}) (|pearson|={best_score:.3f})")
    
    # plot scatter of best reps
    ax.scatter(
        scored_df_uniform[best_i],
        scored_df_uniform[best_j],
        c=scored_df_uniform['color'],
        edgecolors='black',
        s=100,
        alpha=0.75,
        label="Predicted categories"
    )

    # Highlight parent DNA sequence
    parent_mask = scored_df_uniform['aa_mutations'] == '[]'

    ax.scatter(
        scored_df_uniform[best_i][parent_mask],
        scored_df_uniform[best_j][parent_mask],
        c='orange',
        edgecolors='black',
        s=750,
        linewidth=2,
        label="Parent DNA sequence"
    )
    
    # Get the range for the y=x line
    min_val = min(min(scored_df_uniform[best_i]), min(scored_df_uniform[best_j]))
    max_val = max(max(scored_df_uniform[best_i]), max(scored_df_uniform[best_j]))
    
    # Plot y=x line
    plt.plot([min_val, max_val], [min_val, max_val], 'b--', label='y=x')


        
    # Add labels, legend, and title
    ax.set_xlabel(f'Normalized Score {best_i}', fontsize=16)
    ax.set_ylabel(f'Normalized Score {best_j}', fontsize=16)
    ax.set_title(f'K-means clustering of FluA (F9) nanobody activity library {nanobody_id}', fontsize=16)
    ax.legend(fontsize=12, loc="best")

    textstr = (f"Pearson's r: {best_score:.2f}")
    plt.gca().text(0.01, 0.8, textstr, transform=plt.gca().transAxes, fontsize=16, verticalalignment='top')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    out_name_fig = out_dir_figs + f'dyn_library_scores_{expt_id}_countthresh{count_threshold}' + ('filtered' if filtered==True else '')
    
    # save plot
    plt.savefig(out_name_fig+'.png', dpi=400)
    # plt.savefig(out_name_fig+'.pdf', dpi=400)
    plt.savefig(out_name_fig+'.svg', dpi=400)
    plt.close()

    total_seqs = scored_df_uniform.shape[0]
    num_best_seqs = (scored_df_uniform['predicted_category'] == 'Best Activity').sum()
    num_active_seqs = (scored_df_uniform['predicted_category'] == 'Active').sum()
    active_best_seqs = num_best_seqs + num_active_seqs
    num_unactive_sequences = (scored_df_uniform['predicted_category'] == 'Inactive').sum()

    percent_best = num_best_seqs/total_seqs*100
    percent_active = num_active_seqs/total_seqs*100

    percent_best_active = (num_best_seqs + num_active_seqs) / total_seqs * 100

    percent_inactive = num_unactive_sequences / total_seqs * 100

    text_file_name = out_dir + 'results_summary.txt'
    with open(text_file_name, "w") as file:
        file.write(f"{percent_best:.2f}% of sequences ({num_best_seqs}/{total_seqs}) are Best Activity.\n")
        file.write(f"{percent_best_active:.2f}% of sequences ({num_best_seqs + num_active_seqs}/{total_seqs}) are active.\n")
        file.write(f"{percent_inactive:.2f}% of sequences ({num_unactive_sequences}/{total_seqs}) are inactive.")

    # plot hamming distances
    # plot all hamming
    # plot_hamming_distances(annotated_counts_df=scored_df_uniform,
    #                        out_name_1='hamming_vs_parent_all' ,
    #                        out_name_2='pairwise_hamming_all',
    #                        expt_id=expt_id)

    best_annotated_counts_df = scored_df_uniform.copy()[scored_df_uniform['predicted_category']=="Best Activity"]
    active_annotated_counts_df = scored_df_uniform.copy()[scored_df_uniform['predicted_category']=="Active"]
    inactive_annotated_counts_df = scored_df_uniform.copy()[scored_df_uniform['predicted_category']=="Inactive"]


    # plot_hamming_distances(annotated_counts_df=best_annotated_counts_df,
    #                        out_name_1='hamming_vs_parent_Best Activity' ,
    #                        out_name_2='pairwise_hamming_Best Activity',
    #                        expt_id=expt_id)
    # plot_hamming_distances(annotated_counts_df=active_annotated_counts_df,
    #                        out_name_1='hamming_vs_parent_active' ,
    #                        out_name_2='pairwise_hamming_active',
    #                        expt_id=expt_id)
    # plot_hamming_distances(annotated_counts_df=inactive_annotated_counts_df,
    #                        out_name_1='hamming_vs_parent_inactive' ,
    #                        out_name_2='pairwise_hamming_inactive',
    #                        expt_id=expt_id)
    # plot versus parent
    num_sequences = scored_df_uniform.shape[0]
    # Set the Seaborn style and context
    sns.set(style="white")

    # Set global font size and family
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'Arial'
    })
    # Set the Seaborn style
    sns.set_style("ticks")

    best_mean = np.mean(best_annotated_counts_df['number_aa_mutations'])
    best_median = np.median(best_annotated_counts_df['number_aa_mutations'])

    active_mean = np.mean(active_annotated_counts_df['number_aa_mutations'])
    active_median = np.median(active_annotated_counts_df['number_aa_mutations'])

    inactive_mean = np.mean(inactive_annotated_counts_df['number_aa_mutations'])
    inactive_median = np.median(inactive_annotated_counts_df['number_aa_mutations'])

    # histogram plots and KDE lines (match colors)
    fig, ax = plt.subplots(figsize=(7, 5))

    # Combined domain (mutations are non-negative integers)
    combined = [
        best_annotated_counts_df['number_aa_mutations'].dropna().to_numpy(),
        active_annotated_counts_df['number_aa_mutations'].dropna().to_numpy(),
        inactive_annotated_counts_df['number_aa_mutations'].dropna().to_numpy(),
    ]
    xmin = 0
    xmax = int(np.max([a.max() for a in combined if a.size]))  # observed max across groups

    sns.histplot(data=best_annotated_counts_df, x="number_aa_mutations",
                         binwidth=1,stat='density',
                         edgecolor='black', color='orange', linewidth=0.5,
                         alpha=0.3,
                         label=f"Best Activity (n={best_annotated_counts_df.shape[0]} [{best_mean:.1f}|{best_median:.1f}])")

    
    sns.histplot(data=active_annotated_counts_df, x="number_aa_mutations",
                         binwidth=1,stat='density',
                         edgecolor='black',color='#3B71B2', linewidth=0.5,
                         alpha=0.3,
                         label=f"Active (n={active_annotated_counts_df.shape[0]} [{active_mean:.1f}|{active_median:.1f}])")


    sns.histplot(data=inactive_annotated_counts_df, x="number_aa_mutations",
                         binwidth=1,stat='density',
                         edgecolor='black', color='grey',linewidth=0.5,
                         alpha=0.3,
                         label=f"Inactive (n={inactive_annotated_counts_df.shape[0]}[{inactive_mean:.1f}|{inactive_median:.1f}])")
    
    
    sns.kdeplot(data=best_annotated_counts_df, x="number_aa_mutations",
                color='orange', lw=1.5, fill=False, cut=0, clip=(xmin, xmax), bw_adjust=1.5, ax=ax, zorder=5, label='Best Activity KDE')
    
    sns.kdeplot(data=active_annotated_counts_df, x="number_aa_mutations",
                color='#3B71B2', lw=1.5, fill=False, cut=0, clip=(xmin, xmax), bw_adjust=1.5, ax=ax, zorder=5, label='Active KDE')
    
    sns.kdeplot(data=inactive_annotated_counts_df, x="number_aa_mutations",
                color='grey', lw=1.5, fill=False, cut=0, clip=(xmin, xmax), bw_adjust=1.5, ax=ax, zorder=5, label='Inactive KDE')



    # ax = plt.gca()
    # Adjust the ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ticks = ax.get_xticks()
    if len(ticks) > 1:
        new_ticks = ticks[:-1] + 0.5
        ax.set_xticks(new_ticks)
        ax.set_xticklabels([str(int(t)) for t in new_ticks])
        ax.set_xlim([new_ticks[0] + 0.5, new_ticks[-1] - 0.5])
    
    ax.set_xlabel('amino acid mutations')
    ax.set_ylabel(f'density (n={num_sequences} sequences)')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(0.5)
    ax.tick_params(width=0.5)
    
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    outname_hamming_dir = f'../../minibinders_orthorep_data/minibinders_orthorep_outputs/{expt_id}/neutral_drift_library/plots/'
    outname_histogram_all_classes = outname_hamming_dir + 'hamming_all_classes' + f'_countthresh{count_threshold}' + ('filtered' if filtered==True else '')

    
    plt.savefig(outname_histogram_all_classes + '.png', dpi=400)
    # plt.savefig(outname_histogram_all_classes + '.pdf', dpi=400)
    plt.savefig(outname_histogram_all_classes + '.svg', dpi=400)
    plt.close()
    
if __name__ == "__main__":
    main()
