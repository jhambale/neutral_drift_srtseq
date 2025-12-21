#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on December 08 14:31 2025
@author: @jhambale
example run 1: python 19_network_rep.py -i ../../minibinders_orthorep_data/minibinders_orthorep_outputs/jh_008/neutral_drift_library/*scores*.csv -r ../../minibinders_orthorep_data/ngs_raw/jh_008/references/pmaa23_nbonly.fasta -c 50
"""

import pandas as pd
import ast
import pysam
from Bio import SeqIO
import glob
import math
import numpy as np
from itertools import combinations
import seaborn as sns
import matplotlib as mpl
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from histogramplotter import *
import os
import scipy.stats as stats
from scipy.stats import shapiro
from utils_nd import *

import argparse

def plot_seqavgs(dir_prefix, library_org_df, fasta_name, org_outpath):
    fig, ax = plt.subplots(figsize=(8,6))
    colors_org = sns.color_palette("husl", len(library_org_df))
    wtdistances_org = []
    high_var = []
    plt.figure(figsize=(20,6))
    for seq in range(len(library_org_df)):
        score_avg = library_org_df['average normalized score'].iloc[seq]
        score_err = library_org_df['normalized score error'].iloc[seq]
        plt.plot(seq, score_avg, marker = 'o', ms=4, color=colors_org[seq])
        plt.errorbar(seq, score_avg, yerr=score_err, capsize=4, elinewidth=0.5, alpha=0.5, color=colors_org[seq])
        wtdistances_org.append(library_org_df['number_aa_mutations'].iloc[seq])
        if score_err > (math.ceil(1/3*100)/100):
            high_var.append(seq)
        
    plt.xlabel('Sequence')
    plt.ylabel('Average Binding Score')
    plt.title(f'Average Binding Scores for sequences in {fasta_name}')
    # plt.legend(loc = 'upper right', bbox_to_anchor=(1.4, 1), borderaxespad=0, frameon=False)
    plt.ylim(-0.1,1.1)
    plt.xlim(-0.5,None)

    xmino, xmaxo = plt.xlim()
    linedist_org = []
    linedist_org.append(0)
    for disto in range(len(wtdistances_org[:-1])):
        if wtdistances_org[disto] != wtdistances_org[disto+1]:
            nextdisto = wtdistances_org[disto+1] - wtdistances_org[disto] # will determine the number of lines plotted
            for lineo in range(nextdisto):
                linexo = (disto+1.5)+(lineo/nextdisto * 0.4) # max line occupancy is within 0.1 of the next context
                plt.axvline(linexo, color = 'k', linestyle='--', linewidth=0.5)
                linedist_org.append(linexo)
            plt.text((linedist_org[-1]+linedist_org[-(int(nextdisto)+1)])/2, -0.05, f"{wtdistances_org[disto]} aa from wild type", ha='center', va='bottom', rotation = 90, alpha = 0.3, size = 8)
    plt.text((xmaxo+linedist_org[-1])/2, -0.05, f"{wtdistances_org[-1]} aa from wild type", ha='center', va='bottom', rotation = 90, alpha = 0.3, size = 8)                  
    plt.tight_layout()
    plt.savefig(org_outpath, bbox_inches='tight', dpi=400)
    print(f"averages figure saved at {org_outpath}")
    # plt.show()
    plt.close()

    return high_var

def plot_mutation_contexts(pmutslice, colors, muti, mutidist_outpath):
    fig, ax = plt.subplots(figsize=(8,6))
    wtdistances = []
    for row in range(len(pmutslice)):
        if pmutslice['delta score'].iloc[row] == pmutslice['score2'].iloc[row] - pmutslice['score1'].iloc[row]:
            plt.plot(row+1, pmutslice['score1'].iloc[row], marker = 'o', ms=5, color=colors[row])
            plt.plot(row+1, pmutslice['score2'].iloc[row], marker = 'x', ms=5, color=colors[row])
        else:
            plt.plot(row+1, pmutslice['score2'].iloc[row], marker = 'o', ms=5, color=colors[row])
            plt.plot(row+1, pmutslice['score1'].iloc[row], marker = 'x', ms=5, color=colors[row])
        wtdistances.append(pmutslice['wild type distance'].iloc[row])
    
    plt.plot([], [], marker='o', color='k', label='wild type residue')
    plt.plot([], [], marker='x', color='k', label='mutated residue')

    plt.xlabel('Context')
    plt.ylabel('Binding Score')
    plt.title(f'Actual Binding Scores for Contexts with {muti}')
    plt.legend(loc = 'upper right', bbox_to_anchor=(1.4, 1), borderaxespad=0, frameon=False)
    plt.ylim(-0.1,1.1)
    plt.xlim(0.5,None)
    xmin, xmax = plt.xlim()
    linedist = []
    linedist.append(0)
    for dist in range(len(wtdistances[:-1])):
        if wtdistances[dist] != wtdistances[dist+1]:
            nextdist = wtdistances[dist+1] - wtdistances[dist] # will determine the number of lines plotted
            # print(f"num lines: {nextdist}")
            for line in range(nextdist):
                linex = (dist+1.5)+(line/nextdist * 0.4) # max line occupancy is within 0.1 of the next context
                plt.axvline(linex, color = 'k', linestyle='--', linewidth=0.5)
                linedist.append(linex)
            # print(linedist)
            plt.text((linedist[-1]+linedist[-(int(nextdist)+1)])/2, -0.05, f"{wtdistances[dist]} aa from wild type", ha='center', va='bottom', rotation = 90, alpha = 0.3, size = 8)
    plt.text((xmax+linedist[-1])/2, -0.05, f"{wtdistances[-1]} aa from wild type", ha='center', va='bottom', rotation = 90, alpha = 0.3, size = 8)                  
    # plt.tight_layout()
    plt.savefig(mutidist_outpath, bbox_inches='tight', dpi=400)
    print(f"context figure saved at {mutidist_outpath}")
    # plt.show()
    plt.close()

def plot_context_distances(hamming_dist_contexts, mutk, contexts_dict, mutkdist_outpath):
    fig, ax = plt.subplots(figsize=(8,6))
    g = sns.histplot(data=hamming_dist_contexts, x=hamming_dist_contexts[mutk],
                        binwidth=1,stat='probability',
                         edgecolor='black', linewidth=0.5)
    ax = plt.gca()
    g.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    ticks = ax.get_xticks()
    new_ticks = ticks[:-1] + 0.5
    g.set_xticks(new_ticks)
    g.set_xticklabels([str(int(t)) for t in new_ticks])
    
    g.set_xlabel('aa mutations')
    g.set_ylabel(f'frequency (n={len(contexts_dict[mutk])} sequences)')
    g.set_title(f"context distances for {mutk} point mutation")
    g.spines['right'].set_visible(False)
    g.spines['top'].set_visible(False)
    for spine in ['left', 'bottom']:
        g.spines[spine].set_linewidth(0.5)
    g.tick_params(width=0.5)
    g.set_xlim([new_ticks[0]+0.5, new_ticks[-1]-0.5])
    # Show the plot
    plt.savefig(mutkdist_outpath, dpi=400)
    print(f"distance figure saved at {mutkdist_outpath}")
    # plt.show()
    plt.close()


def plot_context_ridges(sorted_mut_dict, ridgedata_df, fasta_name, ridge_outpath):
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    
    order = (
        ridgedata_df.groupby("mutation")["dscore"]
        .median()
        .sort_values(ascending=False)
        .index
    )
    
    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(n_colors=len(order), start=.5, rot=-0.75, light=0.75, dark=0.25)
    # Create a palette dictionary mapping each sample to a color
    # palette = sns.color_palette("tab20", n_colors=len(order))  # or your favorite palette
    palette_dict = dict(zip(order, pal))
    
    g = sns.FacetGrid(ridgedata_df, row="mutation", hue="mutation", aspect=10, height=1, palette=palette_dict, row_order=order)
    
    # Draw the densities in a few steps
    g.map(sns.kdeplot, "dscore",
          bw_adjust=0.5, clip_on=False,
          fill=True, alpha=1, linewidth=1, warn_singular=False)
    
    # Add a white line at the bottom of each plot
    g.map(plt.axhline, y=0, lw=2, clip_on=False)
    # g.map(plt.axvspan(-0.2, 0.2, color='gray', alpha=0.15))
    
    g.map(sns.kdeplot, "dscore", clip_on=False, color="white", lw=2, bw_adjust=0.5)
    
    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)
    
    
    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)
    
    
    g.map(label, "dscore")
    
    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.3)
    
    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[],ylabel="")
    g.set_xlabels("Binding Score Change", fontsize=14) 
    g.despine(bottom=True, left=True)


    plt.suptitle(f"Point Mutations Binding Score Change ({fasta_name})", fontsize=16, y=0.95)  # adjust y as needed
    
    plt.savefig(ridge_outpath, format='png', dpi=500, bbox_inches='tight')
    print(f"ridge figure saved at {ridge_outpath}")
    # plt.show()
    plt.close()


def plot_dms(heatmap_matrix, fasta_name, heatmap_outpath):
    plt.rcParams.update({
        'font.size': 25,
        # 'font.family': 'Arial'
    })
    
    plt.figure(figsize=(len(heatmap_matrix.columns)/1.5, 40))
    ax = sns.heatmap(
        heatmap_matrix,
        annot=True, fmt=".2f",
        cmap='coolwarm', center=0,
        linewidths=0.5, linecolor='gray',
        cbar_kws={'label': 'Score'},
        annot_kws={"fontsize":25,"rotation":90}
    )
    ax.set_xticklabels(heatmap_matrix.columns, rotation=90, fontsize=20)
    ax.set_yticklabels(heatmap_matrix.index, rotation=0, fontsize=20)
    plt.xlabel('Sequence Position (Wild Type Residue)')
    plt.ylabel('Mutant Residue')
    plt.title(f'Single Mutation Effect Heatmap for {fasta_name}', fontsize=40)
    # plt.tight_layout()
    plt.savefig(heatmap_outpath, format='png', dpi=500, bbox_inches='tight')
    print(f"DMS figure saved at {heatmap_outpath}")
    # plt.show()
    plt.close()


# Main execution
def main():

    # add arguments via command line

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', nargs='+', help='input mutation analysis csv files')
    parser.add_argument('-r', help='Path to reference fasta')
    parser.add_argument('-c', help='threshold cutoff of sequences included in library')
    parser.add_argument('-m', help='integer value (up to 3) of desired lists of mutants m amino acids away')
    parser.add_argument('-x', help= 'minimum number of contexts for point mutation analysis')
    parser.add_argument('-d', type=float, help= 'change in binding score required to be considered significant')

    args = parser.parse_args()

    mutation_df_paths = args.i
    cpm = args.c
    dir_prefix =  '/'.join(mutation_df_paths[0].split('/')[:-1]) + '/plots/'
    print(f'output directory: {dir_prefix}')

    keyword = f'thresh{cpm}'
    csv_file = [path for path in mutation_df_paths if isinstance(path, str) and keyword in path]

    expt_id = csv_file[0].split('/')[-3] # experiment id is the entry past the 4th slash

    # reference specifications
    ref_fasta_path = args.r
    fasta_name = ref_fasta_path.split('/')[-1].replace('.fasta','')
    ref_fasta = SeqIO.read(ref_fasta_path, "fasta")
    parent_dna_seq = ref_fasta.seq
    parent_dna_id = ref_fasta.id
    parent_aa_seq = translate_sequence(parent_dna_seq)


    # if interested in wrapping across many cpm, the beginning of the for loop would start here

    library_df = pd.read_csv(csv_file[0], index_col=0)

    rep_list = [col for col in library_df.columns if 'rep' in col.lower()]
    num_replicates = len(rep_list)

    # calculate average scores and send to be plotted and saved

    try:
        library_df['average normalized score'] = library_df[rep_list].mean(axis=1)
        library_df['normalized score error'] = library_df[rep_list].std(axis=1)
        wt_slice = library_df[library_df['aa_sequence'] == parent_aa_seq]
        if wt_slice.empty:
            print("no wild type score found")
        else:
            wt_score = wt_slice['average normalized score']
            print(f"wild type score: {wt_score.to_string(index=False)}")
        library_org_df = library_df.sort_values(by=['number_aa_mutations',
                                                    'average normalized score',
                                                    'normalized score error'])
        org_outpath = dir_prefix + f'{expt_id}_seqs_avgscores.png'
        high_var = plot_seqavgs(dir_prefix, library_org_df, fasta_name, org_outpath)
        if len(high_var) == 0:
            print('no sequences with high variance')
        else:
            print(f'{len(high_var)} sequence replicates with high variance')
    except Exception as e:
        print(f"error: {e}")

    # create families of single mutants from aa sequences within the library

    lib_aaseq = library_df.sort_values('number_aa_mutations')['aa_sequence']
    # print(len(lib_aaseq))
    
    # compile lists of single mutants of a given sequence (iterate through sequences in the set of scores in order)


    mutant_lists = int(args.m)
    min_contexts = int(args.x)
    min_deltascore = args.d
    pointmuts_outdir = dir_prefix + 'point_mutations/'
    make_dir(pointmuts_outdir)

    short_path_highscore_dict = {}

    for mutnum in range(1, mutant_lists+1):
        try:
            print(f"calculating groups of mutants {mutnum} amino acids apart")
            muts_dict = {}
            for i in range(len(lib_aaseq)):
                aaseq = lib_aaseq.iloc[i]
                mutlist = []
                for j in range(i+1, len(lib_aaseq)):
                    seqcomp = lib_aaseq.iloc[j]
                    if seqcomp != aaseq:
                        hamming_pairwise_aa = hamming_distance(aaseq, seqcomp)
                        if hamming_pairwise_aa == mutnum:
                            mutlist.append(seqcomp)
                if len(mutlist) > 0:
                    muts_dict[aaseq] = mutlist
            print(f"{len(list(muts_dict.keys()))} groups of mutants {mutnum} aa apart")
            
            # comprehensive list of pairs of mutants organized by point mutation
            mutrec = []
            for sequence in list(muts_dict.keys()):
                seqdist = hamming_distance(sequence, parent_aa_seq)
                for variant in list(muts_dict[sequence]):
                    vardist = hamming_distance(variant, parent_aa_seq)
                    seqslice = library_org_df[library_org_df['aa_sequence'] == sequence]
                    varslice = library_org_df[library_org_df['aa_sequence'] == variant]
                    scores = [seqslice['average normalized score'].item(),
                              varslice['average normalized score'].item()]
                    if seqdist<=vardist:
                        dist = seqdist
                        dscore = scores[1] - scores[0]
                        pointmut = find_mutations(sequence, variant)
                        wt_muts = find_mutations(parent_aa_seq, sequence)
                    else:
                        dist = vardist
                        dscore = scores[0] - scores[1]
                        pointmut = find_mutations(variant, sequence)
                        wt_muts = find_mutations(parent_aa_seq, variant)
                    mutrec.append({
                        "sequence1": sequence,
                        "sequence2": variant,
                        "point mutation": pointmut,
                        "wild type distance": dist,
                        "wild type mutations": wt_muts,
                        "position": int(''.join([char for char in pointmut[0] if char.isdigit()])),
                        "score1": scores[0],
                        "score2": scores[1],
                        "delta score": dscore})
                
            # take slices of point mutations and build multi sequence contexts
            mutrec_df = pd.DataFrame(mutrec)
            # print(mutrec_df[:5])
            mutrec_sort_df = mutrec_df.sort_values(['wild type distance', 'position'])
            
            # now create context specific plots for single mutations or export cases of short hamming distance but high score impact
            contexts_dict = {}
            
            # break all mutations into a flat list of single point mutations
            mutrec_sort_df['point mutation'] = mutrec_sort_df['point mutation'].astype(str).apply(ast.literal_eval)
            allpointmuts = mutrec_sort_df['point mutation']
            allpmuts_flat = [pmut for sublist in allpointmuts for pmut in sublist]
            # for loop to assess contexts and score impacts of point mutations
            for muti in set(allpmuts_flat):
                # print(muti)
                pmutslice = mutrec_sort_df[mutrec_sort_df['point mutation'].apply(lambda x: muti in x)]
                pmutslice = pmutslice.sort_values(['wild type distance', 'score1', 'score2'])
                pmutslice.dropna(subset=['delta score'], inplace=True)
                # print(pmutslice[['wild type distance','point mutation']])
                # print(mutnum)
                if mutnum == 1:
                    # print(len(pmutslice))
                    if len(pmutslice) >= min_contexts:
                        position = int(muti[1:-1])-1
                        if muti[0] == parent_aa_seq[position]:
                            muti_outdir = pointmuts_outdir + f'{muti}/'
                        else:
                            muti_outdir = pointmuts_outdir + f'{muti}_nonparent/'
                        make_dir(muti_outdir)
                        mutidist_outpath = muti_outdir + f'scoreplots_{muti}.png'
                        colors = sns.color_palette("husl", len(pmutslice))
                        # print(pmutslice)
                        plot_mutation_contexts(pmutslice, colors, muti, mutidist_outpath)
                        pmutslice.to_csv(muti_outdir + f'{muti}_{mutnum}aa_table.csv')
                        
                        # build dictionary of context sequences to assess their distance from one another
                        contexts = []
                        for seq1, seq2 in zip(pmutslice['sequence1'],pmutslice['sequence2']):
                            wt_distances = [hamming_distance(seq1,parent_aa_seq), hamming_distance(seq2,parent_aa_seq)]
                            if wt_distances[0] < wt_distances[1]:
                                contexts.append(seq1)
                            else:
                                contexts.append(seq2)
                        contexts_dict[muti] = contexts
                else:
                    sig_pmut = pmutslice[pmutslice['delta score'].abs() > min_deltascore]
                    if len(sig_pmut) > 0:
                        position = int(muti[1:-1])-1
                        if muti[0] == parent_aa_seq[position]:
                            muti_outdir = pointmuts_outdir + f'{muti}/'
                        else:
                            muti_outdir = pointmuts_outdir + f'{muti}_nonparent/'
                        make_dir(muti_outdir)
                        sig_pmut.to_csv(muti_outdir + f'{muti}_{mutnum}aa_table.csv')
                        short_path_highscore_dict[muti] = sig_pmut
            
            if mutnum > 1:
                all_sig_pmut_df = pd.concat([df.assign(key=key) \
                                        for key, df in short_path_highscore_dict.items()], \
                                        ignore_index=True)
                all_sig_pmut_df.drop_duplicates(subset=['sequence1','sequence2'], keep = 'first')\
                    .rename(columns={"key": "mutation of interest"})\
                    .sort_values(['delta score'])\
                    .to_csv(pointmuts_outdir + f'{expt_id}_all{mutnum}aa_delscore{min_deltascore}.csv')

        except Exception as e:
            print(f"error: {e}")
        # now take entries in contexts dictionary and turn them into hamming distances between each other (only for single mutants)   
        if mutnum == 1:
            hamming_dist_contexts = {}
            for mutj in list(contexts_dict.keys()):
                hamming_dists = []
                for p in range(len(contexts_dict[mutj])):
                    seqp = contexts_dict[mutj][p]
                    for q in range(p+1, len(contexts_dict[mutj])):
                        seqq = contexts_dict[mutj][q]
                        hamming_dists.append(hamming_distance(seqp,seqq))
                hamming_dist_contexts[mutj] = hamming_dists
            for mutk in list(hamming_dist_contexts.keys()):
                position = int(mutk[1:-1])-1
                if mutk[0] == parent_aa_seq[position]:
                    mutk_outdir = pointmuts_outdir + f'{mutk}/'
                else:
                    mutk_outdir = pointmuts_outdir + f'{mutk}_nonparent/'
                mutkdist_outpath = mutk_outdir + f'contextdist_{mutk}.png'
                plot_context_distances(hamming_dist_contexts, mutk, contexts_dict, mutkdist_outpath)

            # group now by point mutation
            # initialize dicts

            mut_dict = {}
            noref_dict = {}
            # print(mutrec_sort_df['point mutation'])
            # print(parent_aa_seq)

            for mutation in mutrec_sort_df['point mutation']:
                # print(mutation[0])
                mutation_string = mutation[0]
                position = int(mutation_string[1:-1])-1
                # print(parent_aa_seq[position])
                # print(mutation_string[0], parent_aa_seq[position])
                # Create a boolean mask by extracting the string from each list
                mask = mutrec_sort_df['point mutation'].apply(lambda x: x[0] if isinstance(x, list) else x) == mutation_string
                if mutation_string[0] == parent_aa_seq[position]:
                    mut_dict[mutation_string] = list(mutrec_sort_df[mask]['delta score'].values)
                    # print(mut_dict)
                else:
                    noref_dict[mutation_string] = list(mutrec_sort_df[mask]['delta score'].values)
                    # print(noref_dict)

            sorted_mut_dict = {k: sorted(v, reverse=True) for k, v in mut_dict.items()}
            # print(sorted_mut_dict)

            # graph above as a ridge plot (see above function outside main)
            # create ridge data
            ridgedata = []
            for key, values in sorted_mut_dict.items():
                if len(list(values)) >= min_contexts:
                    for v in values:
                        ridgedata.append({"mutation": key, "dscore": v})
            # print(ridgedata)

            ridgedata_df = pd.DataFrame(ridgedata)
            # print(len(ridgedata_df))
            ridge_outpath = dir_prefix + f'pointmut_ridges_{min_contexts}contexts.png'

            plot_context_ridges(sorted_mut_dict, ridgedata_df, fasta_name, ridge_outpath)

            # now make DMS like plot (look at all single mutants off wild type)

            amino_acids_dict = {
                'A': 'Alanine',
                'C': 'Cysteine',
                'D': 'Aspartic acid',
                'E': 'Glutamic acid',
                'F': 'Phenylalanine',
                'G': 'Glycine',
                'H': 'Histidine',
                'I': 'Isoleucine',
                'K': 'Lysine',
                'L': 'Leucine',
                'M': 'Methionine',
                'N': 'Asparagine',
                'P': 'Proline',
                'Q': 'Glutamine',
                'R': 'Arginine',
                'S': 'Serine',
                'T': 'Threonine',
                'V': 'Valine',
                'W': 'Tryptophan',
                'Y': 'Tyrosine'
            }

            # initialize variables
            mutscore_dict = {}
            mut_scores = []

            for pos, res in enumerate(parent_aa_seq):
                # print(f"position {pos+1}, residue {res}")
                for mut in list(amino_acids_dict.keys()):
                    singlemut = parent_aa_seq[:pos] + mut + parent_aa_seq[(pos+1):] if len(parent_aa_seq) > pos else parent_aa_seq
                    # singlemut_list.append(singlemut)
                    # print(singlemut)
                    if singlemut != parent_aa_seq:
                        mut_seq_df = library_df[library_df['aa_sequence'] == singlemut]
                        if mut_seq_df.empty:
                            # print(f"{res}{pos+1}{mut} not found") 
                            continue
                        else:
                            smutscore = mut_seq_df['average normalized score'].item()
                        
                        mut_scores.append(smutscore-wt_score)
                        mutation = str(res+str(pos+1)+mut)
                        mutscore_dict[mutation] = mut_scores[-1]

            # begin populating axes of dms plot
            positions = list(range(1,len(parent_aa_seq)+1))
            wtresidues = list(parent_aa_seq)
            wt_df = pd.DataFrame({'position':positions, 'wtres':wtresidues})

            rec = []
            for mut, score in mutscore_dict.items():
                wt = mut[0]
                pos = int(mut[1:-1])
                mut_aa = mut[-1]
                rec.append({"position": pos, "wtres": wt, "mutres": mut_aa, "score": score})

            mut_df = pd.DataFrame(rec)
    
            rec_df = pd.merge(wt_df,mut_df, on=['position','wtres'], how='left')
            rec_df = rec_df.fillna('')

            # Get only mutations with scores
            positions = sorted(rec_df['position'].unique())
            wt_seq_map = rec_df.drop_duplicates('position').set_index('position')['wtres'].to_dict()
            wt_pos_labels = [f"{pos}{wt_seq_map.get(pos, '')}" for pos in positions]
            all_aas = list(amino_acids_dict.keys())
            
            mut_df = rec_df[rec_df['mutres'].notna() & (rec_df['mutres'] != '') & rec_df['score'].notna()]
            
            # Map from position number to its x-axis index (for correct col placement)
            pos2col = dict(zip(positions, wt_pos_labels))
            
            # Prepare the DataFrame to fill
            heatmap_matrix = pd.DataFrame(index=all_aas, columns=wt_pos_labels, dtype=float)
            
            # Fill in scores where known:
            for _, row in mut_df.iterrows():
                pos = int(row['position'])
                if pos not in pos2col:
                    continue
                col = pos2col[pos]
                # .item() extracts the scalar value from a Series
                score = row['score'].item() if hasattr(row['score'], 'item') else row['score']
                heatmap_matrix.at[row['mutres'], col] = score

            heatmap_outpath = dir_prefix + f'1mut_heatmap_{expt_id}.png'

            plot_dms(heatmap_matrix, fasta_name, heatmap_outpath) 
        print('------next amino acid distance------')
        # except Exception as e:
        #     print(f"error: {e}")
                      
            
if __name__ == "__main__":
    main()





    