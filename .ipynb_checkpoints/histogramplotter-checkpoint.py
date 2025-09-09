#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import packages needed
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math


# In[2]:


# #set inputs for plot histogram function
# cell_sort_pop = 'yJH4s17'
# filepath = f"../../Data/2025-01-13/00_fastq/minibinders_data/minibinders_outputs/jh_011/mutation_dfs/{cell_sort_pop}_mutation_freq.csv"
# output_path = f"../../Data/2025-01-13/00_fastq/minibinders_data/minibinders_outputs/jh_011/mutation_dfs/{cell_sort_pop}_mutation_freqhist.png"
# # column_name = "aa_mutations"
# print(filepath)


# In[3]:


#function for dataframe making
def make_dataframe(filepath):
    mutation_df = pd.read_csv(filepath)
    return(mutation_df)


# In[4]:


def log_histogram(mutation_df, column_name, output_path, bin_width):
    #set appropriate path for finding mutation csv files

    #Check that column of interest exists
    if column_name not in mutation_df.columns:
        print(f"Column '{column_name}' not found in the CSV file.")
        return

    #extract specified column
    values = mutation_df[column_name].dropna()

    #num_bins calc
    min_value = values.min()
    max_value = values.max()
    num_bins = int((max_value - min_value) / bin_width)
    print(num_bins)

    #histogram plot
    fig1 = plt.figure(figsize=(16,8))
    plt.hist(values, bins=num_bins, log=True)
    plt.xscale('log')
    plt.xlabel(column_name)
    plt.yscale('linear')
    plt.ylabel('Frequency')
    plt.title(f'{column_name} for sort condition')
    plt.grid(True, which="major", ls="--")
    # plt.show()
    plt.savefig(output_path, format='png', dpi=300)
    plt.close()
    print(f"Histogram saved as {output_path}")
    return(fig1)



# In[5]:


def character_bars(mutation_df, output_path, freq_cutoff, cell_sort_pop, num_sequences):

    # Set global font size and family
    plt.rcParams.update({
        'font.size': 16,
        # 'font.family': 'Arial'
    })
    #count occurrence of each string
    value_counts = mutation_df.value_counts()
    value_counts_filt = value_counts[value_counts >= freq_cutoff]
    # print(value_counts)

    #histogram plot
    fig2 = plt.figure(figsize=(16,8))
    value_counts_filt.plot(kind='bar')
    plt.xlabel(f'{mutation_df.columns[0]} (% of reads analyzed)')
    plt.yscale('linear')
    plt.ylabel('Frequency')
    # Get the current tick labels

    plt.xticks(rotation=0, ha='center')
    plt.title(f'{mutation_df.columns[0]} for {cell_sort_pop} (top {num_sequences} reads analyzed)')
    # plt.grid(True, which="major", ls="--")
    plt.tight_layout()
    # plt.show()
    # plt.close()
    # Get the axes from the figure
    ax = fig2.gca()


    labels = [item.get_text() for item in ax.get_xticklabels()]


    # Remove parentheses and commas
    clean_labels = [label.strip("(),'") for label in labels]

    # Set the cleaned labels
    ax.set_xticklabels(clean_labels)

    # bars = ax.bar(str(value_counts_filt.index), value_counts_filt.values)
    rounded_vals = np.round((value_counts_filt.values/num_sequences*100),decimals=1)

    # Create new labels that include both the category and its count
    new_labels = [f'{cat}\n({count})' for cat, count in zip(clean_labels, rounded_vals)]

    # Set the new x-axis labels
    ax.set_xticks(range(len(value_counts_filt)))
    ax.set_xticklabels(new_labels)
    plt.draw()
    # plt.show()
    plt.savefig(output_path, format='png', dpi=500)
    plt.show()
    plt.close()
    print(f"Histogram saved as {output_path}")
    return(fig2)


# In[6]:


# #run test functions
# bin_width = 1
# freq_cutoff = 150
# mutation_df = make_dataframe(filepath)
# # print(mutation_df)
# # data = pd.Series(['apple', 'banana', 'apple', 'cherry', 'banana', 'apple', 'date', 'apple', 'banana'])
# # print(mutation_df['aa_mutations'])
# # log_histogram(mutation_df, column_name, output_path, bin_width)
# character_bars(mutation_df, output_path, freq_cutoff, cell_sort_pop, 3353)
