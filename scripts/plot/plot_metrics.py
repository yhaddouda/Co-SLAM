import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# 1. Initialization
# ---------------------------------------------------------
df = pd.read_csv('output/replica_tracking_ba_ablation_results.csv')

metrics = ['DepthL1', 'Accuracy', 'Completion', 'CompletionRatio']
configs = ['A', 'B', 'C', 'D']

# Map letters to full descriptions for the legend
legend_labels = {
    'A': 'baseline',
    'B': 'tracking pruning',
    'C': 'BA pruning',
    'D': 'whole pipeline pruning'
}

# Standard metric indicators (Arrow down = lower error is better; Arrow up = higher % is better)
metric_arrows = {
    'Accuracy': r'$\downarrow$',
    'Completion': r'$\downarrow$',
    'CompletionRatio': r'$\uparrow$',
    'DepthL1': r'$\downarrow$'
}

# ---------------------------------------------------------
# 2. Publication-Ready Formatting
# ---------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (8.5, 4.0), # Widened slightly to fit the longer legend labels
    'figure.dpi': 300,
})

colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
hatches = ['', '//', '\\\\', '..']

# ---------------------------------------------------------
# 3. Main Loop For Each Metric
# ---------------------------------------------------------
for metric in metrics:
    plot_df = df.copy()
    
    # Exclude office4 for DepthL1 to avoid outliers skewing chart
    if metric == 'DepthL1':
        plot_df = plot_df[plot_df['Scene'] != 'office4']
        
    grouped = plot_df.groupby(['Scene', 'Config'])[metric]
    means = grouped.mean().unstack()[configs]
    errors = grouped.std().unstack()[configs] 

    fig, ax = plt.subplots()
    x = np.arange(len(means.index))
    width = 0.20 

    for i, col in enumerate(configs):
        offset = (i - 1.5) * width
        
        # Pull the proper label name from dictionary
        ax.bar(x + offset, means[col], width, yerr=errors[col], 
               label=legend_labels[col], color=colors[i], edgecolor='black', 
               hatch=hatches[i], capsize=3, alpha=0.9,
               error_kw={'elinewidth': 1.2, 'capthick': 1.2})

    ax.set_xlabel('Replica Scene')
    ax.set_ylabel(metric)
    
    # Title includes arrow directionality mapping
    ax.set_title(f'{metric} ({metric_arrows[metric]}) across Replica Scenes (Mean & Std Dev)')
    ax.set_xticks(x)
    ax.set_xticklabels(means.index, rotation=0)

    # Place the wider legend slightly outside
    ax.legend(title='Configuration', loc='upper left', bbox_to_anchor=(1.02, 1), 
              framealpha=1.0, edgecolor='black')
              
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(f'ablation_ba_{metric}_final.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig)