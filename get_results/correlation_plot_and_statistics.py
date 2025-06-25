import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# merged_path = "pre_processing_data\\merged_surprisal_dwell_kenlm_pythia.csv"
merged_path = f"pre_processing_data\\merged_surprisal_dwell_kenlm_pythia_new_try.csv"
# outputs paths
results_folder = "results_constructed_part1_new_try\\correlations"
output_graph_path = results_folder + "\\surprisals_correlation_scatter.png"
output_summary_path = results_folder + "\\summary_statistics_surprsial_kenlm_surprisal_pythia_graph.csv"
column_description_to_column_name = {
    "kenlm": "kenlm_surprisal",
    "pythia": "pythia70M_surprisal",
    "pythia_sum": "pythia_sum_surprisal",
    "pythia_average": "pythia_average_surprisal",
    "dwell_time": "IA_DWELL_TIME"
}
axis_titles = {
    "kenlm": "N-gram Model Surprisal (KenLM)",
    "pythia": "Neural Model Surprisal (Pythia-70M)",
    "pythia_sum": "Neural Model Surprisal (Pythia-70M)",
    "pythia_average": "Neural Model Surprisal (Pythia-70M)",
    "dwell_time": "Dwell Time (ms)"
}
graph_titles = {
    "kenlm": "N-gram Surprisal",
    "pythia": "Neural Model Surprisal",
    "pythia_sum": "Neural Model Surprisal",
    "pythia_average": "Neural Model Surprisal",
    "dwell_time": "Dwell Time"
}
short_name = {
    "kenlm": "KenLM",
    "pythia": "Pythia-70M",
    "pythia_sum": "Pythia-70M (Sum)",
    "pythia_average": "Pythia-70M (Average)",
    "dwell_time": "Dwell Time"
}

def load_surprisal_data(csv_file):
    """Load the surprisal data from CSV file."""
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} data points from {csv_file}")
    return df


def clean_data(df, x_column_desc, y_column_desc):
    """Clean the data by removing any rows with missing values."""
    x_column_name = column_description_to_column_name[x_column_desc]
    y_column_name = column_description_to_column_name[y_column_desc]

    initial_count = len(df)
    
    # Remove rows with NaN values in surprisal columns
    df_clean = df.dropna(subset=[x_column_name, y_column_name])
    
    # Remove any infinite values
    df_clean = df_clean[np.isfinite(df_clean[x_column_name]) & 
                       np.isfinite(df_clean[y_column_name])]
    
    final_count = len(df_clean)
    print(f"After cleaning: {final_count} data points ({initial_count - final_count} removed)")
    
    return df_clean


def calculate_statistics(x, y):
    """Calculate correlation statistics."""
    correlation, p_value = pearsonr(x, y)
    
    # Calculate R-squared
    r_squared = correlation ** 2
    
    # Linear regression for trend line
    slope, intercept, _, _, _ = stats.linregress(x, y)
    
    return {
        'correlation': correlation,
        'p_value': p_value,
        'r_squared': r_squared,
        'slope': slope,
        'intercept': intercept
    }


def create_scatter_plot(df, stats_dict, x_column_desc, y_column_desc, output_file, does_to_plot_perfect_corellation, figsize=(10, 8)):
    """Create a scatter plot comparing n-gram vs neural model surprisals."""
    x_column_name = column_description_to_column_name[x_column_desc]
    y_column_name = column_description_to_column_name[y_column_desc]

    plt.figure(figsize=figsize)
    
    # Create scatter plot
    plt.scatter(df[x_column_name], df[y_column_name], 
               alpha=0.6, s=20, color='steelblue', edgecolors='white', linewidth=0.5)
    
    # Add trend line
    x_line = np.linspace(df[x_column_name].min(), df[x_column_name].max(), 100)
    y_line = stats_dict['slope'] * x_line + stats_dict['intercept']
    plt.plot(x_line, y_line, 'r--', alpha=0.8, linewidth=2, label='Best fit line')
    
    # Add diagonal reference line (perfect correlation)
    min_val = min(df[x_column_name].min(), df[y_column_name].min())
    max_val = max(df[x_column_name].max(), df[y_column_name].max())
    if does_to_plot_perfect_corellation:
        plt.plot([min_val, max_val], [min_val, max_val], 'k:', alpha=0.5, 
                linewidth=1, label='Perfect correlation')
    
    # Formatting
    plt.xlabel(axis_titles[x_column_desc], fontsize=12)
    plt.ylabel(axis_titles[y_column_desc], fontsize=12)
    plt.title(f'Relationship between {graph_titles[x_column_desc]} and {graph_titles[y_column_desc]} Estimates', 
              fontsize=14, fontweight='bold')
    
    # Add statistics text box
    # stats_text = f"""Statistics:
    # r = {stats_dict['correlation']:.3f}
    # R² = {stats_dict['r_squared']:.3f}
    # p = {stats_dict['p_value']:.2e}
    # n = {len(df):,} words"""
    
    # plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
    #          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    #          fontsize=10)
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    # plt.legend(loc='lower right')
    
    # Tight layout
    plt.tight_layout()
    
    # Save if output file specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {output_file}")
    
    plt.show()

def print_summary_statistics(df, stats_dict, output_summary_path, x_column_desc, y_column_desc):
    x_column_name = column_description_to_column_name[x_column_desc]
    y_column_name = column_description_to_column_name[y_column_desc]

    """Print summary statistics about the data and correlation."""
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    print(f"Total number of words: {len(df):,}")

    print(f"\n{axis_titles[x_column_desc]}:")
    print(f"  Mean: {df[x_column_name].mean():.3f}")
    print(f"  Std:  {df[x_column_name].std():.3f}")
    print(f"  Min:  {df[x_column_name].min():.3f}")
    print(f"  Max:  {df[x_column_name].max():.3f}")
    
    print(f"\n{axis_titles[y_column_desc]}:")
    print(f"  Mean: {df[y_column_name].mean():.3f}")
    print(f"  Std:  {df[y_column_name].std():.3f}")
    print(f"  Min:  {df[y_column_name].min():.3f}")
    print(f"  Max:  {df[y_column_name].max():.3f}")
    
    print("\nCorrelation Analysis:")
    print(f"  Pearson correlation: {stats_dict['correlation']:.4f}")
    print(f"  R-squared: {stats_dict['r_squared']:.4f}")
    print(f"  P-value: {stats_dict['p_value']:.2e}")
    
    if stats_dict['p_value'] < 0.001:
        print("  Significance: Highly significant (p < 0.001)")
    elif stats_dict['p_value'] < 0.01:
        print("  Significance: Very significant (p < 0.01)")
    elif stats_dict['p_value'] < 0.05:
        print("  Significance: Significant (p < 0.05)")
    else:
        print("  Significance: Not significant (p >= 0.05)")

    # save all printed information to csv
    short_name_x = short_name[x_column_desc]
    short_name_y = short_name[y_column_desc]
    summary_stats = {
        'Total Words': len(df),
        f'{short_name_x} Mean': df[x_column_name].mean(),
        f'{short_name_x} Std': df[x_column_name].std(),
        f'{short_name_x} Min': df[x_column_name].min(),
        f'{short_name_x} Max': df[x_column_name].max(),
        f'{short_name_y} Mean': df[y_column_name].mean(),
        f'{short_name_y} Std': df[y_column_name].std(),
        f'{short_name_y} Min': df[y_column_name].min(),
        f'{short_name_y} Max': df[y_column_name].max(),
        f'Pearson Correlation': stats_dict['correlation'],
        f'R-squared': stats_dict['r_squared'],
        f'P-value': stats_dict['p_value']
    }
    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv(output_summary_path, index=False)

def main(csv_file, scatter_output, output_summary_path, x_column_desc, y_column_desc, does_to_plot_perfect_corellation):
    """Main function to analyze and plot surprisal correlations."""
    x_column_name = column_description_to_column_name[x_column_desc]
    y_column_name = column_description_to_column_name[y_column_desc]
    # Load and clean data
    df = load_surprisal_data(csv_file)
    df_clean = clean_data(df, x_column_desc, y_column_desc)
    
    if len(df_clean) == 0:
        print("No valid data points found after cleaning!")
        return
    
    # Calculate statistics
    stats_dict = calculate_statistics(df_clean[x_column_name], 
                                    df_clean[y_column_name])
    
    # Print summary
    print_summary_statistics(df_clean, stats_dict, output_summary_path, x_column_desc=x_column_desc, y_column_desc=y_column_desc)
    
    # Create plot
    create_scatter_plot(df_clean, stats_dict, x_column_desc=x_column_desc, y_column_desc=y_column_desc, output_file=scatter_output, does_to_plot_perfect_corellation=does_to_plot_perfect_corellation)
    

if __name__ == "__main__":
    # Configuration
    # csv_file = "C:/Users/raque/Desktop/LCC_project/pre_processing_data/merged_surprisal_dwell_kenlm_pythia.csv"
    csv_file = merged_path

    # Optional: specify output file names
    scatter_output = "surprisal_correlation_scatter.png"
    
    # Run analysis
    
    main(merged_path, f"{results_folder}\\correlation_scatter_kenlm_vs_pythia_sum.png", f"{results_folder}\\summary_statistics_kenlm_vs_pythia_sum.csv", x_column_desc="kenlm", y_column_desc="pythia_sum", does_to_plot_perfect_corellation=True)
    main(merged_path, f"{results_folder}\\correlation_scatter_kenlm_vs_pythia_average.png", f"{results_folder}\\summary_statistics_kenlm_vs_pythia_average.csv", x_column_desc="kenlm", y_column_desc="pythia_average", does_to_plot_perfect_corellation=True)
    main(merged_path, f"{results_folder}\\correlation_scatter_kenlm_vs_dwell_time.png", f"{results_folder}\\summary_statistics_kenlm_vs_dwell_time.csv", x_column_desc="dwell_time", y_column_desc="kenlm", does_to_plot_perfect_corellation=False)
    main(merged_path, f"{results_folder}\\correlation_scatter_pythia_sum_vs_dwell_time.png", f"{results_folder}\\summary_statistics_pythia_sum_vs_dwell_time.csv", x_column_desc="dwell_time", y_column_desc="pythia_sum", does_to_plot_perfect_corellation=False)
    main(merged_path, f"{results_folder}\\correlation_scatter_pythia_average_vs_dwell_time.png", f"{results_folder}\\summary_statistics_pythia_average_vs_dwell_time.csv", x_column_desc="dwell_time", y_column_desc="pythia_average", does_to_plot_perfect_corellation=False)