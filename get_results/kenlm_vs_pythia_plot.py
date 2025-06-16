import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

merged_path = "pre_processing_data\\merged_surprisal_dwell_kenlm_pythia.csv"
# outputs paths
results_folder = "results_constructed_part1"
output_graph_path = results_folder + "\\surprisals_correlation_scatter.png"
output_summary_path = results_folder + "\\summary_statistics_surprsial_kenlm_surprisal_pythia_graph.csv"


def load_surprisal_data(csv_file):
    """Load the surprisal data from CSV file."""
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} data points from {csv_file}")
    return df


def clean_data(df):
    """Clean the data by removing any rows with missing values."""
    initial_count = len(df)
    
    # Remove rows with NaN values in surprisal columns
    df_clean = df.dropna(subset=['kenlm_surprisal', 'pythia70M_surprisal'])
    
    # Remove any infinite values
    df_clean = df_clean[np.isfinite(df_clean['kenlm_surprisal']) & 
                       np.isfinite(df_clean['pythia70M_surprisal'])]
    
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


def create_scatter_plot(df, stats_dict, output_file=None, figsize=(10, 8)):
    """Create a scatter plot comparing n-gram vs neural model surprisals."""
    
    plt.figure(figsize=figsize)
    
    # Create scatter plot
    plt.scatter(df['kenlm_surprisal'], df['pythia70M_surprisal'], 
               alpha=0.6, s=20, color='steelblue', edgecolors='white', linewidth=0.5)
    
    # Add trend line
    x_line = np.linspace(df['kenlm_surprisal'].min(), df['kenlm_surprisal'].max(), 100)
    y_line = stats_dict['slope'] * x_line + stats_dict['intercept']
    plt.plot(x_line, y_line, 'r--', alpha=0.8, linewidth=2, label='Best fit line')
    
    # Add diagonal reference line (perfect correlation)
    min_val = min(df['kenlm_surprisal'].min(), df['pythia70M_surprisal'].min())
    max_val = max(df['kenlm_surprisal'].max(), df['pythia70M_surprisal'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k:', alpha=0.5, 
             linewidth=1, label='Perfect correlation')
    
    # Formatting
    plt.xlabel('N-gram Model Surprisal (KenLM)', fontsize=12)
    plt.ylabel('Neural Model Surprisal (Pythia-70M)', fontsize=12)
    plt.title('Relationship between N-gram and Neural Model Surprisal Estimates', 
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

def print_summary_statistics(df, stats_dict, output_summary_path):
    """Print summary statistics about the data and correlation."""
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    print(f"Total number of words: {len(df):,}")

    print("\nN-gram Model (KenLM) Surprisal:")
    print(f"  Mean: {df['kenlm_surprisal'].mean():.3f}")
    print(f"  Std:  {df['kenlm_surprisal'].std():.3f}")
    print(f"  Min:  {df['kenlm_surprisal'].min():.3f}")
    print(f"  Max:  {df['kenlm_surprisal'].max():.3f}")
    
    print("\nNeural Model (Pythia-70M) Surprisal:")
    print(f"  Mean: {df['pythia70M_surprisal'].mean():.3f}")
    print(f"  Std:  {df['pythia70M_surprisal'].std():.3f}")
    print(f"  Min:  {df['pythia70M_surprisal'].min():.3f}")
    print(f"  Max:  {df['pythia70M_surprisal'].max():.3f}")
    
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
    summary_stats = {
        'Total Words': len(df),
        'KenLM Mean': df['kenlm_surprisal'].mean(),
        'KenLM Std': df['kenlm_surprisal'].std(),
        'KenLM Min': df['kenlm_surprisal'].min(),
        'KenLM Max': df['kenlm_surprisal'].max(),
        'Pythia Mean': df['pythia70M_surprisal'].mean(),
        'Pythia Std': df['pythia70M_surprisal'].std(),
        'Pythia Min': df['pythia70M_surprisal'].min(),
        'Pythia Max': df['pythia70M_surprisal'].max(),
        'Pearson Correlation': stats_dict['correlation'],
        'R-squared': stats_dict['r_squared'],
        'P-value': stats_dict['p_value']
    }
    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv(output_summary_path, index=False)

def main(csv_file, scatter_output, output_summary_path):
    """Main function to analyze and plot surprisal correlations."""
    
    # Load and clean data
    df = load_surprisal_data(csv_file)
    df_clean = clean_data(df)
    
    if len(df_clean) == 0:
        print("No valid data points found after cleaning!")
        return
    
    # Calculate statistics
    stats_dict = calculate_statistics(df_clean['kenlm_surprisal'], 
                                    df_clean['pythia70M_surprisal'])
    
    # Print summary
    print_summary_statistics(df_clean, stats_dict, output_summary_path)
    
    # Create plot
    create_scatter_plot(df_clean, stats_dict, scatter_output)
    

if __name__ == "__main__":
    # Run analysis
    main(merged_path, output_graph_path, output_summary_path)