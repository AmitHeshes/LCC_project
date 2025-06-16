import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

merged_path = "merged_surprisal_dwell.csv"
results_folder = "results_constructed_part1"

def get_praph_results_part1(merged_dataframe):
    """
    This function is a placeholder for the first part of the graph results.
    It currently returns an empty dictionary.
    
    Returns:
        dict: An empty dictionary representing the graph results.
    """
    plt.figure(figsize=(8, 6))
    sns.regplot(x="IA_DWELL_TIME", y="kenlm_surprisal", data=merged_dataframe, scatter_kws={"s": 10}, line_kws={"color": "red"})
    plt.title("Correlation between KenLM Surprisal and IA DWELL TIME")
    plt.xlabel("KenLM Surprisal")
    plt.ylabel("IA DWELL TIME (Total Fixation Duration)")

    plt.tight_layout()
    plt.savefig(results_folder + "/kenlm_surprisal_vs_dwelltime.png")  # Save the plot
    plt.show()

def main(merged_path):
    # Load the merged dataframe
    merged_df = pd.read_csv(merged_path)
    
    # Call the function to get graph results
    get_praph_results_part1(merged_df)


if __name__ == "__main__":
    main(merged_path)