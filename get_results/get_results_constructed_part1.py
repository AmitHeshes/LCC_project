import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# merged_path = "pre_processing_data\\merged_surprisal_dwell_kenlm_pythia.csv"
merged_path = f"pre_processing_data\\merged_surprisal_dwell_kenlm_pythia_new_try.csv"
results_folder = "results_constructed_part1_new_try"

model_name_to_model_surprisal_column = {
    "kenlm": "kenlm_surprisal",
    "pythia": "pythia70M_surprisal",
    "pythia_sum": "pythia_sum_surprisal",
    "pythia_average": "pythia_average_surprisal"
}


def get_praph_results_part1(merged_dataframe, model_name="kenlm"):
    """
    This function is a placeholder for the first part of the graph results.
    It currently returns an empty dictionary.
    
    Returns:
        dict: An empty dictionary representing the graph results.
    """
    model_surprisal_column = model_name_to_model_surprisal_column[model_name]
    plt.figure(figsize=(8, 6))
    sns.regplot(x="IA_DWELL_TIME", y=model_surprisal_column, data=merged_dataframe, scatter_kws={"s": 10}, line_kws={"color": "red"})
    plt.title(f"Correlation between {model_name} Surprisal and IA DWELL TIME")
    plt.xlabel("IA DWELL TIME (Total Fixation Duration)")
    plt.ylabel(f"{model_name} Surprisal")

    plt.tight_layout()
    plt.savefig(results_folder + f"\\{model_name}_surprisal_vs_dwelltime.png")  # Save the plot
    plt.show()


def main(merged_path):
    # Load the merged dataframe
    merged_df = pd.read_csv(merged_path)
    # Call the function to get graph results
    get_praph_results_part1(merged_df, "kenlm")
    # get_praph_results_part1(merged_df, "pythia")
    get_praph_results_part1(merged_df, "pythia_sum")
    get_praph_results_part1(merged_df, "pythia_average")


if __name__ == "__main__":
    main(merged_path)