import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

kenlm_surprisal_path = "kenlm_surprisals_simple.csv"
dwell_time_path = "ia_dwell_time_simple.csv"
merged_path = "merged_surprisal_dwell.csv"

def merge_surprisal_dwell(kenlm_surprisal_path, dwell_time_path, merged_path):
    """
    Merge KenLM surprisal and dwell time dataframes on participant_id, TRIAL_INDEX, and word.
    """
    # Load both
    kenlm_surprisal_df = pd.read_csv(kenlm_surprisal_path)
    dwell_df = pd.read_csv(dwell_time_path)

    # Merge
    merged = pd.merge(kenlm_surprisal_df, dwell_df, on=["participant_id", "TRIAL_INDEX", "word"], how="inner")
    print("Finished merging dataframes")

    # Save merged file
    merged.to_csv(merged_path, index=False)
    print(f"Merged file saved: {merged_path}")

    # Load merged file
    # merged = pd.read_csv("merged_surprisal_dwell.csv")

    # Optional: filter out weird values (zero or negative dwell time, etc.)
    # merged = merged[merged["IA_DWELL_TIME"] > 0]
    # merged = merged[merged["kenlm_surprisal"] > 0]

    # Plot
    # plt.figure(figsize=(8, 6))
    # sns.regplot(x="kenlm_surprisal", y="IA_DWELL_TIME", data=merged, scatter_kws={"s": 10}, line_kws={"color": "red"})

    # plt.title("Correlation between KenLM Surprisal and IA DWELL TIME")
    # plt.xlabel("KenLM Surprisal")
    # plt.ylabel("IA DWELL TIME (Total Fixation Duration)")

    # plt.tight_layout()
    # plt.savefig("kenlm_surprisal_vs_dwelltime.png")  # Save the plot
    # plt.show()


if __name__ == "__main__":
    merge_surprisal_dwell(kenlm_surprisal_path, dwell_time_path, merged_path)