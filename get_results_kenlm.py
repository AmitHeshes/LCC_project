import pandas as pd

# Load both
kenlm_df = pd.read_csv("kenlm_surprisals_simple.csv")
dwell_df = pd.read_csv("ia_dwell_time_simple.csv")

# Merge
merged = pd.merge(kenlm_df, dwell_df, on=["participant_id", "TRIAL_INDEX", "word"], how="inner")

# Save merged file
merged.to_csv("merged_surprisal_dwell.csv", index=False)

import matplotlib.pyplot as plt
import seaborn as sns

# Load merged file
merged = pd.read_csv("merged_surprisal_dwell.csv")

# Optional: filter out weird values (zero or negative dwell time, etc.)
merged = merged[merged["IA_DWELL_TIME"] > 0]
merged = merged[merged["kenlm_surprisal"] > 0]

# Plot
plt.figure(figsize=(8, 6))
sns.regplot(x="kenlm_surprisal", y="IA_DWELL_TIME", data=merged, scatter_kws={"s": 10}, line_kws={"color": "red"})

plt.title("Correlation between KenLM Surprisal and IA DWELL TIME")
plt.xlabel("KenLM Surprisal")
plt.ylabel("IA DWELL TIME (Total Fixation Duration)")

plt.tight_layout()
plt.savefig("kenlm_surprisal_vs_dwelltime.png")  # Save the plot
plt.show()


