import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load both
kenlm_df = pd.read_csv("pythia70M_surprisals_fast.csv")
dwell_df = pd.read_csv("ia_dwell_time_simple.csv")

# Merge
merged = pd.merge(kenlm_df, dwell_df, on=["participant_id", "TRIAL_INDEX", "word"], how="inner")

# Save merged file
merged.to_csv("merged_surprisal_dwell_pythia70M.csv", index=False)

# Load merged file
merged = pd.read_csv("merged_surprisal_dwell_pythia70M.csv")

# Optional: filter out weird values
merged = merged[merged["IA_DWELL_TIME"] > 0]
merged = merged[merged["pythia70M_surprisal"] > 0]

# Plot (swapped axes)
plt.figure(figsize=(8, 6))
sns.regplot(
    x="IA_DWELL_TIME", 
    y="pythia70M_surprisal", 
    data=merged, 
    scatter_kws={"s": 10}, 
    line_kws={"color": "red"}
)

plt.title("Correlation between IA DWELL TIME and pythia70M Surprisal")
plt.xlabel("IA DWELL TIME (Total Fixation Duration)")
plt.ylabel("pythia70M Surprisal")

plt.xlim(0, 600)

plt.tight_layout()
plt.savefig("dwelltime_vs_pythia70M_surprisal.png")  # Save the new plot
plt.show()
