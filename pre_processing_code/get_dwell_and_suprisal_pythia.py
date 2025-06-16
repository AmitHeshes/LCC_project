import pandas as pd

# Load both
kenlm_df = pd.read_csv("pythia70M_surprisals_ID.csv")
dwell_df = pd.read_csv("ia_dwell_time_simple.csv")

# Merge
merged = pd.merge(kenlm_df, dwell_df, on=["participant_id", "TRIAL_INDEX", "word", "IA_ID"], how="inner")

# Save merged file
merged.to_csv("merged_surprisal_dwell_pythia70M.csv", index=False)