from scipy.stats import pearsonr, spearmanr
import pandas as pd

# === KenLM ===
kenlm_df = pd.read_csv("merged_surprisal_dwell_kenlm.csv")

# Clean
kenlm_df = kenlm_df[(kenlm_df["IA_DWELL_TIME"] > 0) & (kenlm_df["kenlm_surprisal"] > 0)]

# Correlation
pearson_kenlm = pearsonr(kenlm_df["kenlm_surprisal"], kenlm_df["IA_DWELL_TIME"])
spearman_kenlm = spearmanr(kenlm_df["kenlm_surprisal"], kenlm_df["IA_DWELL_TIME"])

print("KenLM:")
print("  Pearson:", pearson_kenlm)
print("  Spearman:", spearman_kenlm)

# === Pythia ===
pythia_df = pd.read_csv("merged_surprisal_dwell_pythia70M.csv")

# Clean
pythia_df = pythia_df[(pythia_df["IA_DWELL_TIME"] > 0) & (pythia_df["pythia70M_surprisal"] > 0)]

# Correlation
pearson_pythia = pearsonr(pythia_df["pythia70M_surprisal"], pythia_df["IA_DWELL_TIME"])
spearman_pythia = spearmanr(pythia_df["pythia70M_surprisal"], pythia_df["IA_DWELL_TIME"])

print("Pythia 70M:")
print("  Pearson:", pearson_pythia)
print("  Spearman:", spearman_pythia)
