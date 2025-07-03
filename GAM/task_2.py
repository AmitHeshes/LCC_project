import pandas as pd
import numpy as np
from pygam import LinearGAM, s
import matplotlib.pyplot as plt

# Load data
INPUT_DATA_PATH = "pre_processing_data/merged_surprisal_dwell_kenlm_pythia.csv"
frequency_column = 'wordfreq_frequency'


df = pd.read_csv(INPUT_DATA_PATH)

# Control variables
df['log_freq'] = -df[frequency_column]

# List of dependent variables (reading time measures)
reading_measures = {
    "Total Fixation Duration": "IA_DWELL_TIME",
    "Gaze Duration": "IA_FIRST_RUN_DWELL_TIME",
    "Regression Path Duration": "IA_REGRESSION_PATH_DURATION"
}

# Loop through each reading time measure
for label, y_col in reading_measures.items():
    print(f"\nProcessing {label} ({y_col})...")
    
    # Drop rows with missing data in relevant columns
    subset = df[['pythia_sum_surprisal', 'log_freq', 'word_length', y_col]].dropna()
    
    # Also remove infinity values
    # subset = subset[~np.isinf(subset).any(axis=1)]
    
    print(f"Rows after cleaning: {len(subset)} (removed {len(df) - len(subset)} rows)")
    
    # Prepare features and target
    X = subset[['pythia_sum_surprisal', 'log_freq', 'word_length']].values
    # Replace missing values in y_col with "0" (or any other appropriate value)
    subset[y_col] = subset[y_col].replace({".": "0"})
    # Convert y_col to numeric, coercing errors to NaN
    subset[y_col] = pd.to_numeric(subset[y_col])
    y = subset[y_col].values

    # Fit GAM
    gam = LinearGAM(s(0) + s(1) + s(2)).fit(X, y)
    
    # Plot
    XX = gam.generate_X_grid(term=0, n=500)
    plt.figure(figsize=(7, 4))
    plt.plot(XX[:, 0], gam.partial_dependence(term=0, X=XX))
    plt.title(f"Effect of Pythia Surprisal on {label}")
    plt.xlabel("Pythia-70M Surprisal")
    plt.ylabel(label)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # save the plot
    plt.savefig(f"task2/plots/gam_{label.replace(' ', '_').lower()}.png")

    # print GAM summary
    print(f"\nGAM Summary for {label}")
    gam.summary()

    # print coefficients
    print("\nCoefficients:")
    for i, coef in enumerate(gam.coef_):
        print(f"Term {i}: {coef:.4f}")


