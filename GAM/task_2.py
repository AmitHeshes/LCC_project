import pandas as pd
import numpy as np
from pygam import LinearGAM, s
import matplotlib.pyplot as plt

# Load data
INPUT_DATA_PATH = "merge_file.csv"
frequency_column = 'subtlex_frequency'


df = pd.read_csv(INPUT_DATA_PATH)

# Control variables
df['log_freq'] = np.log2(df[frequency_column] + 1)
# Assuming 'word_length' column already exists

# List of dependent variables (reading time measures)
reading_measures = {
    "Total Fixation Duration": "IA_DWELL_TIME",
    "Gaze Duration": "IA_FIRST_RUN_DWELL_TIME",
    "Regression Path Duration": "IA_REGRESSION_PATH_DURATION"
}

# Loop through each reading time measure
for label, y_col in reading_measures.items():
    # Drop rows with missing data in relevant columns
    subset = df[['pythia_sum_surprisal', 'log_freq', 'word_length', y_col]]
    
    # Prepare features and target
    X = subset[['pythia_sum_surprisal', 'log_freq', 'word_length']].values
    y = subset[y_col].values

    # Fit GAM
    gam = LinearGAM(s(0) + s(1) + s(2)).fit(X, y)
    
    # Plot
    XX = gam.generate_X_grid(term=0)
    plt.figure(figsize=(7, 4))
    plt.plot(XX[:, 0], gam.partial_dependence(term=0, X=XX))
    plt.title(f"Effect of Pythia Surprisal on {label}")
    plt.xlabel("Pythia-70M Surprisal")
    plt.ylabel(label)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # print GAM summary
    print(f"\nGAM Summary for {label}")
    gam.summary()
