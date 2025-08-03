import pandas as pd
import numpy as np
from pygam import LinearGAM, s
import matplotlib.pyplot as plt
import textwrap


# Load data
# INPUT_DATA_PATH = "pre_processing_data/merged_surprisal_dwell_kenlm_pythia.csv"
INPUT_DATA_PATH  = "pre_processing_data\\merged_after_spilover.csv"
frequency_column = 'wordfreq_frequency'
OUTPUT_IMAGE_PATH = "task2\\plots_new\\gam_"


COLUMNS_TO_USE = [
                        "pythia_sum_surprisal",
                        "word_length",
                        "log_freq",
                        "next_word_IA_DWELL_TIME",
                        "next_word_IA_FIRST_RUN_DWELL_TIME",
                        "next_word_IA_REGRESSION_PATH_DURATION",
                        "IA_DWELL_TIME",
                        "IA_FIRST_RUN_DWELL_TIME",
                        "IA_REGRESSION_PATH_DURATION"
]


df = pd.read_csv(INPUT_DATA_PATH)

# Control variables
df['log_freq'] = -df[frequency_column]

# List of dependent variables (reading time measures)
reading_measures = {
    "Spillover Total Fixation Duration": "next_word_IA_DWELL_TIME",
    "Spillover Gaze Duration": "next_word_IA_FIRST_RUN_DWELL_TIME",
    "Spillover Regression Path Duration": "next_word_IA_REGRESSION_PATH_DURATION",
    "Total Fixation Duration": "IA_DWELL_TIME",
    "Gaze Duration": "IA_FIRST_RUN_DWELL_TIME",
    "Regression Path Duration": "IA_REGRESSION_PATH_DURATION",
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
    XX = gam.generate_X_grid(term=0)
    plt.figure(figsize=(10, 8))
    plt.plot(XX[:, 0], gam.predict(XX), color='red', linewidth=5)
    title_text = f"Effect of Pythia Surprisal on {label}"
    wrapped_title = textwrap.fill(title_text, width=40)
    plt.title(wrapped_title, fontsize=25, fontweight='bold')
    # plt.title(f"Effect of Pythia Surprisal on {label}", fontsize=25, fontweight='bold')
    plt.xlabel("Pythia-70M Surprisal", fontsize=25)
    plt.ylabel(label, fontsize=25)
    plt.xlim(0, 40)
    plt.grid(True)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.show(block=False)  # Show the plot without blocking the script
    # plt.pause(1)  # Pause to allow the plot to render

    plt.tight_layout()
    # add scatter plot in the same figure
    # plt.scatter(subset['pythia_sum_surprisal'], y, alpha=0.5, color='gray', s=10)

    # save the plot
    plt.savefig(f"{OUTPUT_IMAGE_PATH}{label.replace(' ', '_').lower()}.png", dpi=300, bbox_inches='tight')

    # print GAM summary
    print(f"\nGAM Summary for {label}")
    gam.summary()

    # print coefficients
    print("\nCoefficients:")
    # save coefficients to a file
    with open(f"{OUTPUT_IMAGE_PATH}{label.replace(' ', '_').lower()}_coefficients.txt", "w") as f:
        f.write("Coefficients:\n")
        for i, coef in enumerate(gam.coef_):
            f.write(f"Term {i}: {coef:.4f}\n")

        # save R^2 value
        f.write(f"\nR^2: {list(gam.statistics_['pseudo_r2'].values())[0]:.4f}\n")


