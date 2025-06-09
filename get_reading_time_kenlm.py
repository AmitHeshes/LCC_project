import pandas as pd
import csv
from tqdm import tqdm

# Output file
output_file = "ia_dwell_time_simple.csv"
csv_file = "ia_Paragraph_ordinary.csv"

def create_dwell_time_file(output_file, csv_file):
    """
    Create a CSV file with participant_id, TRIAL_INDEX, word, and IA_DWELL_TIME.
    """
        
    print("Loading CSV (needed columns only)...")
    df = pd.read_csv(csv_file, usecols=["participant_id", "TRIAL_INDEX", "IA_LABEL", "IA_DWELL_TIME"])

    # Group by participant_id and TRIAL_INDEX
    print("Processing trials...")
    grouped = df.groupby(["participant_id", "TRIAL_INDEX"])

    # Open output CSV and write properly
    with open(output_file, "w", newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        # Write header
        writer.writerow(["participant_id", "TRIAL_INDEX", "word", "IA_DWELL_TIME"])

        # Loop through trials
        for (participant_id, trial_index), group in tqdm(grouped, desc="Trials"):
            for idx, row in group.iterrows():
                word = str(row["IA_LABEL"]).strip()
                dwell_time = row["IA_DWELL_TIME"]

                # Write one row
                writer.writerow([participant_id, trial_index, word, dwell_time])

    print("Dwell time file written:", output_file)

if __name__ == "__main__":
    create_dwell_time_file(output_file, csv_file)
