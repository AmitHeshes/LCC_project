import pandas as pd
import csv
from tqdm import tqdm
import kenlm
import math

# Input file
input_eye_scan_path = "ia_Paragraph_ordinary.csv"
# Output file
dwell_time_path = "ia_dwell_time_simple.csv"
kenlm_surprisal_path = "kenlm_surprisals_simple.csv"
merged_path = "merged_surprisal_dwell.csv"
# Load KenLM model once
kenlm_trigram_model = kenlm.Model("wikitext103_trigram.binary")

def create_dwell_time_file(output_dwell_time_path, input_eye_scan_path):
    """
    Create a CSV file with participant_id, TRIAL_INDEX, word, and IA_DWELL_TIME.
    """
        
    print("Loading CSV (needed columns only)...")
    df = pd.read_csv(input_eye_scan_path, usecols=["participant_id", "TRIAL_INDEX", "IA_ID", "IA_LABEL", "IA_DWELL_TIME"])

    # Group by participant_id and TRIAL_INDEX
    print("Processing trials...")
    grouped = df.groupby(["participant_id", "TRIAL_INDEX"])

    # Open output CSV and write properly
    with open(output_dwell_time_path, "w", newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        # Write header
        writer.writerow(["participant_id", "TRIAL_INDEX", "IA_ID", "word", "IA_DWELL_TIME"])

        # Loop through trials
        for (participant_id, trial_index), group in tqdm(grouped, desc="Trials"):
            for idx, row in group.iterrows():
                word = str(row["IA_LABEL"]).strip()
                dwell_time = row["IA_DWELL_TIME"]
                ia_id = row["IA_ID"]

                # Write one row
                writer.writerow([participant_id, trial_index, ia_id, word, dwell_time])

    print("Dwell time file written:", output_dwell_time_path)


# Define function to compute surprisal of kenlm model
def word_surprisal_kenlm(model, context, word):
    full_sentence = context + " " + word
    log10_prob = model.score(full_sentence, bos=False, eos=False) - model.score(context, bos=False, eos=False)
    surprisal = -log10_prob * math.log2(10)
    return surprisal


def create_surprisal_kenlm_file(output_kenlm_surprisal_path, input_eye_scan_path, model):
    print("Loading CSV (needed columns only)...")
    df = pd.read_csv(input_eye_scan_path, usecols=["participant_id", "TRIAL_INDEX", "IA_ID", "IA_LABEL"])

    # Group safely by participant_id and TRIAL_INDEX - for getting context
    print("Processing trials...")
    grouped = df.groupby(["participant_id", "TRIAL_INDEX"])

    with open(output_kenlm_surprisal_path, "w", newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        # Write header
        writer.writerow(["participant_id", "TRIAL_INDEX", "IA_ID", "word", "kenlm_surprisal"])

        # Loop through trials
        for (participant_id, trial_index), group in tqdm(grouped, desc="Trials"):
            context = ""

            for idx, row in group.iterrows():
                word = str(row["IA_LABEL"]).strip()
                if word in [".", ",", "", " "] or pd.isna(word):
                    surprisal = 0.0
                else:
                    surprisal = word_surprisal_kenlm(model, context, word)

                ia_id = row["IA_ID"]

                # Write one row surprisal
                writer.writerow([participant_id, trial_index, ia_id, word, surprisal])

                # Update context
                context = context + " " + word

    print("Surprisal file written:", output_kenlm_surprisal_path)


def merge_surprisal_dwell(kenlm_surprisal_path, dwell_time_path, merged_path):
    """
    Merge KenLM surprisal and dwell time dataframes on participant_id, TRIAL_INDEX, and word.
    """
    # Load both
    kenlm_surprisal_df = pd.read_csv(kenlm_surprisal_path)
    dwell_df = pd.read_csv(dwell_time_path)

    # Merge
    merged = pd.merge(kenlm_surprisal_df, dwell_df, on=["participant_id", "TRIAL_INDEX", "IA_ID", "word"], how="inner")
    print("Finished merging dataframes")

    # Save merged file
    merged.to_csv(merged_path, index=False)
    print(f"Merged file saved: {merged_path}")


def create_merge_file_from_scratch(input_eye_scan_path, dwell_time_path, kenlm_surprisal_path, merged_path, 
                                   kenlm_trigram_model, dwell_time_file_exists=False, kenlm_surprisal_file_exists=False):
    """
    Create both dwell time and KenLM surprisal files, then merge them.
    """
    if not dwell_time_file_exists:
        create_dwell_time_file(dwell_time_path, input_eye_scan_path)
    if not kenlm_surprisal_file_exists:
        create_surprisal_kenlm_file(kenlm_surprisal_path, input_eye_scan_path, kenlm_trigram_model)
    merge_surprisal_dwell(kenlm_surprisal_path, dwell_time_path, merged_path)


if __name__ == "__main__":
    create_merge_file_from_scratch(input_eye_scan_path, dwell_time_path, kenlm_surprisal_path, merged_path,
                                   kenlm_trigram_model,)