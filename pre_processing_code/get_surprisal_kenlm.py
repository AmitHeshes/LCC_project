import pandas as pd
import kenlm
import math
import csv
from tqdm import tqdm

# Load KenLM model once
model = kenlm.Model("wikitext103_trigram.binary")

# Define function to compute surprisal
def word_surprisal(model, context, word):
    full_sentence = context + " " + word
    log10_prob = model.score(full_sentence, bos=False, eos=False) - model.score(context, bos=False, eos=False)
    surprisal = -log10_prob * math.log2(10)
    return surprisal

# Prepare output CSV — open once and keep csv.writer ready
output_file = "kenlm_surprisals_simple.csv"
csv_file = "ia_Paragraph_ordinary.csv"

def create_surprisal_kenlm_file(output_file, csv_file, model):
    print("Loading CSV (needed columns only)...")
    df = pd.read_csv(csv_file, usecols=["participant_id", "TRIAL_INDEX", "IA_LABEL"])

    # Group safely by participant_id and TRIAL_INDEX - for getting context
    print("Processing trials...")
    grouped = df.groupby(["participant_id", "TRIAL_INDEX"])

    with open(output_file, "w", newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        # Write header
        writer.writerow(["participant_id", "TRIAL_INDEX", "word", "kenlm_surprisal"])

        # Loop through trials
        for (participant_id, trial_index), group in tqdm(grouped, desc="Trials"):
            context = ""

            for idx, row in group.iterrows():
                word = str(row["IA_LABEL"]).strip()
                if word in [".", ",", "", " "] or pd.isna(word):
                    surprisal = 0.0
                else:
                    surprisal = word_surprisal(model, context, word)

                # Write one row surprisal
                writer.writerow([participant_id, trial_index, word, surprisal])

                # Update context
                context = context + " " + word

    print("Surprisal file written:", output_file)
    
if __name__ == "__main__":
    create_surprisal_kenlm_file(output_file, csv_file, model)