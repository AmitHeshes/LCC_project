import pandas as pd
import csv
from tqdm import tqdm
import kenlm
import math
from datasets import load_dataset
import re
from creating_pythia_surprisals_second_try import create_surprisals_file_using_pythia


# Input file
input_eye_scan_path = f"data\\outside_critical_span_words.csv"
pre_processing_folder = f"open_part_data_and_results\\question_and_paragraph_as_context\\outside_span\\pre_processed_data"
IS_QUESTION_IN_PROMPT = True  # Set to True if you want to include the question in the prompt for Pythia
# Output file
dwell_time_path = f"{pre_processing_folder}\\ia_dwell_time.csv"
kenlm_surprisal_path = f"{pre_processing_folder}\\kenlm_surprisals.csv"
# pythia_surprisal_path = f"pre_processing_data\\pythia70M_surprisals_try_modular_version.csv"
pythia_surprisal_path = f"{pre_processing_folder}\\pythia70M_surprisals.csv"
merged_kenlm_dwell_time_path = f"{pre_processing_folder}\\merged_surprisal_dwell_kenlm.csv"
# pythia_surprisal_path = f"pre_processing_data\\pythia70M_surprisals.csv"
# merged_path = f"pre_processing_data\\merged_surprisal_dwell_kenlm_pythia.csv"
merged_path = f"{pre_processing_folder}\\merged_surprisal_dwell_kenlm_pythia.csv"
# Load KenLM model once
trained_model_path = "training_models_saved_files\\wikitext103_trigram.binary"
kenlm_trigram_model = kenlm.Model(trained_model_path)
pythia_model_name = "EleutherAI/pythia-70m"
# training data for Pythia 70M
# pythia_train_data_file_path = "training_models_saved_files\\wikitext103_train.txt"
COLUMNS_TO_ADD = ["subtlex_frequency", "word_length", "IA_FIRST_RUN_DWELL_TIME", "IA_REGRESSION_PATH_DURATION"]

##############################################################################################
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


##############################################################################################
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


##############################################################################################
def merge_surprisal_dwell_kenlm(kenlm_surprisal_path, dwell_time_path, merged_path):
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


##############################################################################################
# def is_title_line(text):
#     # Matches lines like = Title =, == Section ==, etc.
#     return bool(re.match(r"^=+.*=+$", text.strip()))

# def create_train_file_for_pythia():
#     # dataset_text_file_path = "wikitext103_train.txt"
#     # Load the wikitext-103-raw-v1 dataset
#     dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

#     # create the train file
#     with open(pythia_train_data_file_path, "w", encoding="utf-8") as f:
#         for example in dataset["train"]:
#             text = example["text"].strip()
#             if text and not is_title_line(text):
#                 f.write(text + "\n")


###############################################################################################
def merge_surprisal_dwell_kenlm_and_pythia(merged_kenlm_dwell_time_path, pythia_surprisal_path, merged_path):
    """
    Merge KenLM surprisal and dwell time dataframes on participant_id, TRIAL_INDEX, and word.
    """
    # Load both
    merged_kenlm_dwell_time_df = pd.read_csv(merged_kenlm_dwell_time_path)
    pythia_surprisal_df = pd.read_csv(pythia_surprisal_path)

    # Merge
    merged = pd.merge(merged_kenlm_dwell_time_df, pythia_surprisal_df, on=["participant_id", "TRIAL_INDEX", "IA_ID", "word"], how="inner")
    print("Finished merging dataframes")

    # Save merged file
    merged.to_csv(merged_path, index=False)
    print(f"Merged file saved: {merged_path}")

def add_columns_to_merged_file(original_data, merged_path, columns_to_add):
    merged_df = pd.read_csv(merged_path)
    key_columns = ["participant_id", "TRIAL_INDEX", "IA_ID"]
    original_df = pd.read_csv(original_data, usecols=key_columns + columns_to_add)
    result = pd.merge(merged_df, original_df, on=key_columns, how='inner')
    result.to_csv(merged_path, index=False)


##############################################################################################
def create_merge_file_from_scratch(input_eye_scan_path, dwell_time_path, kenlm_surprisal_path, merged_kenlm_dwell_time_path, 
                                   kenlm_trigram_model, pythia_surprisal_path, merged_path, columns_to_add, dwell_time_file_exists=True, kenlm_surprisal_file_exists=True, pythia_surprisal_file_exists=True):
    """
    Create both dwell time and KenLM surprisal files, then merge them.
    """
    if not dwell_time_file_exists:
        create_dwell_time_file(dwell_time_path, input_eye_scan_path)
    if not kenlm_surprisal_file_exists:
        create_surprisal_kenlm_file(kenlm_surprisal_path, input_eye_scan_path, kenlm_trigram_model)
    merge_surprisal_dwell_kenlm(kenlm_surprisal_path, dwell_time_path, merged_kenlm_dwell_time_path)
    if not pythia_surprisal_file_exists:
        create_surprisals_file_using_pythia(input_data_path=input_eye_scan_path, pythia_surprisals_path=pythia_surprisal_path, model_name=pythia_model_name, is_question_in_prompt=IS_QUESTION_IN_PROMPT)
    #     create_pythia70M_surprisals_file(input_eye_scan_path, pythia_surprisal_path)
    merge_surprisal_dwell_kenlm_and_pythia(merged_kenlm_dwell_time_path, pythia_surprisal_path, merged_path)
    add_columns_to_merged_file(input_eye_scan_path, merged_path, columns_to_add)
    


if __name__ == "__main__":
    create_merge_file_from_scratch(input_eye_scan_path, dwell_time_path, kenlm_surprisal_path, merged_kenlm_dwell_time_path, 
                                   kenlm_trigram_model, pythia_surprisal_path, merged_path, COLUMNS_TO_ADD,
                                   dwell_time_file_exists=False, kenlm_surprisal_file_exists=False, pythia_surprisal_file_exists=False)
    # merge_surprisal_dwell_kenlm_and_pythia(merged_kenlm_dwell_time_path, pythia_surprisal_path, merged_path, COLUMNS_TO_ADD)