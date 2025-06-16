import pandas as pd
import math
import csv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F


def load_data(csv_file):
    """Load and group the CSV data by participant and trial."""
    df = pd.read_csv(csv_file, usecols=["participant_id", "TRIAL_INDEX", "IA_LABEL", "IA_ID"])
    grouped = df.groupby(["participant_id", "TRIAL_INDEX"])
    return grouped


def load_model(model_name="EleutherAI/pythia-70m"):
    """Load the Pythia model and tokenizer."""
    print("Loading Pythia model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    return model, tokenizer


def clean_words(words):
    """Clean and filter words, removing punctuation and empty strings."""
    words_clean = [str(w).strip() for w in words if str(w).strip() not in [".", ",", "", " "] and not pd.isna(w)]
    return words_clean


def calculate_token_surprisals(model, tokenizer, text):
    """Calculate surprisal for each token in the text."""
    # Tokenize
    encodings = tokenizer(text, return_tensors="pt", return_attention_mask=True, 
                         return_offsets_mapping=True, return_special_tokens_mask=True)
    input_ids = encodings.input_ids

    # Get model outputs
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        logits = outputs.logits

    log_probs = F.log_softmax(logits, dim=-1)

    # Calculate surprisal for each token (skip first token)
    surprisals = []
    for i in range(1, input_ids.shape[1]):
        target_token_id = input_ids[0, i]
        log_prob = log_probs[0, i-1, target_token_id].item()
        surprisal = -log_prob / math.log(2)
        surprisals.append(surprisal)

    return surprisals


def aggregate_word_surprisals(tokenizer, text, token_surprisals):
    """Aggregate token-level surprisals to word-level surprisals."""
    # Get word IDs for each token
    tokenized = tokenizer(text, return_offsets_mapping=True)
    word_ids = tokenized.word_ids()

    # Aggregate surprisal per word (sum over subword tokens)
    word_surprisals = {}
    for i, word_id in enumerate(word_ids[1:]):  # Skip first token
        if word_id is None:
            continue
        surprisal = token_surprisals[i]
        if word_id not in word_surprisals:
            word_surprisals[word_id] = surprisal
        else:
            word_surprisals[word_id] += surprisal

    return word_surprisals


def process_trial_group(model, tokenizer, group):
    """Process a single trial group and return word surprisals."""
    # Build full context
    words = [str(w).strip() for w in group["IA_LABEL"].tolist()]
    words_clean = clean_words(words)
    full_context = " ".join(words_clean)

    if not full_context.strip():
        # Handle empty context
        return {}, words, group["IA_ID"].tolist()

    # Calculate token surprisals
    token_surprisals = calculate_token_surprisals(model, tokenizer, full_context)
    
    # Aggregate to word level
    word_surprisals = aggregate_word_surprisals(tokenizer, full_context, token_surprisals)

    return word_surprisals, words, group["IA_ID"].tolist()


def get_word_surprisal(word, word_id, word_surprisals):
    """Get surprisal for a specific word, handling edge cases."""
    word = str(word).strip()
    
    if word in [".", ",", "", " "] or pd.isna(word):
        return 0.0
    elif word_id in word_surprisals:
        return word_surprisals[word_id]
    else:
        return 0.0  # Fallback


def write_results(output_file, grouped_data, model, tokenizer):
    """Process all trials and write results to CSV."""
    with open(output_file, "w", newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["participant_id", "TRIAL_INDEX", "word", "IA_ID", "pythia70M_surprisal"])

        for (participant_id, trial_index), group in tqdm(grouped_data, desc="Processing trials"):
            word_surprisals, words, ia_ids = process_trial_group(model, tokenizer, group)

            # Write results for each word
            for word_id, (word, ia_id) in enumerate(zip(words, ia_ids)):
                surprisal = get_word_surprisal(word, word_id, word_surprisals)
                writer.writerow([participant_id, trial_index, word, ia_id, surprisal])


def create_pythia70M_surprisals_file(csv_file="ia_Paragraph_ordinary.csv", output_file="pythia70M_surprisals_ID.csv", 
         model_name="EleutherAI/pythia-70m"):
    """Main function to orchestrate the surprisal calculation process."""
    # Load data
    grouped_data = load_data(csv_file)
    
    # Load model
    model, tokenizer = load_model(model_name)
    
    # Process and write results
    write_results(output_file, grouped_data, model, tokenizer)
    
    print(f"Pythia surprisal file written: {output_file}")


if __name__ == "__main__":
    # Settings
    csv_file = "ia_Paragraph_ordinary.csv"
    output_file = "pythia70M_surprisals_ID.csv"
    
    create_pythia70M_surprisals_file(csv_file, output_file)