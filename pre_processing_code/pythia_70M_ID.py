# pythia70M_surprisals_fast.py

import pandas as pd
import math
import csv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

# Settings
csv_file = "ia_Paragraph_ordinary.csv"
output_file = "pythia70M_surprisals_ID.csv"

# Load CSV with IA_ID
df = pd.read_csv(csv_file, usecols=["participant_id", "TRIAL_INDEX", "IA_LABEL", "IA_ID"])
grouped = df.groupby(["participant_id", "TRIAL_INDEX"])

# Load Pythia 70M
print("Loading Pythia model...")
model_name = "EleutherAI/pythia-70m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
model.eval()

# Write output CSV
with open(output_file, "w", newline='', encoding='utf-8') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(["participant_id", "TRIAL_INDEX", "word", "IA_ID", "pythia70M_surprisal"])

    for (participant_id, trial_index), group in tqdm(grouped, desc="Trials"):
        # Build full context
        words = [str(w).strip() for w in group["IA_LABEL"].tolist()]
        words_clean = [w for w in words if w not in [".", ",", "", " "] and not pd.isna(w)]
        full_context = " ".join(words_clean)

        # Tokenize with word alignment
        encodings = tokenizer(full_context, return_tensors="pt", return_attention_mask=True, return_offsets_mapping=True, return_special_tokens_mask=True)
        input_ids = encodings.input_ids

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            logits = outputs.logits

        log_probs = F.log_softmax(logits, dim=-1)

        # Surprisal for each token (skip first token)
        surprisals = []
        for i in range(1, input_ids.shape[1]):
            target_token_id = input_ids[0, i]
            log_prob = log_probs[0, i-1, target_token_id].item()
            surprisal = -log_prob / math.log(2)
            surprisals.append(surprisal)

        # Map tokens → words
        tokenized = tokenizer(full_context, return_tensors="pt", return_attention_mask=True, return_offsets_mapping=True)
        tokens = tokenizer.convert_ids_to_tokens(tokenized.input_ids[0])
        word_ids = tokenizer(full_context, return_offsets_mapping=True).word_ids()

        # Aggregate surprisal per word (sum over subword tokens)
        word_surprisals = {}
        for i, word_id in enumerate(word_ids[1:]):  # Skip first token
            if word_id is None:
                continue
            s = surprisals[i]
            if word_id not in word_surprisals:
                word_surprisals[word_id] = s
            else:
                word_surprisals[word_id] += s

        # === FINAL: Correct writing loop ===
        # Loop over words + IA_IDs with correct alignment
        for word_id, (word, ia_id) in enumerate(zip(words, group["IA_ID"].tolist())):
            word = str(word).strip()

            if word in [".", ",", "", " "] or pd.isna(word):
                surprisal = 0.0
            elif word_id in word_surprisals:
                surprisal = word_surprisals[word_id]
            else:
                surprisal = 0.0  # Fallback

            writer.writerow([participant_id, trial_index, word, ia_id, surprisal])

print("Pythia surprisal file written:", output_file)
