import pandas as pd
import subprocess
from tqdm import tqdm

# Load CSV and lowercase
df = pd.read_csv("ia_Paragraph_ordinary.csv")
df['WORD'] = df['WORD'].astype(str).str.lower()

# Group words into paragraphs
paragraphs = df.groupby("PARAGRAPH_TEXT_ID")['WORD'].apply(list)

# DEBUG: try fewer paragraphs to test stability
paragraphs = paragraphs.head(2)  # Increase later if stable

# KenLM paths (adjust to your local paths if needed)
kenlm_bin = r"C:/Users/raque/Desktop/LCC/kenlm/build/bin/query.exe"
kenlm_path = r"C:/Users/raque/Desktop/LCC/wikitext_trigram.binary"

# Store results
surprisal_records = []

print("Computing surprisals (batch mode)...")

# Loop with progress bar
for pid, words in tqdm(paragraphs.items(), total=len(paragraphs)):
    padded = ["<s>", "<s>"] + words
    trigrams = [" ".join(padded[i - 2:i + 1]) for i in range(2, len(padded))]

    # Join all trigrams with newlines
    input_text = "\n".join(trigrams) + "\n"

    try:
        result = subprocess.run(
            f'"{kenlm_bin}" "{kenlm_path}"',
            input=input_text,
            capture_output=True,
            text=True,
            shell=True
        )

        if result.returncode != 0:
            print(f"⚠️ KenLM crashed: {result.stderr}")
            raise RuntimeError("KenLM query failed.")

        lines = result.stdout.strip().splitlines()
        probs = [float(line.split()[1]) for line in lines]

        for i, log10_p in enumerate(probs):
            surprisal = -log10_p * 3.32193
            surprisal_records.append((pid, i, surprisal))

    except Exception as e:
        print(f"❌ Error in paragraph {pid}: {e}")
        for i in range(len(trigrams)):
            surprisal_records.append((pid, i, None))

# Merge with original data
surprisal_df = pd.DataFrame(surprisal_records, columns=['PARAGRAPH_TEXT_ID', 'WORD_INDEX', 'KENLM_SURPRISAL'])
df['WORD_INDEX'] = df.groupby("PARAGRAPH_TEXT_ID").cumcount()
merged = pd.merge(df, surprisal_df, on=['PARAGRAPH_TEXT_ID', 'WORD_INDEX'], how='left')

# Save output
merged.to_csv("kenlm_surprisal_output.csv", index=False)
print("✅ Done! Output saved to kenlm_surprisal_output.csv")
