import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from minicons import scorer

INPUT_DATA_PATH = "data\\ia_Paragraph_ordinary.csv"
# INPUT_DATA_PATH = "data\\preview.csv"
MODEL_NAME = "EleutherAI/pythia-70m"
OUTPUT_SURPRISALS_PATH = "pre_processing_data\\pythia70M_surprisals.csv"

def create_surprisals_file_using_pythia(input_data_path, pythia_surprisals_path, model_name):
    print("Loading Pythia model and tokenizer...")
    sc = scorer.IncrementalLMScorer(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading data...")
    input_data_df = pd.read_csv(input_data_path, usecols=["participant_id", "TRIAL_INDEX", "IA_ID", "IA_LABEL", "paragraph"])
    # surprisals_df = pd.DataFrame(columns=["participant_id", "TRIAL_INDEX", "IA_ID", "word", "pythia_sum_surprisal", "pythia_average_surprisal"])
    grouped_data = input_data_df.groupby(["participant_id", "TRIAL_INDEX"])

    print("Calculating surprisals...")
    num_grouped_data = len(grouped_data)
    records = []
    for (participant_id, trial_index), group in tqdm(grouped_data, total=num_grouped_data, desc="Calculating Pythia Surprisals"):
        # Calculate common variables for the group
        paragraph = group.iloc[0]["paragraph"]
        words = paragraph.strip().split()
        words_with_spaces = [" " + word if i > 0 else word for i, word in enumerate(words)]
        tokens_per_word = [(tokenizer.tokenize(word)) for word in words_with_spaces]
        num_prev_tokens = [0]
        for tokens in tokens_per_word[:-1]:
            num_prev_tokens.append(num_prev_tokens[-1] + len(tokens))

        all_surprisals = sc.token_score(paragraph, base_two=True)

        # calculate surprisals for each word in the group
        for _, row in group.iterrows():
            ia_id = row["IA_ID"]
            word = row["IA_LABEL"]
            assert paragraph == row["paragraph"], "Paragraph mismatch in the input data"
            assert words[ia_id - 1] == word
            # prev_tokens = tokenizer.tokenize(" ".join(words[:ia_id - 1]))

            word_with_space = " " + word if ia_id > 1 else word
            curr_tokens = tokenizer.tokenize(word_with_space)

            tokens_surprisals = []
            # Check if the current word's tokens match the expected tokens and calculate tokens' surprisals
            for i in range(len(curr_tokens)):
                assert curr_tokens[i] == all_surprisals[0][num_prev_tokens[ia_id - 1] + i][0]
                tokens_surprisals.append(-all_surprisals[0][num_prev_tokens[ia_id - 1] + i][1])
                assert tokens_surprisals[i] >= 0, f"Negative surprisal found for token {curr_tokens[i]} in paragraph: {paragraph}"
            
            # calculate sum and average of the tokens' surprisals
            sum_surprisals = sum(tokens_surprisals)
            average_surprisals = sum_surprisals / len(tokens_surprisals)

            # Append the results to the DataFrame
            new_record = {
                "participant_id": participant_id,
                "TRIAL_INDEX": trial_index,
                "IA_ID": ia_id,
                "word": word,
                "pythia_sum_surprisal": sum_surprisals,
                "pythia_average_surprisal": average_surprisals
            }
            records.append(new_record)
            # surprisals_df = pd.concat([surprisals_df, pd.DataFrame([new_record])], ignore_index=True)
    
    # Save the records to a CSV file
    surprisals_df = pd.DataFrame(records)
    surprisals_df.to_csv(pythia_surprisals_path, index=False, encoding='utf-8')
    print(f"Pythia surprisals saved to {pythia_surprisals_path}")


if __name__ == "__main__":
    create_surprisals_file_using_pythia(INPUT_DATA_PATH, OUTPUT_SURPRISALS_PATH, MODEL_NAME)