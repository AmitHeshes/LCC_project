import pandas as pd

def add_columns_to_merged_file(original_data, merged_path, columns_to_add):
    merged_df = pd.read_csv(merged_path)
    key_columns = ["participant_id", "TRIAL_INDEX", "IA_ID"]
    original_df = pd.read_csv(original_data, usecols=key_columns + columns_to_add)
    result = pd.merge(merged_df, original_df, on=key_columns, how='inner')
    result.to_csv(merged_path, index=False)

if __name__ == "__main__":
    original_data = f"data\\inside_critical_span_words.csv"
    merged_path = f"open_part\\question_and_paragraph_as_context\\inside_span\\pre_processed_data\\merged_after_spilover.csv"
    COLUMNS_TO_ADD = ["wordfreq_frequency"]
    add_columns_to_merged_file(original_data, merged_path, COLUMNS_TO_ADD)