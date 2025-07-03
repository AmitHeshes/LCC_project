import pandas as pd


MERGED_PATH_BEFORE_SPILLOVER = "pre_processing_data\\merged_surprisal_dwell_kenlm_pythia.csv"
MERGED_PATH_AFTER_SPILLOVER = "pre_processing_data\\merged_after_spilover.csv"
PREV_COLUMNS_TO_SHIFT = ["IA_DWELL_TIME", "IA_FIRST_RUN_DWELL_TIME", "IA_REGRESSION_PATH_DURATION"]
NEW_COLUMNS_NAME = [f"next_word_{prev_column_to_shift}" for prev_column_to_shift in PREV_COLUMNS_TO_SHIFT]

def add_spilover_column(merged_path_before_spilover, merged_path_after_spilover, prev_columns_to_shift, new_columns_name):
    """
    Adds a new column to the DataFrame that contains the <prev_column_to_shift> of the next word
    NOTE: The row of the last word from each passage will be deleted
    """

    df = pd.read_csv(merged_path_before_spilover)
    df.sort_values(by=['participant_id', 'TRIAL_INDEX', 'IA_ID'], inplace=True)

    # Shift the specified column to create the new column
    for prev_column_to_shift, new_column_name in zip(prev_columns_to_shift, new_columns_name):
        df[new_column_name] = (
            df.groupby(['participant_id', 'TRIAL_INDEX'])[prev_column_to_shift].shift(-1))
    
    # Create a new column to check if the next IA_ID is the next word
    df['next_IA_ID'] = df.groupby(['participant_id', 'TRIAL_INDEX'])['IA_ID'].shift(-1)

    # Drop the last row of each group where the next IA_ID is not the next word
    df = df[df['next_IA_ID'] == df['IA_ID'] + 1]

    df.drop(columns=['next_IA_ID'], inplace=True)

    df.to_csv(merged_path_after_spilover, index=False, encoding='utf-8')


if __name__ == "__main__":
    add_spilover_column(MERGED_PATH_BEFORE_SPILLOVER, MERGED_PATH_AFTER_SPILLOVER, PREV_COLUMNS_TO_SHIFT, NEW_COLUMNS_NAME)
    print(f"Spilover reading time added and saved to {MERGED_PATH_AFTER_SPILLOVER}")