import pandas as pd
from tqdm import tqdm

merged_path = "pre_processing_data\\merged_surprisal_dwell_kenlm_pythia_new_try.csv"
data_path = "data\\ia_Paragraph_ordinary.csv"
reconstructed_examples_file_pythia_sum = "results_constructed_part1_new_try\\reconstructed_paragraphs_pythia_sum.csv"
reconstructed_examples_file_pythia_average = "results_constructed_part1_new_try\\reconstructed_paragraphs_pythia_average.csv"
kenlm_surprisal_column = "kenlm_surprisal"
pythia_surprisal_column = "pythia70M_surprisal"
surprisal_variation_to_model_surprisal_column = {
    "kenlm": "kenlm_surprisal",
    "pythia": "pythia70M_surprisal",
    "pythia_sum": "pythia_sum_surprisal",
    "pythia_average": "pythia_average_surprisal"
}

def define_regions(sum_or_average):
    regions_sum = {
        "Low Pythia and High KenLM": {
            "pythia_min": 0, "pythia_max": 3,
            "kenlm_min": 25, "kenlm_max": 30,
            "description": "Pythia < 3, KenLM 25-30"
        },
        "Very High Pythia and High KenLM": {
            "pythia_min": 40, "pythia_max": 55,
            "kenlm_min": 20, "kenlm_max": 30,
            "description": "Pythia > 40, KenLM 20-30"
        },
        "High Pythia and Low KenLM": {
            "pythia_min": 15, "pythia_max": 30,
            "kenlm_min": 0, "kenlm_max": 7,
            "description": "Pythia 15-30, KenLM 0-7"
        },
    }

    regions_average = {
        "Low Pythia and High KenLM": {
            "pythia_min": 0, "pythia_max": 3,
            "kenlm_min": 25, "kenlm_max": 30,
            "description": "Pythia < 3, KenLM 25-30"
        }
    }

    if sum_or_average == "pythia_sum":
        regions = regions_sum
    elif sum_or_average == "pythia_average":
        regions = regions_average
    else:
        raise ValueError("Invalid sum_or_average value. Use 'pythia_sum' or 'pythia_average'.")
    return regions

def get_examples(data, num_examples, 
                 kenlm_surprisal_column,
                 pythia_surprisal_column, sum_or_average): 
    print("Selecting examples from each region...")
    regions = define_regions(sum_or_average)
    examples = {key: [] for key in regions.keys()}

    max_iterations = len(data)

    for i, (index, row) in enumerate(tqdm(data.iterrows())):
        if i >= max_iterations or all(len(examples[region]) >= num_examples for region in examples):
            break
        pythia_surprisal = row[pythia_surprisal_column]
        kenlm_surprisal = row[kenlm_surprisal_column]
        
        for region, criteria in regions.items():
            if region in examples and len(examples[region]) < num_examples:
                if (criteria['pythia_min'] <= pythia_surprisal < criteria['pythia_max'] and
                    criteria['kenlm_min'] <= kenlm_surprisal < criteria['kenlm_max']):
                    examples[region].append(row)
                    break  # Only add to the first matching region

    return examples

def reconstruct_paragraph(merged_path, data_path, num_examples,
                          kenlm_surprisal_column,
                          pythia_surprisal_column,
                          sum_or_average):
    print("Loading data...")
    merged_data = pd.read_csv(merged_path)
    data = pd.read_csv(data_path, usecols=['participant_id', 'TRIAL_INDEX', 'IA_ID', 'paragraph'])

    # Get examples from each region
    examples = get_examples(merged_data, num_examples,
                            kenlm_surprisal_column,
                            pythia_surprisal_column,
                            sum_or_average)

    reconstructed_paragraphs = {}
    for region, rows in examples.items():
        reconstructed_paragraphs[region] = []
        for row in rows:
            participant_id = row['participant_id']
            trial_index = row['TRIAL_INDEX']
            ia_id = row['IA_ID']
            word = row['word']
            surprisal_kenlm = row[kenlm_surprisal_column]
            surprisal_pythia = row[pythia_surprisal_column]
            
            # Find the corresponding paragraph in the original data
            paragraph_row = data[(data['participant_id'] == participant_id) & 
                                 (data['TRIAL_INDEX'] == trial_index) & 
                                 (data['IA_ID'] == ia_id)]
            
            if not paragraph_row.empty:
                paragraph_text = paragraph_row['paragraph'].values[0]
                reconstructed_paragraphs[region].append((ia_id, word, surprisal_kenlm, surprisal_pythia, paragraph_text))
    
    return reconstructed_paragraphs

def main(kenlm_surprisal_column, pythia_surprisal_column, reconstructed_examples_file,
         sum_or_average):
    print("Reconstructing paragraphs based on surprisal data...")
    reconstructed_paragraphs = reconstruct_paragraph(merged_path, data_path, 3,
                                                     kenlm_surprisal_column,
                                                     pythia_surprisal_column,
                                                     sum_or_average)

    with open(reconstructed_examples_file, "w", encoding='utf-8') as f_out:
        f_out.write("Region,IA_ID,Word,KenLM Surprisal,Pythia Surprisal,Paragraph\n")
        for region, paragraphs in reconstructed_paragraphs.items():
            print(f"\nRegion: {region}")
            for ia_id, word, surprisal_kenlm, surprisal_pythia, paragraph in paragraphs:
                f_out.write(f"{region},{ia_id},{word},{surprisal_kenlm},{surprisal_pythia},{paragraph}\n")
                print(f"IA_ID: {ia_id}, Word: {word}, KenLM Surprisal: {surprisal_kenlm}, Pythia Surprisal: {surprisal_pythia}")
                print(f"Paragraph: {paragraph}\n")

if __name__ == "__main__":
    main(kenlm_surprisal_column=kenlm_surprisal_column,
         pythia_surprisal_column=surprisal_variation_to_model_surprisal_column["pythia_sum"],
         reconstructed_examples_file=reconstructed_examples_file_pythia_sum,
         sum_or_average="pythia_sum")
    main(kenlm_surprisal_column=kenlm_surprisal_column,
         pythia_surprisal_column=surprisal_variation_to_model_surprisal_column["pythia_average"],
         reconstructed_examples_file=reconstructed_examples_file_pythia_average,
         sum_or_average="pythia_average")
    


