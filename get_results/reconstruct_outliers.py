import pandas as pd
from tqdm import tqdm

merged_path = "pre_processing_data\\merged_surprisal_dwell_kenlm_pythia.csv"
data_path = "data\\ia_Paragraph_ordinary.csv"
reconstructed_examples_file = "results_constructed_part1\\reconstructed_paragraphs.csv"

def define_regions():
    regions = {
        "High Pythia, Mid KenLM": {
            "pythia_min": 50, "pythia_max": 70,
            "kenlm_min": 20, "kenlm_max": 30,
            "description": "Pythia > 50, KenLM 25-30"
        },
        "High Pythia, Low KenLM": {
            "pythia_min": 40, "pythia_max": 50,
            "kenlm_min": 0, "kenlm_max": 10,
            "description": "Pythia 40-50, KenLM 0-5"
        },
        "Low Pythia, Mid KenLM": {
            "pythia_min": 0, "pythia_max": 10,
            "kenlm_min": 25, "kenlm_max": 35,
            "description": "Pythia 0-5, KenLM ~30"
        }
    }
    return regions

def get_examples(data, num_examples): 
    print("Selecting examples from each region...")
    regions = define_regions()
    examples = {key: [] for key in regions.keys()}

    for index, row in tqdm(data.iterrows()):
        if all(len(examples[region]) >= num_examples for region in examples):
            break
        pythia_surprisal = row['pythia70M_surprisal']
        kenlm_surprisal = row['kenlm_surprisal']
        
        for region, criteria in regions.items():
            if region in examples and len(examples[region]) < num_examples:
                if (criteria['pythia_min'] <= pythia_surprisal < criteria['pythia_max'] and
                    criteria['kenlm_min'] <= kenlm_surprisal < criteria['kenlm_max']):
                    examples[region].append(row)
                    break  # Only add to the first matching region

    return examples

def reconstruct_paragraph(merged_path, data_path, num_examples):
    print("Loading data...")
    merged_data = pd.read_csv(merged_path)
    data = pd.read_csv(data_path, usecols=['participant_id', 'TRIAL_INDEX', 'IA_ID', 'paragraph'])

    # Get examples from each region
    examples = get_examples(merged_data, num_examples)

    reconstructed_paragraphs = {}
    for region, rows in examples.items():
        reconstructed_paragraphs[region] = []
        for row in rows:
            participant_id = row['participant_id']
            trial_index = row['TRIAL_INDEX']
            ia_id = row['IA_ID']
            word = row['word']
            surprisal_kenlm = row['kenlm_surprisal']
            surprisal_pythia = row['pythia70M_surprisal']
            
            # Find the corresponding paragraph in the original data
            paragraph_row = data[(data['participant_id'] == participant_id) & 
                                 (data['TRIAL_INDEX'] == trial_index) & 
                                 (data['IA_ID'] == ia_id)]
            
            if not paragraph_row.empty:
                paragraph_text = paragraph_row['paragraph'].values[0]
                reconstructed_paragraphs[region].append((ia_id, word, surprisal_kenlm, surprisal_pythia, paragraph_text))
    
    return reconstructed_paragraphs

def main():
    print("Reconstructing paragraphs based on surprisal data...")
    reconstructed_paragraphs = reconstruct_paragraph(merged_path, data_path, 3)

    with open(reconstructed_examples_file, "w", encoding='utf-8') as f_out:
        f_out.write("Region,IA_ID,Word,KenLM Surprisal,Pythia Surprisal,Paragraph\n")
        for region, paragraphs in reconstructed_paragraphs.items():
            print(f"\nRegion: {region}")
            for ia_id, word, surprisal_kenlm, surprisal_pythia, paragraph in paragraphs:
                f_out.write(f"{region},{ia_id},{word},{surprisal_kenlm},{surprisal_pythia},{paragraph}\n")
                print(f"IA_ID: {ia_id}, Word: {word}, KenLM Surprisal: {surprisal_kenlm}, Pythia Surprisal: {surprisal_pythia}")
                print(f"Paragraph: {paragraph}\n")

if __name__ == "__main__":
    main()
    


