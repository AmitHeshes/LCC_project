import pandas as pd

data_path = "data\\preview_information_seeking.csv"

output_folder_path = "data\\preview_"
output_inside_path = output_folder_path + "inside_critical_span_words.csv"
output_outside_path = output_folder_path + "outside_critical_span_words.csv"

df = pd.read_csv(data_path)

# Filter for inside critical span words
inside_critical_span = df[df['auxiliary_span_type'] == "critical"]
inside_critical_span.to_csv(output_inside_path, index=False)

# Filter for outside critical span words
outside_critical_span = df[df['auxiliary_span_type'] != "critical"]
outside_critical_span.to_csv(output_outside_path, index=False)

# Print the number of rows in each category
print(f"Number of inside critical span words: {len(inside_critical_span)}")
print(f"Number of outside critical span words: {len(outside_critical_span)}")
print(f"Inside critical span words saved to: {output_inside_path}")
print(f"Outside critical span words saved to: {output_outside_path}")