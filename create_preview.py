import pandas as pd

data_path = "data\\ia_Paragraph_information_seeking.csv"
output_preview_path = "data\\preview_information_seeking.csv"

df = pd.read_csv(data_path)

# Select the first 1000 rows for preview
df_preview = df.head(1000)

# Save the preview to a new CSV file
df_preview.to_csv(output_preview_path, index=False)
