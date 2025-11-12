import pandas as pd
import numpy as np
import sys
import json
import re
from json_repair import repair_json
from sentence_transformers import SentenceTransformer


data_path = sys.argv[1]
attributes_path = sys.argv[2]
attributes_n_embeddings_path = sys.argv[3]

data = pd.read_csv(data_path)

cleaned_jsons = []
for idx, row in data.iterrows():
    # print(idx)
    text = row['raw_response']
    matches = re.findall(r'\{[^{}]*\}', text)

    if len(matches) == 0:
        cleaned = ""
    else:
        target_char = "//" # Example: remove content after a semicolon

        # Escape the target character if it's a special regex character
        escaped_target_char = re.escape(target_char)
        
        # Define the regex pattern to match the target character and everything after it on the same line
        pattern = fr'{escaped_target_char}.*'
        
        # Use re.sub with the re.M (multiline) flag to apply the pattern to each line
        cleaned = re.sub(pattern, '', matches[0], flags=re.M)
        cleaned = repair_json(cleaned)
    try:
        cleaned_jsons.append(json.loads(cleaned))
    except:
        print(f"Could not json loads for {idx}")
        cleaned_jsons.append("")
        continue

data['cleaned_jsons'] = cleaned_jsons

normalized_df = pd.json_normalize(data['cleaned_jsons'])
normalized_df = normalized_df.dropna(how='all')
if 'website' in normalized_df.columns:
    normalized_df = normalized_df.drop('website',axis=1)

normalized_df.to_csv(attributes_path, index=False)
print("Attributes written to ",attributes_path)


df = normalized_df.copy()

# Step 1: Convert each merchant row into a descriptive text
def row_to_text(row):
    return (
        f"{row['merchant_name']} is a company in the {row['sector']} sector, "
        f"based in {row['city']}, {row['country']}. "
        f"It has around {row['number_of_customers']} customers and {row['daily_customers']} daily customers. "
        f"The yearly revenue is {row['yearly_revenue']} with a valuation of {row['valuation']}. "
        f"Founded in {row['founded_year']}, it has been in business for {row['years_in_business']} years "
        f"and employs approximately {row['num_employees']} people."
    )

df["text"] = df.apply(row_to_text, axis=1)

# Step 2: Load a compact and efficient embedding model (no API key needed)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 3: Generate embeddings
df["embedding"] = model.encode(df["text"].tolist(), show_progress_bar=True).tolist()

# Step 4: Inspect results
print(df[["merchant_name", "text"]].head())
print("\nEmbedding vector size:", len(df["embedding"][0]))

df.to_csv(attributes_n_embeddings_path,index=False)
print("Attributes written to ",attributes_n_embeddings_path)