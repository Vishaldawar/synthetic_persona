import pandas as pd
import numpy as np
import sys
from sklearn.metrics.pairwise import cosine_similarity


match_data_path = sys.argv[1]
customer_name = sys.argv[2]
customer_data_path = sys.argv[3]
matched_data_write_path = sys.argv[4]

data = pd.read_csv(data_path)

def findSimilarity(w1, w2):
    """
    Returns normalized levenshtein similarity between two input strings w1 and w2.
    """
    dot_product = np.dot(w1, w2)
    l2_w1 = np.linalg.norm(w1)
    l2_w2 = np.linalg.norm(w2)
    cosine_simi = dot_product / (l2_w1 * l2_w2)
    return cosine_simi

def match_customers(customer_name: str, embedding: List[float], data: pd.DataFrame):
    """
    Finds the top 3 similar job names based on the input query using levenshtein similarity w1 and w2.
    """

    all_simi = pd.DataFrame(columns=['input','match','score'])
    for idx, row in data.iterrows():
        w1 = customer_name
        for col in ['embedding']:
            w2 = row[col]
            simi = findSimilarity(embedding, w2)
            all_simi.loc[len(all_simi)] = [w1, row['merchant_name'], simi]
    matches = all_simi.sort_values('score',ascending=False).head(10)[['input','match','score']]
    return matches


match_data = pd.read_csv(match_data_path)
customer_data = pd.read_csv(customer_data_path)

embedding = customer_data[customer_data['merchant_name'] == customer_name]['embedding'].values[0]
matched_data = match_customers(customer_name, embedding, match_data)

matched_data.to_csv(matched_data_write_path, index=False)
print(f"Matched data written at {matched_data_write_path}!")