import streamlit as st
import pandas as pd

from langchain_community.tools import DuckDuckGoSearchRun
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI  # or use your local model wrapper
import numpy as np

import time
import os
import json
import re
from json_repair import repair_json
from sentence_transformers import SentenceTransformer
from langchain import LlamaCpp
from typing import List
import ast

data = pd.read_csv("attributes.csv")

def get_new_customer_attributes(input_value):
    llm = LlamaCpp(
        model_path = '/Users/vishaldawar/Phi-3-mini-4k-instruct-fp16.gguf',
        n_gpu_layers=-1,
        max_tokens=500,
        n_ctx = 2048,
        seed=42,
        verbose=False
    )

    search = DuckDuckGoSearchRun()


    merchants = [input_value]

    # Define the prompt
    prompt_template = """
    You are a data research assistant. Given the merchant name below and search results,
    extract structured factual information. Use only verified business facts (from official or reliable news/financial sources).

    Merchant: {merchant_name}
    Search Results:
    {search_results}

    Return a JSON object with the following fields (leave blank if not found):
    {{
        "merchant_name": "{merchant_name}",
        "country": "",
        "city": "",
        "number_of_customers": "",
        "daily_customers": "",
        "yearly_revenue": "",
        "valuation": "",
        "years_in_business": "",
        "founded_year": "",
        "sector": "",
        "num_employees": ""
    }}
    """

    prompt = PromptTemplate(
        input_variables=["merchant_name", "search_results"],
        template=prompt_template
    )

    chain = LLMChain(llm=llm, prompt=prompt)


    records = []

    for i, merchant in enumerate(merchants):
        print(f"Searching for {i}, {merchant}...")
        # Step 1: Search
        query = f"{merchant} company profile number of customers revenue valuation headquarters site:linkedin.com OR site:crunchbase.com OR site:forbes.com OR site:reuters.com"
        search_results = search.run(query)

        # Step 2: Send to LLM for extraction
        response = chain.run(merchant_name=merchant, search_results=search_results)
        
        try:
            record = eval(response)  # response should be JSON-like
        except Exception:
            record = {"merchant_name": merchant, "raw_response": response}
        
        records.append(record)
        time.sleep(2)  # polite delay to avoid hammering APIs

    # Convert to dataframe
    df = pd.DataFrame(records)
    pattern = input_value.lower().replace(" ","_").replace(".","")
    df.to_csv(f"real_merchants_{pattern}.csv", index=False)

def attributes(input_value):
    all_files = os.listdir("./")
    pattern = input_value.lower().replace(" ","_").replace(".","")
    file_name = f"real_merchants_{pattern}.csv"
    if file_name in all_files:
        pass
    else:
        get_new_customer_attributes(input_value)
    data_path = f"real_merchants_{pattern}.csv"
    attributes_path = f"attributes_{pattern}.csv"
    attributes_n_embeddings_path = f"attributes_n_embeddings_{pattern}.csv"
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
            w2 = np.array([float(x) for x in w2.replace("[","").replace("]","").split(",")],dtype='float')
            # print(w2, embedding, w2.shape, embedding.shape)
            # embedding = np.array([float(x) for x in embedding.replace("[","").replace("]","").split(",")])
            simi = findSimilarity(embedding, w2)
            all_simi.loc[len(all_simi)] = [w1, row['merchant_name'], simi]
    matches = all_simi.sort_values('score',ascending=False).head(10)[['input','match','score']]
    return matches

def get_top10_closest(input_value):
    match_data = pd.read_csv("attributes_n_embeddings.csv")
    pattern = input_value.lower().replace(" ","_").replace(".","")
    customer_data = pd.read_csv(f"attributes_n_embeddings_{pattern}.csv")

    embedding = customer_data[customer_data['merchant_name'] == input_value]['embedding'].values[0]
    embedding = ast.literal_eval(embedding)
    # print("Get top10",type(np.array(embedding)))
    # print(np.array(embedding))
    matched_data = match_customers(input_value, np.array(embedding, dtype='float'), match_data)

    matched_data_write_path = f"top10_matches_{pattern}.csv"
    matched_data.to_csv(matched_data_write_path, index=False)
    print(f"Matched data written at {matched_data_write_path}!")
    return matched_data_write_path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

def generate_synthetic_merchants(
    df: pd.DataFrame,
    n_synthetic: int = None,
    oversample_factor: float = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
):
    """
    Generate realistic synthetic merchant/transaction data using embeddings + nearest neighbors.

    Parameters:
    - df: Original DataFrame
    - n_synthetic: Total number of synthetic samples to generate (optional)
    - oversample_factor: Multiplier for oversampling (optional)
    - model_name: HuggingFace sentence transformer to embed text fields

    Returns:
    - augmented_df: pd.DataFrame (original + synthetic)
    """
    # --- Determine oversampling size
    if n_synthetic is None and oversample_factor is None:
        raise ValueError("Please specify either n_synthetic or oversample_factor")
    if oversample_factor is not None:
        n_synthetic = int(len(df) * oversample_factor)

    # --- Auto-detect text & numeric columns
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    print(f"🧠 Text columns: {text_cols}")
    print(f"🔢 Numeric columns: {numeric_cols}")
    print(f"📈 Generating {n_synthetic} synthetic samples...")

    # --- Combine text columns into semantic text for embedding
    text_data = df[text_cols].astype(str).agg(" ".join, axis=1)
    # print(text_data)
    # --- Generate embeddings
    model = SentenceTransformer(model_name)
    embeddings = model.encode(text_data, show_progress_bar=True)

    # --- Scale embeddings
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings)

    # --- Generate new synthetic embeddings (random interpolation)
    np.random.seed(42)
    idx1 = np.random.randint(0, len(embeddings), n_synthetic)
    idx2 = np.random.randint(0, len(embeddings), n_synthetic)
    lam = np.random.rand(n_synthetic, 1)
    synthetic_emb = lam * embeddings[idx1] + (1 - lam) * embeddings[idx2]
    synthetic_scaled = scaler.transform(synthetic_emb)

    # --- Match to nearest neighbors in original space
    nn = NearestNeighbors(n_neighbors=1, metric='cosine')
    nn.fit(X_scaled)
    distances, indices = nn.kneighbors(synthetic_scaled)

    synthetic_like = df.iloc[indices.flatten()].copy().reset_index(drop=True)

    # --- Add slight random noise to numeric features
    for col in numeric_cols:
        noise = np.random.normal(0, 0.05, len(synthetic_like))
        synthetic_like[col] = synthetic_like[col] + (synthetic_like[col] * noise)
        if np.issubdtype(df[col].dtype, np.integer):
            synthetic_like[col] = synthetic_like[col].round().astype(int)
        else:
            synthetic_like[col] = synthetic_like[col].round(2)

    synthetic_like["merchant"] = synthetic_like["merchant"]

    # --- Combine both
    df['type'] = 'real'
    synthetic_like['type'] = 'synthetic'
    augmented_df = pd.concat([df, synthetic_like], ignore_index=True)
    return augmented_df



# Configure the page
st.set_page_config(
    page_title="Merchant Data Search",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("🔍 Merchant Data Search")
st.markdown("Enter a merchant name to search for relevant business information")

# Create two columns for better layout
col1, col2 = st.columns([2, 1])

with col1:
    # Input field for merchant name
    merchant_name = st.text_input(
        "Merchant Name",
        placeholder="e.g., Walmart, Amazon, Starbucks...",
        help="Enter the name of the merchant you want to search for"
    )

with col2:
    st.write("")  # Spacing
    st.write("")  # Spacing
    # Search button
    search_button = st.button("🔎 Search", type="primary", use_container_width=False)

# Add a divider
st.divider()

# Initialize session state for storing results
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'last_search' not in st.session_state:
    st.session_state.last_search = ""

# Handle search action
if search_button and merchant_name:
    with st.spinner(f"Searching for information about {merchant_name}..."):
        try:
            # TODO: Replace this with your actual function call
            # Example: df = your_search_function(merchant_name)
            
            # Placeholder - Replace this section with your actual code
            # This is where you'll call your existing function that:
            # 1. Takes merchant_name as input
            # 2. Uses LLM to search the internet
            # 3. Returns a pandas DataFrame
            
            # Example integration:
            # from your_module import search_merchant_data
            # df = search_merchant_data(merchant_name)
            
            # For demonstration, creating a sample dataframe
            # REMOVE THIS and replace with your actual function
            pattern = merchant_name.lower().replace(" ","_").replace(".","")
            if merchant_name in data['merchant_name'].unique():
                df = data[data['merchant_name'] == merchant_name].T.reset_index()
            elif f"attributes_{pattern}.csv" in os.listdir("./"):
                attributes_path = f"attributes_{pattern}.csv"
                df = pd.read_csv(attributes_path).T.reset_index()
            else:
                st.success(f"✅ Retrieving data for {merchant_name}")
                attributes(merchant_name)
                
                attributes_path = f"attributes_{pattern}.csv"
                df = pd.read_csv(attributes_path).T.reset_index()
            
            df.columns = ['Attribute','Value']
            # Store results in session state
            st.session_state.search_results = df
            st.session_state.last_search = merchant_name
            
            st.success(f"✅ Successfully retrieved data for {merchant_name}")
            
        except Exception as e:
            st.error(f"❌ Error searching for {merchant_name}: {str(e)}")
            st.session_state.search_results = None

elif search_button and not merchant_name:
    st.warning("⚠️ Please enter a merchant name to search")

# Display results if available
if st.session_state.search_results is not None:
    st.subheader(f"Attributes for: {st.session_state.last_search}")
    
    # Display the dataframe
    st.dataframe(
        st.session_state.search_results,
        use_container_width=True,
        hide_index=True
    )
    
    # Add download button for CSV
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        csv = st.session_state.search_results.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"{st.session_state.last_search}_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Option to clear results
        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.search_results = None
            st.session_state.last_search = ""
            st.rerun()
    
    # Display some statistics if dataframe has multiple rows
    if len(st.session_state.search_results) > 1:
        st.divider()
        st.subheader("📊 Quick Stats")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", len(st.session_state.search_results))
        with col2:
            st.metric("Columns", len(st.session_state.search_results.columns))
        with col3:
            st.metric("Data Points", st.session_state.search_results.size)

    st.divider()
    path = get_top10_closest(merchant_name)
    matches = pd.read_csv(path)
    st.session_state.search_results = matches
    st.subheader(f"Top 10 Closest Personas to: {st.session_state.last_search}")
    st.session_state.last_search = merchant_name
    st.dataframe(
            st.session_state.search_results,
            use_container_width=True,
            hide_index=True
        )
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        csv2 = st.session_state.search_results.to_csv(index=False)
        st.download_button(
            label="📥 Download Top 10 Closest Personas",
            data=csv2,
            file_name=f"{st.session_state.last_search}_top10.csv",
            mime="text/csv",
            use_container_width=True
        )


    st.divider()
    all_transactions = pd.read_csv("all_transactions.csv")
    all_txns = all_transactions[all_transactions['merchant'].isin(matches['match'].unique())].reset_index(drop=True)
    synthetic_txns = generate_synthetic_merchants(all_txns, n_synthetic=1000)
    st.session_state.search_results = synthetic_txns
    st.subheader(f"Synthetic Transactions generated for : {st.session_state.last_search}")
    st.session_state.last_search = merchant_name
    st.dataframe(
            st.session_state.search_results,
            use_container_width=True,
            hide_index=True
        )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        csv2 = st.session_state.search_results.to_csv(index=False)
        st.download_button(
            label=f"📥 Download Synthetic Persona for {st.session_state.last_search}",
            data=csv2,
            file_name=f"{st.session_state.last_search}_synthetic_persona.csv",
            mime="text/csv",
            use_container_width=True
        )
    
# Sidebar with instructions
with st.sidebar:
    st.header("ℹ️ Instructions")
    st.markdown("""
    1. Enter the merchant name in the search box
    2. Click the **Search** button
    3. Wait for the results to load
    4. View the data in the table below
    5. Download the results as CSV if needed
    
    ### Features:
    - 🔍 Real-time merchant search
    - 📊 Display results in table format
    - 📥 Download data as CSV
    - 🗑️ Clear results option
    - 📥 Download top 10 closest customers as CSV
    """)
    
    # st.divider()
    
    # st.header("🔧 Integration Guide")
    # st.markdown("""
    # To integrate your existing code:
    
    # ```python
    # # Import your function
    # from your_module import search_function
    
    # # Replace line ~44 with:
    # df = search_function(merchant_name)
    # ```
    # """)
    
    st.divider()
    
    st.caption("Merchant Data Search Tool v1.0")