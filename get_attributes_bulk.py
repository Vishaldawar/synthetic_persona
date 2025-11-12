from langchain_community.tools import DuckDuckGoSearchRun
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI  # or use your local model wrapper
import pandas as pd
import time

from langchain import LlamaCpp

llm = LlamaCpp(
    model_path = '/Users/vishaldawar/Phi-3-mini-4k-instruct-fp16.gguf',
    n_gpu_layers=-1,
    max_tokens=500,
    n_ctx = 2048,
    seed=42,
    verbose=False
)

search = DuckDuckGoSearchRun()


merchants = ['Walmart', 'Amazon', 'Costco', 'Target', 'The Home Depot',
       'Kroger', 'Lowes', 'CVS Health', 'Walgreens', 'Best Buy',
       'McDonalds', 'Starbucks', 'Subway', 'Chick-fil-A', 'Taco Bell',
       'Burger King', 'Wendys', 'Dunkin', 'Chipotle', 'Dominos Pizza',
       'Apple', 'Microsoft', 'Google', 'eBay', 'PayPal', 'Etsy',
       'Shopify', 'Netflix', 'Spotify', 'Uber', 'American Airlines',
       'Delta Air Lines', 'United Airlines', 'Southwest Airlines',
       'Emirates', 'Lufthansa', 'Air France-KLM', 'British Airways',
       'Singapore Airlines', 'Cathay Pacific', 'Marriott International',
       'Hilton', 'Hyatt', 'InterContinental Hotels', 'Airbnb',
       'Booking.com', 'Expedia', 'Nike', 'Adidas', 'H&M', 'Zara',
       'Uniqlo', 'Gap Inc', 'Lululemon', 'TJX Companies', 'Ross Stores',
       'Tesla', 'Toyota', 'Ford', 'General Motors', 'Shell', 'ExxonMobil',
       'BP', 'Chevron', 'Macys', 'Nordstrom', 'Kohls',
       'Saks Fifth Avenue', 'Neiman Marcus', 'Bloomingdales', 'Disney',
       'Universal Parks', 'AMC Theatres', 'Regal Cinemas', 'Ticketmaster',
       'Verizon', 'AT&T', 'T-Mobile', 'Comcast', 'Charter Communications',
       'Tesco', 'Carrefour', 'Aldi', 'Lidl', 'Whole Foods', 'Trader Joes',
       'Publix', 'Safeway', 'KFC', 'Pizza Hut', 'Papa Johns', 'Five Guys',
       'Panera Bread', 'Shake Shack', 'IKEA', 'Wayfair',
       'Williams-Sonoma', 'Bed Bath & Beyond']

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
df.to_csv("real_merchants_info.csv", index=False)
print("Data saved to real_merchants_info.csv")
