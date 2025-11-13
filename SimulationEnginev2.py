# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 10:20:04 2025

@author: e151270
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from typing import Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.llms.base import LLM
from pydantic import Field, BaseModel
from transformers import GenerationConfig
import plotly.graph_objects as go
import plotly.express as px

# ============================================================================
# LLM WRAPPER FOR DEEPSEEK
# ============================================================================

class DeepSeekWrapper(LLM, BaseModel):
    """Wrapper for DeepSeek model to work with LangChain"""
    model: Any = Field(default=None)
    tokenizer: Any = Field(default=None)
    generation_config: Any = Field(default=None)

    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, model, tokenizer, **kwargs):
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)
        self.generation_config = GenerationConfig(
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
        )
    
    @property
    def _llm_type(self) -> str:
        return "deepseek"

    def _call(self, prompt: str, stop: Optional[list] = None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# ============================================================================
# DATA LOADING AND CACHING
# ============================================================================

@st.cache_data
def load_cluster_data():
    """Load all cluster profile and aggregate data"""
    data = {}
    
    # Load cluster profiles
    if os.path.exists('customer_cluster_profiles.csv'):
        data['customer_profiles'] = pd.read_csv('customer_cluster_profiles.csv')
    if os.path.exists('merchant_cluster_profiles.csv'):
        data['merchant_profiles'] = pd.read_csv('merchant_cluster_profiles.csv')
    if os.path.exists('issuer_cluster_profiles.csv'):
        data['issuer_profiles'] = pd.read_csv('issuer_cluster_profiles.csv')
    if os.path.exists('acquirer_cluster_profiles.csv'):
        data['acquirer_profiles'] = pd.read_csv('acquirer_cluster_profiles.csv')
    
    # Load clustered data (with individual entities)
    if os.path.exists('customer_clustered.csv'):
        data['customers'] = pd.read_csv('customer_clustered.csv')
    if os.path.exists('merchant_clustered.csv'):
        data['merchants'] = pd.read_csv('merchant_clustered.csv')
    if os.path.exists('issuer_clustered.csv'):
        data['issuers'] = pd.read_csv('issuer_clustered.csv')
    if os.path.exists('acquirer_clustered.csv'):
        data['acquirers'] = pd.read_csv('acquirer_clustered.csv')
    
    return data

@st.cache_resource
def load_llm_model(model_path: str):
    """Load DeepSeek model and tokenizer"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float32
        ).to(device)
        
        llm = DeepSeekWrapper(model=model, tokenizer=tokenizer)
        return llm, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# ============================================================================
# TRANSACTION SIMULATION ENGINE
# ============================================================================

class TransactionSimulator:
    """Core transaction simulation engine"""
    
    def __init__(self, cluster_data: Dict):
        self.data = cluster_data
        
    def get_entity_from_cluster(self, entity_type: str, cluster_id: int):
        """Get a random entity from a specific cluster"""
        entity_key = f"{entity_type}s"
        if entity_key not in self.data:
            return None
        
        entities = self.data[entity_key]
        cluster_entities = entities[entities['cluster'] == cluster_id]
        
        if len(cluster_entities) == 0:
            return None
            
        return cluster_entities.sample(1).iloc[0]
    
    def simulate_transaction(
        self,
        customer_cluster: int,
        merchant_cluster: int,
        issuer_cluster: int,
        acquirer_cluster: int,
        transaction_amount: Optional[float] = None,
        entry_mode: Optional[str] = None,
        merchant_country: Optional[str] = None
    ) -> Dict[str, Any]:
        """Simulate a transaction based on selected clusters"""
        
        # Get entities from clusters
        customer = self.get_entity_from_cluster('customer', customer_cluster)
        merchant = self.get_entity_from_cluster('merchant', merchant_cluster)
        issuer = self.get_entity_from_cluster('issuer', issuer_cluster)
        acquirer = self.get_entity_from_cluster('acquirer', acquirer_cluster)
        
        if any(x is None for x in [customer, merchant, issuer, acquirer]):
            return {"error": "Could not find entities in selected clusters"}
        
        # Determine merchant country first
        final_merchant_country = merchant_country if merchant_country else 'US'
        
        # Generate transaction details
        transaction = {
            # Entity IDs
            'customer_id': customer['customer_id'],
            'merchant_id': merchant['merchant_id'],
            'issuer_id': issuer['issuer_id'],
            'acquirer_id': acquirer['acquirer_id'],
            
            # Cluster information
            'customer_cluster': customer_cluster,
            'merchant_cluster': merchant_cluster,
            'issuer_cluster': issuer_cluster,
            'acquirer_cluster': acquirer_cluster,
            
            # Transaction amount (use provided or derive from patterns)
            'transaction_amount': transaction_amount if transaction_amount else self._generate_amount(customer, merchant),
            
            # Entry mode
            'entry_mode': entry_mode if entry_mode else self._generate_entry_mode(customer, merchant),
            
            # Merchant details
            'merchant_name': merchant.get('merchant_name', 'Unknown'),
            'mcc': merchant.get('mcc', '0000'),
            'merchant_country': final_merchant_country,
            
            # Cross-border indicator (must be set before authorization)
            'cross_border': 1 if final_merchant_country != 'US' else 0,
            
            # Timestamp
            'transaction_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            
            # Risk scores (derived from cluster characteristics)
            'fraud_score': self._calculate_fraud_score(customer, merchant, issuer),
            
            # Network details
            'network': 'MASTERCARD',
            'interchange_rate': round(np.random.uniform(1.5, 3.0), 2),
            
            # Authorization decision (will be calculated)
            'authorization_decision': None,
            'response_code': None,
            'decline_reason': None,
        }
        
        # Calculate network fee
        transaction['network_fee'] = round(transaction['transaction_amount'] * (transaction['interchange_rate'] / 100), 2)
        
        # Calculate authorization decision
        auth_result = self._authorize_transaction(transaction, customer, merchant, issuer, acquirer)
        transaction.update(auth_result)
        
        return transaction
    
    def _generate_amount(self, customer, merchant) -> float:
        """Generate transaction amount based on customer and merchant patterns"""
        base_amount = merchant.get('avg_ticket_size', customer.get('avg_transaction_amount', 50))
        variation = base_amount * 0.3
        amount = np.random.normal(base_amount, variation)
        return max(5.0, round(amount, 2))
    
    def _generate_entry_mode(self, customer, merchant) -> str:
        """Generate entry mode based on customer and merchant patterns"""
        online_ratio = customer.get('online_transaction_ratio', 0.15)
        merchant_online_ratio = merchant.get('online_ratio', 0.15)
        
        combined_online_prob = (online_ratio + merchant_online_ratio) / 2
        
        if np.random.random() < combined_online_prob:
            return 'e-commerce'
        else:
            return np.random.choice(['chip', 'contactless', 'swipe'], p=[0.5, 0.35, 0.15])
    
    def _calculate_fraud_score(self, customer, merchant, issuer) -> float:
        """Calculate fraud score based on entity risk profiles"""
        customer_fraud = customer.get('avg_fraud_score', 20)
        merchant_fraud = merchant.get('avg_fraud_score', 20)
        issuer_fraud = issuer.get('avg_fraud_score', 20)
        
        # Weighted average with some randomness
        base_score = (customer_fraud * 0.4 + merchant_fraud * 0.3 + issuer_fraud * 0.3)
        noise = np.random.normal(0, 10)
        
        return max(0, min(100, round(base_score + noise, 2)))
    
    def _authorize_transaction(self, transaction, customer, merchant, issuer, acquirer) -> Dict:
        """Determine authorization decision based on multiple factors"""
        
        decline_probability = 0.02  # Base decline rate
        
        # Factor 1: Fraud score
        if transaction['fraud_score'] > 80:
            decline_probability += 0.15
        elif transaction['fraud_score'] > 60:
            decline_probability += 0.08
        
        # Factor 2: Amount vs customer pattern
        customer_avg = customer.get('avg_transaction_amount', 50)
        if transaction['transaction_amount'] > customer_avg * 3:
            decline_probability += 0.10
        
        # Factor 3: Merchant decline rate
        merchant_decline_rate = merchant.get('decline_rate', 0.02)
        decline_probability += merchant_decline_rate * 0.5
        
        # Factor 4: Issuer approval rate
        issuer_approval_rate = issuer.get('approval_rate', 0.98)
        decline_probability += (1 - issuer_approval_rate) * 0.3
        
        # Factor 5: Cross-border
        if transaction['cross_border'] == 1:
            decline_probability += 0.05
        
        # Make decision
        if np.random.random() < decline_probability:
            # Declined
            decline_reasons = {
                '05': 'Do not honor',
                '51': 'Insufficient funds',
                '61': 'Exceeds withdrawal limit',
                '54': 'Expired card',
                '57': 'Transaction not permitted',
                '62': 'Restricted card',
            }
            response_code = np.random.choice(list(decline_reasons.keys()))
            
            return {
                'authorization_decision': 'declined',
                'response_code': response_code,
                'decline_reason': decline_reasons[response_code],
                'authorization_code': None
            }
        else:
            # Approved
            return {
                'authorization_decision': 'approved',
                'response_code': '00',
                'decline_reason': None,
                'authorization_code': f"AUTH{np.random.randint(100000, 999999)}"
            }

# ============================================================================
# LLM ANALYSIS FUNCTIONS
# ============================================================================

def analyze_transaction_with_llm(llm, transaction: Dict, cluster_data: Dict) -> str:
    """Use LLM to provide insights about the simulated transaction"""
    
    # Get cluster profile information
    customer_profile = cluster_data.get('customer_profiles')
    merchant_profile = cluster_data.get('merchant_profiles')
    issuer_profile = cluster_data.get('issuer_profiles')
    acquirer_profile = cluster_data.get('acquirer_profiles')
    
    # Build context
    context = f"""
You are analyzing a payment transaction simulation. Here are the details:

TRANSACTION DETAILS:
- Amount: ${transaction['transaction_amount']:.2f}
- Entry Mode: {transaction['entry_mode']}
- Merchant: {transaction['merchant_name']} (MCC: {transaction['mcc']})
- Country: {transaction['merchant_country']}
- Cross-Border: {'Yes' if transaction['cross_border'] else 'No'}
- Fraud Score: {transaction['fraud_score']:.2f}

AUTHORIZATION RESULT:
- Decision: {transaction['authorization_decision'].upper()}
- Response Code: {transaction['response_code']}
{f"- Decline Reason: {transaction['decline_reason']}" if transaction['decline_reason'] else ""}

CLUSTER INFORMATION:
- Customer Cluster: {transaction['customer_cluster']}
- Merchant Cluster: {transaction['merchant_cluster']}
- Issuer Cluster: {transaction['issuer_cluster']}
- Acquirer Cluster: {transaction['acquirer_cluster']}

Provide a brief analysis of this transaction including:
1. Why it was approved/declined
2. Key risk factors
3. Recommendations for optimization
4. Expected behavior based on cluster profiles

Keep the analysis concise and actionable (3-4 paragraphs).
"""
    
    try:
        response = llm(context)
        # Extract the relevant part of the response (after the prompt)
        if "Provide a brief analysis" in response:
            analysis = response.split("Provide a brief analysis")[1].split("\n\n", 1)[1] if "\n\n" in response else response
        else:
            analysis = response
        return analysis.strip()
    except Exception as e:
        return f"Error generating analysis: {str(e)}"

# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.set_page_config(
        page_title="Transaction Simulation Engine",
        page_icon="ðŸ’³",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .stApp {
            max-width: 1400px;
            margin: 0 auto;
        }
        .success-box {
            padding: 20px;
            border-radius: 5px;
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            margin: 10px 0;
        }
        .error-box {
            padding: 20px;
            border-radius: 5px;
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("ðŸ’³ Transaction Simulation Engine")
    st.markdown("Simulate realistic payment transactions using cluster-based entity selection and AI-powered analysis")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("âš™ï¸ Configuration")
        
        # Model path
        # value="C:/Users/E151270/OneDrive - Mastercard/Documents/Projects/ChatBot Development/Deep Seek/",
        model_path = st.text_input(
            "DeepSeek Model Path",
            value = "../../Downloads/deep_seek",
            help="Path to your DeepSeek model directory"
        )
        
        use_llm = st.checkbox("Enable AI Analysis", value=True)
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("This tool simulates transactions based on pre-clustered entities from your payment ecosystem.")
    
    # Load data
    with st.spinner("Loading cluster data..."):
        cluster_data = load_cluster_data()
    
    if not cluster_data:
        st.error("Could not load cluster data. Please ensure clustering.py has been run.")
        return
    
    # Initialize simulator
    simulator = TransactionSimulator(cluster_data)
    
    # Load LLM if enabled
    llm = None
    if use_llm:
        with st.spinner("Loading AI model..."):
            llm, device = load_llm_model(model_path)
            if llm:
                st.sidebar.success(f"âœ“ AI Model loaded on {device}")
    
    # Main content area - Two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("ðŸŽ¯ Entity Selection")
        
        # Customer cluster selection
        if 'customer_profiles' in cluster_data:
            customer_profiles = cluster_data['customer_profiles']
            customer_options = {
                f"Cluster {row['cluster_id']}: {row['name']} ({row['size']} customers)": row['cluster_id']
                for _, row in customer_profiles.iterrows()
            }
            selected_customer = st.selectbox("Select Customer Cluster", list(customer_options.keys()))
            customer_cluster_id = customer_options[selected_customer]
        
        # Merchant cluster selection
        if 'merchant_profiles' in cluster_data:
            merchant_profiles = cluster_data['merchant_profiles']
            merchant_options = {
                f"Cluster {row['cluster_id']}: {row['name']} ({row['size']} merchants)": row['cluster_id']
                for _, row in merchant_profiles.iterrows()
            }
            selected_merchant = st.selectbox("Select Merchant Cluster", list(merchant_options.keys()))
            merchant_cluster_id = merchant_options[selected_merchant]
        
        # Issuer cluster selection
        if 'issuer_profiles' in cluster_data:
            issuer_profiles = cluster_data['issuer_profiles']
            issuer_options = {
                f"Cluster {row['cluster_id']}: {row['name']} ({row['size']} issuers)": row['cluster_id']
                for _, row in issuer_profiles.iterrows()
            }
            selected_issuer = st.selectbox("Select Issuer Cluster", list(issuer_options.keys()))
            issuer_cluster_id = issuer_options[selected_issuer]
        
        # Acquirer cluster selection
        if 'acquirer_profiles' in cluster_data:
            acquirer_profiles = cluster_data['acquirer_profiles']
            acquirer_options = {
                f"Cluster {row['cluster_id']}: {row['name']} ({row['size']} acquirers)": row['cluster_id']
                for _, row in acquirer_profiles.iterrows()
            }
            selected_acquirer = st.selectbox("Select Acquirer Cluster", list(acquirer_options.keys()))
            acquirer_cluster_id = acquirer_options[selected_acquirer]
        
        st.header("ðŸ”§ Transaction Parameters")
        
        # Optional parameters
        with st.expander("Advanced Settings (Optional)", expanded=False):
            custom_amount = st.number_input("Transaction Amount ($)", min_value=1.0, value=None, step=10.0)
            
            entry_mode_options = ['Auto', 'chip', 'contactless', 'swipe', 'e-commerce', 'manual']
            selected_entry_mode = st.selectbox("Entry Mode", entry_mode_options)
            entry_mode = None if selected_entry_mode == 'Auto' else selected_entry_mode
            
            country_options = ['Auto', 'US', 'UK', 'CA', 'DE', 'FR', 'IN', 'AU']
            selected_country = st.selectbox("Merchant Country", country_options)
            merchant_country = None if selected_country == 'Auto' else selected_country
        
        # Simulate button
        simulate_button = st.button("ðŸš€ Simulate Transaction", type="primary", use_container_width=True)
    
    with col2:
        st.header("ðŸ“Š Simulation Results")
        
        if simulate_button:
            with st.spinner("Simulating transaction..."):
                # Simulate transaction
                transaction = simulator.simulate_transaction(
                    customer_cluster=customer_cluster_id,
                    merchant_cluster=merchant_cluster_id,
                    issuer_cluster=issuer_cluster_id,
                    acquirer_cluster=acquirer_cluster_id,
                    transaction_amount=custom_amount,
                    entry_mode=entry_mode,
                    merchant_country=merchant_country
                )
                
                if 'error' in transaction:
                    st.error(transaction['error'])
                else:
                    # Display result
                    if transaction['authorization_decision'] == 'approved':
                        st.markdown(f"""
                        <div class="success-box">
                            <h3>âœ… TRANSACTION APPROVED</h3>
                            <p><strong>Authorization Code:</strong> {transaction['authorization_code']}</p>
                            <p><strong>Response Code:</strong> {transaction['response_code']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="error-box">
                            <h3>âŒ TRANSACTION DECLINED</h3>
                            <p><strong>Response Code:</strong> {transaction['response_code']}</p>
                            <p><strong>Reason:</strong> {transaction['decline_reason']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Transaction details
                    st.subheader("Transaction Details")
                    
                    detail_col1, detail_col2 = st.columns(2)
                    
                    with detail_col1:
                        st.metric("Amount", f"${transaction['transaction_amount']:.2f}")
                        st.metric("Fraud Score", f"{transaction['fraud_score']:.1f}/100")
                        st.metric("Entry Mode", transaction['entry_mode'])
                    
                    with detail_col2:
                        st.metric("Merchant", transaction['merchant_name'])
                        st.metric("MCC", transaction['mcc'])
                        st.metric("Country", transaction['merchant_country'])
                    
                    # Entity information
                    with st.expander("Entity Information", expanded=True):
                        entity_col1, entity_col2 = st.columns(2)
                        
                        with entity_col1:
                            st.markdown(f"**Customer ID:** {transaction['customer_id']}")
                            st.markdown(f"**Customer Cluster:** {transaction['customer_cluster']}")
                            st.markdown(f"**Issuer ID:** {transaction['issuer_id']}")
                            st.markdown(f"**Issuer Cluster:** {transaction['issuer_cluster']}")
                        
                        with entity_col2:
                            st.markdown(f"**Merchant ID:** {transaction['merchant_id']}")
                            st.markdown(f"**Merchant Cluster:** {transaction['merchant_cluster']}")
                            st.markdown(f"**Acquirer ID:** {transaction['acquirer_id']}")
                            st.markdown(f"**Acquirer Cluster:** {transaction['acquirer_cluster']}")
                    
                    # Network information
                    with st.expander("Network & Fees"):
                        fee_col1, fee_col2, fee_col3 = st.columns(3)
                        
                        with fee_col1:
                            st.metric("Network", transaction['network'])
                        with fee_col2:
                            st.metric("Interchange Rate", f"{transaction['interchange_rate']:.2f}%")
                        with fee_col3:
                            st.metric("Network Fee", f"${transaction['network_fee']:.2f}")
                    
                    # AI Analysis
                    if use_llm and llm:
                        st.subheader("ðŸ¤– AI-Powered Analysis")
                        with st.spinner("Generating AI analysis..."):
                            analysis = analyze_transaction_with_llm(llm, transaction, cluster_data)
                            st.markdown(analysis)
                    
                    # Export option
                    st.subheader("ðŸ’¾ Export Transaction")
                    
                    # Convert numpy types to Python native types for JSON serialization
                    transaction_serializable = {}
                    for key, value in transaction.items():
                        if isinstance(value, (np.integer, np.int64, np.int32)):
                            transaction_serializable[key] = int(value)
                        elif isinstance(value, (np.floating, np.float64, np.float32)):
                            transaction_serializable[key] = float(value)
                        elif isinstance(value, np.ndarray):
                            transaction_serializable[key] = value.tolist()
                        else:
                            transaction_serializable[key] = value
                    
                    # Convert to JSON
                    transaction_json = json.dumps(transaction_serializable, indent=2)
                    
                    col_json, col_csv = st.columns(2)
                    with col_json:
                        st.download_button(
                            label="Download as JSON",
                            data=transaction_json,
                            file_name=f"transaction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    
                    with col_csv:
                        # Convert to DataFrame for CSV export
                        df = pd.DataFrame([transaction])
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download as CSV",
                            data=csv,
                            file_name=f"transaction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )

if __name__ == "__main__":
    main()