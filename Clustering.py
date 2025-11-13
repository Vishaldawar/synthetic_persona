import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# PART 1: GENERATE SYNTHETIC TRANSACTION DATA
# ============================================================================

def generate_transaction_data(n_transactions=1000000, n_customers=10000, n_merchants=500, n_issuers=50, n_acquirers=30):
    """
    Generate synthetic transaction data with all ecosystem variables
    """
    
    # Merchant categories with typical characteristics
    merchant_categories = {
        '5411': {'name': 'Grocery Stores', 'avg_amount': 75, 'std': 30},
        '5812': {'name': 'Restaurants', 'avg_amount': 45, 'std': 25},
        '5541': {'name': 'Gas Stations', 'avg_amount': 60, 'std': 20},
        '5311': {'name': 'Department Stores', 'avg_amount': 120, 'std': 60},
        '5912': {'name': 'Pharmacies', 'avg_amount': 35, 'std': 15},
        '5999': {'name': 'E-commerce', 'avg_amount': 85, 'std': 50},
        '4121': {'name': 'Taxi/Rideshare', 'avg_amount': 25, 'std': 10},
        '7011': {'name': 'Hotels', 'avg_amount': 200, 'std': 80},
    }
    
    transactions = []
    start_date = datetime.now() - timedelta(days=90)
    
    for i in range(n_transactions):
        # Customer variables
        customer_id = f"CUST_{np.random.randint(1, n_customers+1):06d}"
        card_number = f"****{np.random.randint(1000, 9999)}"
        
        # Issuer assignment (each customer belongs to one issuer)
        issuer_id = f"ISSUER_{np.random.randint(1, n_issuers+1):03d}"
        issuer_bin = f"{np.random.randint(400000, 499999)}"
        
        # Merchant/Acquirer variables
        merchant_id = f"MERCH_{np.random.randint(1, n_merchants+1):05d}"
        acquirer_id = f"ACQ_{np.random.randint(1, n_acquirers+1):03d}"
        mcc = np.random.choice(list(merchant_categories.keys()))
        merchant_name = merchant_categories[mcc]['name']
        merchant_country = np.random.choice(['US', 'UK', 'CA', 'DE', 'FR', 'IN', 'AU'], 
                                           p=[0.5, 0.15, 0.1, 0.08, 0.07, 0.05, 0.05])
        terminal_id = f"TERM_{np.random.randint(10000, 99999)}"
        acquirer_bin = f"{np.random.randint(500000, 599999)}"
        
        # Transaction details
        base_amount = merchant_categories[mcc]['avg_amount']
        std_amount = merchant_categories[mcc]['std']
        transaction_amount = max(5, np.random.normal(base_amount, std_amount))
        transaction_currency = 'USD' if merchant_country == 'US' else np.random.choice(['GBP', 'EUR', 'CAD', 'INR'])
        
        # Timestamp with realistic patterns
        days_ago = np.random.randint(0, 90)
        hour = int(np.random.beta(5, 2) * 24)
        transaction_timestamp = start_date + timedelta(days=days_ago, hours=hour, 
                                                       minutes=np.random.randint(0, 60))
        
        entry_mode = np.random.choice(['chip', 'contactless', 'swipe', 'e-commerce', 'manual'],
                                     p=[0.4, 0.25, 0.15, 0.15, 0.05])
        transaction_type = np.random.choice(['purchase', 'refund', 'cash_advance'],
                                           p=[0.93, 0.05, 0.02])
        
        # Issuer variables
        account_balance = np.random.uniform(100, 50000)
        available_credit = np.random.uniform(1000, 20000)
        spending_limit_daily = np.random.choice([500, 1000, 2500, 5000, 10000])
        account_age_days = np.random.randint(30, 3650)  # 1 month to 10 years
        
        # Risk scoring
        fraud_score = np.random.beta(1, 10) * 100
        velocity_1h = np.random.poisson(0.5)
        velocity_24h = np.random.poisson(3)
        cardholder_risk_score = np.random.beta(2, 8) * 100
        
        # Authorization decision
        decline_probability = 0.02 + (0.1 if fraud_score > 80 else 0) + \
                            (0.05 if transaction_amount > spending_limit_daily else 0)
        authorization_decision = 'approved' if np.random.random() > decline_probability else 'declined'
        
        decline_reason_codes = ['05', '51', '61', '65', '54', '57', '62', '63']
        response_code = '00' if authorization_decision == 'approved' else np.random.choice(decline_reason_codes)
        authorization_code = f"AUTH{np.random.randint(100000, 999999)}" if authorization_decision == 'approved' else None
        decline_reason = None
        if response_code != '00':
            decline_reasons = {
                '05': 'Do not honor',
                '51': 'Insufficient funds',
                '61': 'Exceeds withdrawal limit',
                '65': 'Exceeds withdrawal frequency',
                '54': 'Expired card',
                '57': 'Transaction not permitted',
                '62': 'Restricted card',
                '63': 'Security violation'
            }
            decline_reason = decline_reasons.get(response_code, 'Other')
        
        # Network variables
        network_transaction_id = f"NET{i:012d}"
        network = np.random.choice(['MASTERCARD'], p=[1])
        interchange_rate = np.random.uniform(1.5, 3.0)
        network_fee = transaction_amount * (interchange_rate / 100)
        cross_border = 1 if merchant_country != 'US' else 0
        
        # Customer behavior indicators
        cardholder_email = f"{customer_id.lower()}@email.com"
        ip_address = f"{np.random.randint(1,255)}.{np.random.randint(1,255)}.{np.random.randint(1,255)}.{np.random.randint(1,255)}"
        device_id = f"DEV_{np.random.randint(1, n_customers+1):06d}"
        
        # 3DS authentication (for e-commerce)
        auth_3ds_result = None
        if entry_mode == 'e-commerce':
            auth_3ds_result = np.random.choice(['success', 'failed', 'not_attempted'], p=[0.85, 0.05, 0.1])
        
        # Chargeback indicator (small probability)
        chargeback = 1 if np.random.random() < 0.005 else 0
        
        transaction = {
            # Customer variables
            'customer_id': customer_id,
            'card_number': card_number,
            'cardholder_email': cardholder_email,
            'device_id': device_id,
            'ip_address': ip_address,
            
            # Issuer variables
            'issuer_id': issuer_id,
            'issuer_bin': issuer_bin,
            'account_balance': round(account_balance, 2),
            'available_credit': round(available_credit, 2),
            'spending_limit_daily': spending_limit_daily,
            'account_age_days': account_age_days,
            'cardholder_risk_score': round(cardholder_risk_score, 2),
            'fraud_score': round(fraud_score, 2),
            'velocity_1h': velocity_1h,
            'velocity_24h': velocity_24h,
            'authorization_decision': authorization_decision,
            'response_code': response_code,
            'decline_reason': decline_reason,
            'authorization_code': authorization_code,
            'auth_3ds_result': auth_3ds_result,
            
            # Merchant/Acquirer variables
            'merchant_id': merchant_id,
            'merchant_name': merchant_name,
            'merchant_category_code': mcc,
            'merchant_country': merchant_country,
            'terminal_id': terminal_id,
            'acquirer_id': acquirer_id,
            'acquirer_bin': acquirer_bin,
            'transaction_amount': round(transaction_amount, 2),
            'transaction_currency': transaction_currency,
            'transaction_timestamp': transaction_timestamp,
            'transaction_type': transaction_type,
            'entry_mode': entry_mode,
            'pos_condition_code': '00',
            'chargeback': chargeback,
            
            # Network variables
            'network_transaction_id': network_transaction_id,
            'network': network,
            'interchange_rate': round(interchange_rate, 2),
            'network_fee': round(network_fee, 2),
            'cross_border_indicator': cross_border,
        }
        
        transactions.append(transaction)
    
    return pd.DataFrame(transactions)

# ============================================================================
# PART 2: AGGREGATE FEATURES FOR CLUSTERING
# ============================================================================

def create_customer_aggregates(df):
    """Create aggregated customer-level features for clustering"""
    
    df['transaction_timestamp'] = pd.to_datetime(df['transaction_timestamp'])
    df['hour'] = df['transaction_timestamp'].dt.hour
    df['day_of_week'] = df['transaction_timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    customer_agg = df.groupby('customer_id').agg({
        'transaction_amount': ['mean', 'std', 'sum', 'count'],
        'merchant_id': lambda x: x.nunique(),
        'merchant_category_code': lambda x: x.nunique(),
        'entry_mode': lambda x: (x == 'e-commerce').sum() / len(x),
        'cross_border_indicator': 'mean',
        'hour': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.mean(),
        'is_weekend': 'mean',
        'fraud_score': 'mean',
        'velocity_24h': 'mean',
        'authorization_decision': lambda x: (x == 'declined').sum() / len(x),
        'available_credit': 'mean',
        'spending_limit_daily': 'mean',
        'chargeback': 'sum',
    }).reset_index()
    
    customer_agg.columns = [
        'customer_id', 'avg_transaction_amount', 'transaction_amount_std',
        'total_spend', 'transaction_count', 'unique_merchants', 'unique_mcc_count',
        'online_transaction_ratio', 'international_ratio', 'peak_hour',
        'weekend_ratio', 'avg_fraud_score', 'avg_velocity_24h',
        'decline_rate', 'avg_available_credit', 'avg_spending_limit', 'chargeback_count'
    ]
    
    customer_agg['transaction_frequency'] = customer_agg['transaction_count'] / 90
    customer_agg['merchant_diversity'] = customer_agg['unique_merchants'] / customer_agg['transaction_count']
    customer_agg['spend_per_merchant'] = customer_agg['total_spend'] / customer_agg['unique_merchants']
    customer_agg['amount_variance_coef'] = customer_agg['transaction_amount_std'] / customer_agg['avg_transaction_amount']
    customer_agg['chargeback_rate'] = customer_agg['chargeback_count'] / customer_agg['transaction_count']
    
    return customer_agg

def create_merchant_aggregates(df):
    """Create aggregated merchant-level features for clustering"""
    
    merchant_agg = df.groupby('merchant_id').agg({
        'transaction_amount': ['mean', 'sum', 'count'],
        'customer_id': lambda x: x.nunique(),
        'fraud_score': 'mean',
        'authorization_decision': lambda x: (x == 'declined').sum() / len(x),
        'transaction_type': lambda x: (x == 'refund').sum() / len(x),
        'cross_border_indicator': 'mean',
        'merchant_category_code': 'first',
        'merchant_name': 'first',
        'chargeback': 'sum',
        'entry_mode': lambda x: (x == 'e-commerce').sum() / len(x),
    }).reset_index()
    
    merchant_agg.columns = [
        'merchant_id', 'avg_ticket_size', 'total_volume', 'transaction_count',
        'unique_customers', 'avg_fraud_score', 'decline_rate', 'refund_rate',
        'international_customer_ratio', 'mcc', 'merchant_name',
        'chargeback_count', 'online_ratio'
    ]
    
    merchant_agg['transactions_per_customer'] = merchant_agg['transaction_count'] / merchant_agg['unique_customers']
    merchant_agg['repeat_customer_score'] = 1 / merchant_agg['transactions_per_customer']
    merchant_agg['chargeback_rate'] = merchant_agg['chargeback_count'] / merchant_agg['transaction_count']
    
    return merchant_agg

def create_issuer_aggregates(df):
    """Create aggregated issuer-level features for clustering"""
    
    issuer_agg = df.groupby('issuer_id').agg({
        # Portfolio metrics
        'customer_id': lambda x: x.nunique(),
        'card_number': lambda x: x.nunique(),
        'transaction_amount': ['sum', 'mean', 'count'],
        
        # Credit and limits
        'available_credit': 'mean',
        'spending_limit_daily': 'mean',
        'account_age_days': 'mean',
        
        # Risk metrics
        'fraud_score': 'mean',
        'cardholder_risk_score': 'mean',
        'authorization_decision': lambda x: (x == 'declined').sum() / len(x),
        'response_code': lambda x: (x == '51').sum() / len(x),  # Insufficient funds rate
        'chargeback': 'sum',
        
        # Transaction patterns
        'cross_border_indicator': 'mean',
        'entry_mode': lambda x: (x == 'e-commerce').sum() / len(x),
        'transaction_type': lambda x: (x == 'cash_advance').sum() / len(x),
        'auth_3ds_result': lambda x: (x == 'success').sum() / x.notna().sum() if x.notna().sum() > 0 else 0,
        
        # Velocity
        'velocity_24h': 'mean',
    }).reset_index()
    
    issuer_agg.columns = [
        'issuer_id', 'active_customers', 'active_cards', 'total_transaction_volume',
        'avg_transaction_amount', 'transaction_count', 'avg_available_credit',
        'avg_spending_limit', 'avg_account_age_days', 'avg_fraud_score',
        'avg_cardholder_risk_score', 'decline_rate', 'insufficient_funds_rate',
        'chargeback_count', 'cross_border_ratio', 'ecommerce_ratio',
        'cash_advance_ratio', '3ds_success_rate', 'avg_velocity_24h'
    ]
    
    # Derived metrics
    issuer_agg['transactions_per_customer'] = issuer_agg['transaction_count'] / issuer_agg['active_customers']
    issuer_agg['volume_per_customer'] = issuer_agg['total_transaction_volume'] / issuer_agg['active_customers']
    issuer_agg['chargeback_rate'] = issuer_agg['chargeback_count'] / issuer_agg['transaction_count']
    issuer_agg['approval_rate'] = 1 - issuer_agg['decline_rate']
    issuer_agg['cards_per_customer'] = issuer_agg['active_cards'] / issuer_agg['active_customers']
    
    return issuer_agg

def create_acquirer_aggregates(df):
    """Create aggregated acquirer-level features for clustering"""
    
    acquirer_agg = df.groupby('acquirer_id').agg({
        # Portfolio metrics
        'merchant_id': lambda x: x.nunique(),
        'merchant_category_code': lambda x: x.nunique(),
        'transaction_amount': ['sum', 'mean', 'count'],
        'network_fee': 'sum',
        
        # Merchant characteristics
        'customer_id': lambda x: x.nunique(),
        'merchant_country': lambda x: x.nunique(),
        
        # Risk metrics
        'fraud_score': 'mean',
        'authorization_decision': lambda x: (x == 'declined').sum() / len(x),
        'chargeback': 'sum',
        
        # Transaction patterns
        'cross_border_indicator': 'mean',
        'entry_mode': lambda x: (x == 'e-commerce').sum() / len(x),
        'transaction_type': lambda x: (x == 'refund').sum() / len(x),
        
        # Network distribution
        'network': lambda x: x.value_counts().iloc[0] if len(x) > 0 else None,
        'interchange_rate': 'mean',
    }).reset_index()
    
    acquirer_agg.columns = [
        'acquirer_id', 'merchant_portfolio_size', 'mcc_diversity',
        'total_processing_volume', 'avg_transaction_amount', 'transaction_count',
        'total_network_fees', 'unique_customers', 'country_diversity',
        'avg_fraud_score', 'decline_rate', 'chargeback_count',
        'cross_border_ratio', 'ecommerce_ratio', 'refund_rate',
        'primary_network', 'avg_interchange_rate'
    ]
    
    # Derived metrics
    acquirer_agg['transactions_per_merchant'] = acquirer_agg['transaction_count'] / acquirer_agg['merchant_portfolio_size']
    acquirer_agg['volume_per_merchant'] = acquirer_agg['total_processing_volume'] / acquirer_agg['merchant_portfolio_size']
    acquirer_agg['chargeback_rate'] = acquirer_agg['chargeback_count'] / acquirer_agg['transaction_count']
    acquirer_agg['approval_rate'] = 1 - acquirer_agg['decline_rate']
    acquirer_agg['avg_fee_per_transaction'] = acquirer_agg['total_network_fees'] / acquirer_agg['transaction_count']
    acquirer_agg['customers_per_merchant'] = acquirer_agg['unique_customers'] / acquirer_agg['merchant_portfolio_size']
    
    return acquirer_agg

# ============================================================================
# PART 3: CLUSTERING IMPLEMENTATION
# ============================================================================

def perform_clustering(df, n_clusters=5, entity_type='customer'):
    """Perform K-means clustering and analyze results"""
    
    # Select numeric features for clustering
    id_cols = ['customer_id', 'merchant_id', 'issuer_id', 'acquirer_id', 
               'mcc', 'merchant_name', 'primary_network']
    feature_cols = [col for col in df.columns if col not in id_cols]
    
    X = df[feature_cols].fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to dataframe
    df['cluster'] = clusters
    
    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['pca_1'] = X_pca[:, 0]
    df['pca_2'] = X_pca[:, 1]
    
    return df, kmeans, scaler, pca, feature_cols

def name_and_tag_clusters(df, entity_type='customer'):
    """Analyze clusters and assign meaningful names and tags"""
    
    cluster_profiles = []
    
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id]
        
        profile = {
            'cluster_id': cluster_id,
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(df) * 100
        }
        
        if entity_type == 'customer':
            profile.update({
                'avg_transaction_amount': cluster_data['avg_transaction_amount'].mean(),
                'transaction_frequency': cluster_data['transaction_frequency'].mean(),
                'online_ratio': cluster_data['online_transaction_ratio'].mean(),
                'international_ratio': cluster_data['international_ratio'].mean(),
                'decline_rate': cluster_data['decline_rate'].mean(),
                'avg_fraud_score': cluster_data['avg_fraud_score'].mean(),
                'unique_merchants': cluster_data['unique_merchants'].mean(),
            })
            
            if profile['avg_transaction_amount'] > df['avg_transaction_amount'].quantile(0.75):
                if profile['transaction_frequency'] > df['transaction_frequency'].quantile(0.75):
                    name = "High-Value Frequent Shoppers"
                    tags = ['premium', 'high-spend', 'frequent', 'valuable']
                else:
                    name = "Big Ticket Buyers"
                    tags = ['premium', 'high-spend', 'occasional', 'large-purchases']
            elif profile['online_ratio'] > 0.7:
                name = "Digital-First Customers"
                tags = ['online', 'tech-savvy', 'e-commerce', 'digital']
            elif profile['international_ratio'] > 0.3:
                name = "International Travelers"
                tags = ['traveler', 'cross-border', 'diverse', 'global']
            elif profile['transaction_frequency'] > df['transaction_frequency'].quantile(0.75):
                name = "Everyday Shoppers"
                tags = ['frequent', 'regular', 'everyday', 'loyal']
            elif profile['decline_rate'] > df['decline_rate'].quantile(0.75):
                name = "High-Risk Customers"
                tags = ['risky', 'high-decline', 'fraud-prone', 'monitoring-needed']
            else:
                name = "Occasional Shoppers"
                tags = ['infrequent', 'low-engagement', 'casual', 'basic']
                
        elif entity_type == 'merchant':
            profile.update({
                'avg_ticket_size': cluster_data['avg_ticket_size'].mean(),
                'transaction_volume': cluster_data['transaction_count'].mean(),
                'unique_customers': cluster_data['unique_customers'].mean(),
                'decline_rate': cluster_data['decline_rate'].mean(),
                'refund_rate': cluster_data['refund_rate'].mean(),
            })
            
            if profile['avg_ticket_size'] > df['avg_ticket_size'].quantile(0.75):
                name = "Premium Merchants"
                tags = ['high-value', 'premium', 'luxury']
            elif profile['transaction_volume'] > df['transaction_count'].quantile(0.75):
                name = "High-Volume Merchants"
                tags = ['busy', 'popular', 'high-traffic']
            elif profile['decline_rate'] > df['decline_rate'].quantile(0.75):
                name = "High-Risk Merchants"
                tags = ['risky', 'high-decline', 'problematic']
            elif profile['refund_rate'] > df['refund_rate'].quantile(0.75):
                name = "High-Refund Merchants"
                tags = ['returns', 'customer-issues', 'quality-concerns']
            else:
                name = "Standard Merchants"
                tags = ['typical', 'average', 'stable']
                
        elif entity_type == 'issuer':
            profile.update({
                'active_customers': cluster_data['active_customers'].mean(),
                'approval_rate': cluster_data['approval_rate'].mean(),
                'avg_fraud_score': cluster_data['avg_fraud_score'].mean(),
                'volume_per_customer': cluster_data['volume_per_customer'].mean(),
                'chargeback_rate': cluster_data['chargeback_rate'].mean(),
                'ecommerce_ratio': cluster_data['ecommerce_ratio'].mean(),
            })
            
            if profile['active_customers'] > df['active_customers'].quantile(0.75):
                if profile['approval_rate'] > df['approval_rate'].quantile(0.5):
                    name = "Large Conservative Issuers"
                    tags = ['large-portfolio', 'established', 'conservative', 'stable']
                else:
                    name = "Large Cautious Issuers"
                    tags = ['large-portfolio', 'risk-averse', 'strict-controls', 'high-decline']
            elif profile['avg_fraud_score'] > df['avg_fraud_score'].quantile(0.75):
                name = "High-Risk Portfolio Issuers"
                tags = ['risky-portfolio', 'fraud-prone', 'monitoring-heavy', 'challenging']
            elif profile['volume_per_customer'] > df['volume_per_customer'].quantile(0.75):
                name = "High-Activity Issuers"
                tags = ['active-customers', 'high-usage', 'engaged', 'frequent-transactions']
            elif profile['ecommerce_ratio'] > df['ecommerce_ratio'].quantile(0.75):
                name = "Digital-First Issuers"
                tags = ['digital', 'online-focused', 'tech-savvy', 'modern']
            elif profile['chargeback_rate'] > df['chargeback_rate'].quantile(0.75):
                name = "High-Dispute Issuers"
                tags = ['disputes', 'chargebacks', 'customer-issues', 'problematic']
            else:
                name = "Standard Issuers"
                tags = ['typical', 'balanced', 'stable', 'moderate']
                
        elif entity_type == 'acquirer':
            profile.update({
                'merchant_portfolio_size': cluster_data['merchant_portfolio_size'].mean(),
                'total_processing_volume': cluster_data['total_processing_volume'].mean(),
                'mcc_diversity': cluster_data['mcc_diversity'].mean(),
                'approval_rate': cluster_data['approval_rate'].mean(),
                'chargeback_rate': cluster_data['chargeback_rate'].mean(),
                'ecommerce_ratio': cluster_data['ecommerce_ratio'].mean(),
            })
            
            if profile['merchant_portfolio_size'] > df['merchant_portfolio_size'].quantile(0.75):
                if profile['mcc_diversity'] > df['mcc_diversity'].quantile(0.5):
                    name = "Large Diversified Acquirers"
                    tags = ['large-portfolio', 'diversified', 'multi-industry', 'enterprise']
                else:
                    name = "Large Specialized Acquirers"
                    tags = ['large-portfolio', 'specialized', 'industry-focused', 'niche']
            elif profile['total_processing_volume'] > df['total_processing_volume'].quantile(0.75):
                name = "High-Volume Acquirers"
                tags = ['high-volume', 'busy', 'major-processor', 'scale']
            elif profile['ecommerce_ratio'] > df['ecommerce_ratio'].quantile(0.75):
                name = "E-commerce Focused Acquirers"
                tags = ['digital', 'online', 'card-not-present', 'modern']
            elif profile['chargeback_rate'] > df['chargeback_rate'].quantile(0.75):
                name = "High-Risk Acquirers"
                tags = ['risky', 'chargebacks', 'disputes', 'problematic-merchants']
            elif profile['approval_rate'] < df['approval_rate'].quantile(0.25):
                name = "Struggling Acquirers"
                tags = ['low-approval', 'challenges', 'merchant-issues', 'underperforming']
            else:
                name = "Standard Acquirers"
                tags = ['typical', 'balanced', 'stable', 'moderate']
        
        profile['name'] = name
        profile['tags'] = tags
        
        cluster_profiles.append(profile)
    
    return pd.DataFrame(cluster_profiles)

# ============================================================================
# PART 4: VISUALIZATION AND REPORTING
# ============================================================================

def visualize_clusters(df, cluster_profiles, entity_type='customer'):
    """Create visualizations for cluster analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Cluster distribution
    cluster_counts = df['cluster'].value_counts().sort_index()
    axes[0, 0].bar(cluster_counts.index, cluster_counts.values, color='steelblue')
    axes[0, 0].set_xlabel('Cluster ID')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title(f'{entity_type.title()} Distribution Across Clusters')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # 2. PCA visualization
    colors = plt.cm.tab10(np.linspace(0, 1, df['cluster'].nunique()))
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id]
        axes[0, 1].scatter(cluster_data['pca_1'], cluster_data['pca_2'], 
                          c=[colors[cluster_id]], label=f'Cluster {cluster_id}', alpha=0.6)
    axes[0, 1].set_xlabel('First Principal Component')
    axes[0, 1].set_ylabel('Second Principal Component')
    axes[0, 1].set_title('Cluster Visualization (PCA)')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Key metric comparison
    if entity_type == 'customer':
        metric1 = 'avg_transaction_amount'
        metric2 = 'transaction_frequency'
        label1 = 'Avg Transaction Amount'
        label2 = 'Transaction Frequency'
    elif entity_type == 'merchant':
        metric1 = 'avg_ticket_size'
        metric2 = 'transaction_count'
        label1 = 'Avg Ticket Size'
        label2 = 'Transaction Volume'
    elif entity_type == 'issuer':
        metric1 = 'total_transaction_volume'
        metric2 = 'active_customers'
        label1 = 'Total Transaction Volume'
        label2 = 'Active Customers'
    else:  # acquirer
        metric1 = 'total_processing_volume'
        metric2 = 'merchant_portfolio_size'
        label1 = 'Total Processing Volume'
        label2 = 'Merchant Portfolio Size'
    
    cluster_means = df.groupby('cluster')[[metric1, metric2]].mean()
    x = np.arange(len(cluster_means))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, cluster_means[metric1], width, label=label1, color='coral')
    ax2 = axes[1, 0].twinx()
    ax2.bar(x + width/2, cluster_means[metric2], width, label=label2, color='lightblue')
    
    axes[1, 0].set_xlabel('Cluster ID')
    axes[1, 0].set_ylabel(label1, color='coral')
    ax2.set_ylabel(label2, color='lightblue')
    axes[1, 0].set_title('Key Metrics by Cluster')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(cluster_means.index)
    axes[1, 0].legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # 4. Cluster profile summary
    axes[1, 1].axis('off')
    profile_text = "Cluster Profiles:\n\n"
    for _, profile in cluster_profiles.iterrows():
        profile_text += f"Cluster {profile['cluster_id']}: {profile['name']}\n"
        profile_text += f"  Size: {profile['size']} ({profile['percentage']:.1f}%)\n"
        profile_text += f"  Tags: {', '.join(profile['tags'])}\n\n"
    
    axes[1, 1].text(0.1, 0.9, profile_text, transform=axes[1, 1].transAxes,
                   fontsize=9, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    return fig

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("TRANSACTION DATA GENERATION & CLUSTERING ANALYSIS")
    print("ALL ENTITIES: CUSTOMER | MERCHANT | ISSUER | ACQUIRER")
    print("=" * 80)
    
    # Step 1: Generate transaction data
    print("\n[1/9] Generating synthetic transaction data...")
    transactions_df = generate_transaction_data(
        n_transactions=10000, 
        n_customers=1000, 
        n_merchants=500,
        n_issuers=50,
        n_acquirers=30
    )
    print(f"✓ Generated {len(transactions_df)} transactions")
    print(f"✓ Columns: {len(transactions_df.columns)} total")
    print(f"  - Customers: {transactions_df['customer_id'].nunique()}")
    print(f"  - Merchants: {transactions_df['merchant_id'].nunique()}")
    print(f"  - Issuers: {transactions_df['issuer_id'].nunique()}")
    print(f"  - Acquirers: {transactions_df['acquirer_id'].nunique()}")
    
    transactions_df.to_csv('transactions_raw.csv', index=False)
    print("✓ Saved to 'transactions_raw.csv'")
    
    # Step 2: Create customer aggregates and clustering
    print("\n[2/9] Creating customer-level aggregated features...")
    customer_agg = create_customer_aggregates(transactions_df)
    print(f"✓ Aggregated {len(customer_agg)} customers with {len(customer_agg.columns)-1} features")
    customer_agg.to_csv('customer_aggregates.csv', index=False)
    
    print("\n[3/9] Performing customer clustering...")
    customer_clustered, kmeans_cust, scaler_cust, pca_cust, features_cust = \
        perform_clustering(customer_agg, n_clusters=6, entity_type='customer')
    customer_profiles = name_and_tag_clusters(customer_clustered, entity_type='customer')
    
    print("\n" + "=" * 80)
    print("CUSTOMER CLUSTER PROFILES")
    print("=" * 80)
    for _, profile in customer_profiles.iterrows():
        print(f"\nCluster {profile['cluster_id']}: {profile['name']}")
        print(f"  Size: {profile['size']} customers ({profile['percentage']:.1f}%)")
        print(f"  Tags: {', '.join(profile['tags'])}")
        print(f"  Avg Transaction: ${profile['avg_transaction_amount']:.2f}")
        print(f"  Frequency: {profile['transaction_frequency']:.2f} txns/day")
        print(f"  Decline Rate: {profile['decline_rate']*100:.2f}%")
    
    customer_profiles.to_csv('customer_cluster_profiles.csv', index=False)
    customer_clustered.to_csv('customer_clustered.csv', index=False)
    
    # Step 3: Create merchant aggregates and clustering
    print("\n[4/9] Creating merchant-level aggregated features...")
    merchant_agg = create_merchant_aggregates(transactions_df)
    print(f"✓ Aggregated {len(merchant_agg)} merchants with {len(merchant_agg.columns)-1} features")
    merchant_agg.to_csv('merchant_aggregates.csv', index=False)
    
    print("\n[5/9] Performing merchant clustering...")
    merchant_clustered, kmeans_merch, scaler_merch, pca_merch, features_merch = \
        perform_clustering(merchant_agg, n_clusters=5, entity_type='merchant')
    merchant_profiles = name_and_tag_clusters(merchant_clustered, entity_type='merchant')
    
    print("\n" + "=" * 80)
    print("MERCHANT CLUSTER PROFILES")
    print("=" * 80)
    for _, profile in merchant_profiles.iterrows():
        print(f"\nCluster {profile['cluster_id']}: {profile['name']}")
        print(f"  Size: {profile['size']} merchants ({profile['percentage']:.1f}%)")
        print(f"  Tags: {', '.join(profile['tags'])}")
        print(f"  Avg Ticket: ${profile['avg_ticket_size']:.2f}")
        print(f"  Transaction Volume: {profile['transaction_volume']:.0f}")
    
    merchant_profiles.to_csv('merchant_cluster_profiles.csv', index=False)
    merchant_clustered.to_csv('merchant_clustered.csv', index=False)
    
    # Step 4: Create issuer aggregates and clustering
    print("\n[6/9] Creating issuer-level aggregated features...")
    issuer_agg = create_issuer_aggregates(transactions_df)
    print(f"✓ Aggregated {len(issuer_agg)} issuers with {len(issuer_agg.columns)-1} features")
    issuer_agg.to_csv('issuer_aggregates.csv', index=False)
    
    print("\n[7/9] Performing issuer clustering...")
    issuer_clustered, kmeans_issuer, scaler_issuer, pca_issuer, features_issuer = \
        perform_clustering(issuer_agg, n_clusters=5, entity_type='issuer')
    issuer_profiles = name_and_tag_clusters(issuer_clustered, entity_type='issuer')
    
    print("\n" + "=" * 80)
    print("ISSUER CLUSTER PROFILES")
    print("=" * 80)
    for _, profile in issuer_profiles.iterrows():
        print(f"\nCluster {profile['cluster_id']}: {profile['name']}")
        print(f"  Size: {profile['size']} issuers ({profile['percentage']:.1f}%)")
        print(f"  Tags: {', '.join(profile['tags'])}")
        print(f"  Active Customers: {profile['active_customers']:.0f}")
        print(f"  Approval Rate: {profile['approval_rate']*100:.1f}%")
        print(f"  Volume per Customer: ${profile['volume_per_customer']:.2f}")
        print(f"  Chargeback Rate: {profile['chargeback_rate']*100:.3f}%")
    
    issuer_profiles.to_csv('issuer_cluster_profiles.csv', index=False)
    issuer_clustered.to_csv('issuer_clustered.csv', index=False)
    
    # Step 5: Create acquirer aggregates and clustering
    print("\n[8/9] Creating acquirer-level aggregated features...")
    acquirer_agg = create_acquirer_aggregates(transactions_df)
    print(f"✓ Aggregated {len(acquirer_agg)} acquirers with {len(acquirer_agg.columns)-1} features")
    acquirer_agg.to_csv('acquirer_aggregates.csv', index=False)
    
    print("\n[9/9] Performing acquirer clustering...")
    acquirer_clustered, kmeans_acq, scaler_acq, pca_acq, features_acq = \
        perform_clustering(acquirer_agg, n_clusters=5, entity_type='acquirer')
    acquirer_profiles = name_and_tag_clusters(acquirer_clustered, entity_type='acquirer')
    
    print("\n" + "=" * 80)
    print("ACQUIRER CLUSTER PROFILES")
    print("=" * 80)
    for _, profile in acquirer_profiles.iterrows():
        print(f"\nCluster {profile['cluster_id']}: {profile['name']}")
        print(f"  Size: {profile['size']} acquirers ({profile['percentage']:.1f}%)")
        print(f"  Tags: {', '.join(profile['tags'])}")
        print(f"  Merchant Portfolio: {profile['merchant_portfolio_size']:.0f}")
        print(f"  Processing Volume: ${profile['total_processing_volume']:.2f}")
        print(f"  MCC Diversity: {profile['mcc_diversity']:.1f}")
        print(f"  Approval Rate: {profile['approval_rate']*100:.1f}%")
    
    acquirer_profiles.to_csv('acquirer_cluster_profiles.csv', index=False)
    acquirer_clustered.to_csv('acquirer_clustered.csv', index=False)
    
    # Step 6: Create labelled transaction dataset
    print("\n" + "=" * 80)
    print("CREATING LABELLED TRANSACTION DATASET")
    print("=" * 80)
    
    print("\nMerging cluster labels back to transactions...")
    
    # Create mapping dictionaries for cluster labels
    customer_cluster_map = customer_clustered[['customer_id', 'cluster']].rename(
        columns={'cluster': 'customer_cluster'}
    )
    merchant_cluster_map = merchant_clustered[['merchant_id', 'cluster']].rename(
        columns={'cluster': 'merchant_cluster'}
    )
    issuer_cluster_map = issuer_clustered[['issuer_id', 'cluster']].rename(
        columns={'cluster': 'issuer_cluster'}
    )
    acquirer_cluster_map = acquirer_clustered[['acquirer_id', 'cluster']].rename(
        columns={'cluster': 'acquirer_cluster'}
    )
    
    # Create mapping dictionaries for cluster names
    customer_name_map = customer_profiles[['cluster_id', 'name']].rename(
        columns={'cluster_id': 'customer_cluster', 'name': 'customer_cluster_name'}
    )
    merchant_name_map = merchant_profiles[['cluster_id', 'name']].rename(
        columns={'cluster_id': 'merchant_cluster', 'name': 'merchant_cluster_name'}
    )
    issuer_name_map = issuer_profiles[['cluster_id', 'name']].rename(
        columns={'cluster_id': 'issuer_cluster', 'name': 'issuer_cluster_name'}
    )
    acquirer_name_map = acquirer_profiles[['cluster_id', 'name']].rename(
        columns={'cluster_id': 'acquirer_cluster', 'name': 'acquirer_cluster_name'}
    )
    
    # Merge all cluster labels to transactions
    transactions_labelled = transactions_df.copy()
    
    # Merge cluster IDs
    transactions_labelled = transactions_labelled.merge(
        customer_cluster_map, on='customer_id', how='left'
    )
    transactions_labelled = transactions_labelled.merge(
        merchant_cluster_map, on='merchant_id', how='left'
    )
    transactions_labelled = transactions_labelled.merge(
        issuer_cluster_map, on='issuer_id', how='left'
    )
    transactions_labelled = transactions_labelled.merge(
        acquirer_cluster_map, on='acquirer_id', how='left'
    )
    
    # Merge cluster names
    transactions_labelled = transactions_labelled.merge(
        customer_name_map, on='customer_cluster', how='left'
    )
    transactions_labelled = transactions_labelled.merge(
        merchant_name_map, on='merchant_cluster', how='left'
    )
    transactions_labelled = transactions_labelled.merge(
        issuer_name_map, on='issuer_cluster', how='left'
    )
    transactions_labelled = transactions_labelled.merge(
        acquirer_name_map, on='acquirer_cluster', how='left'
    )
    
    # Reorder columns to put cluster labels together
    cluster_cols = [
        'customer_cluster', 'customer_cluster_name',
        'merchant_cluster', 'merchant_cluster_name',
        'issuer_cluster', 'issuer_cluster_name',
        'acquirer_cluster', 'acquirer_cluster_name'
    ]
    
    other_cols = [col for col in transactions_labelled.columns if col not in cluster_cols]
    transactions_labelled = transactions_labelled[other_cols + cluster_cols]
    
    # Save labelled transactions
    transactions_labelled.to_csv('transactions_labelled.csv', index=False)
    print(f"✓ Created labelled transaction dataset with {len(transactions_labelled)} rows")
    print(f"✓ Added 8 cluster label columns (4 cluster IDs + 4 cluster names)")
    print("✓ Saved to 'transactions_labelled.csv'")
    
    # Print sample statistics
    print("\n📊 Labelled Dataset Statistics:")
    print(f"  - Customer Clusters: {transactions_labelled['customer_cluster'].nunique()}")
    print(f"  - Merchant Clusters: {transactions_labelled['merchant_cluster'].nunique()}")
    print(f"  - Issuer Clusters: {transactions_labelled['issuer_cluster'].nunique()}")
    print(f"  - Acquirer Clusters: {transactions_labelled['acquirer_cluster'].nunique()}")
    
    # Show example cluster combinations
    print("\n🔍 Top 5 Cluster Combinations:")
    combo_counts = transactions_labelled.groupby([
        'customer_cluster_name', 'merchant_cluster_name', 
        'issuer_cluster_name', 'acquirer_cluster_name'
    ]).size().sort_values(ascending=False).head(5)
    
    for idx, (combo, count) in enumerate(combo_counts.items(), 1):
        cust, merch, iss, acq = combo
        print(f"\n  {idx}. {count} transactions:")
        print(f"     Customer: {cust}")
        print(f"     Merchant: {merch}")
        print(f"     Issuer: {iss}")
        print(f"     Acquirer: {acq}")
    
    # Step 7: Create visualizations
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    
    print("\nGenerating customer cluster visualization...")
    fig_customer = visualize_clusters(customer_clustered, customer_profiles, 'customer')
    plt.savefig('customer_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved to 'customer_clusters.png'")
    
    print("\nGenerating merchant cluster visualization...")
    fig_merchant = visualize_clusters(merchant_clustered, merchant_profiles, 'merchant')
    plt.savefig('merchant_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved to 'merchant_clusters.png'")
    
    print("\nGenerating issuer cluster visualization...")
    fig_issuer = visualize_clusters(issuer_clustered, issuer_profiles, 'issuer')
    plt.savefig('issuer_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved to 'issuer_clusters.png'")
    
    print("\nGenerating acquirer cluster visualization...")
    fig_acquirer = visualize_clusters(acquirer_clustered, acquirer_profiles, 'acquirer')
    plt.savefig('acquirer_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved to 'acquirer_clusters.png'")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("=" * 80)
    print(f"\n📊 Transaction Data:")
    print(f"  - Total Transactions: {len(transactions_df):,}")
    print(f"  - Date Range: {transactions_df['transaction_timestamp'].min()} to {transactions_df['transaction_timestamp'].max()}")
    print(f"  - Total Volume: ${transactions_df['transaction_amount'].sum():,.2f}")
    print(f"  - Approval Rate: {(transactions_df['authorization_decision']=='approved').mean()*100:.2f}%")
    
    print(f"\n👥 Customer Clustering:")
    print(f"  - Total Customers: {len(customer_clustered)}")
    print(f"  - Number of Clusters: {customer_clustered['cluster'].nunique()}")
    print(f"  - Largest Cluster: {customer_profiles.loc[customer_profiles['size'].idxmax(), 'name']}")
    
    print(f"\n🏪 Merchant Clustering:")
    print(f"  - Total Merchants: {len(merchant_clustered)}")
    print(f"  - Number of Clusters: {merchant_clustered['cluster'].nunique()}")
    print(f"  - Largest Cluster: {merchant_profiles.loc[merchant_profiles['size'].idxmax(), 'name']}")
    
    print(f"\n🏦 Issuer Clustering:")
    print(f"  - Total Issuers: {len(issuer_clustered)}")
    print(f"  - Number of Clusters: {issuer_clustered['cluster'].nunique()}")
    print(f"  - Largest Cluster: {issuer_profiles.loc[issuer_profiles['size'].idxmax(), 'name']}")
    
    print(f"\n💳 Acquirer Clustering:")
    print(f"  - Total Acquirers: {len(acquirer_clustered)}")
    print(f"  - Number of Clusters: {acquirer_clustered['cluster'].nunique()}")
    print(f"  - Largest Cluster: {acquirer_profiles.loc[acquirer_profiles['size'].idxmax(), 'name']}")
    
    print("\n" + "=" * 80)
    print("GENERATED FILES")
    print("=" * 80)
    print("\n📁 Raw Data:")
    print("  1. transactions_raw.csv")
    print("  2. transactions_labelled.csv (WITH ALL CLUSTER LABELS!)")
    
    print("\n📁 Aggregated Features:")
    print("  3. customer_aggregates.csv")
    print("  4. merchant_aggregates.csv")
    print("  5. issuer_aggregates.csv")
    print("  6. acquirer_aggregates.csv")
    
    print("\n📁 Clustered Data:")
    print("  7. customer_clustered.csv")
    print("  8. merchant_clustered.csv")
    print("  9. issuer_clustered.csv")
    print("  10. acquirer_clustered.csv")
    
    print("\n📁 Cluster Profiles:")
    print("  11. customer_cluster_profiles.csv")
    print("  12. merchant_cluster_profiles.csv")
    print("  13. issuer_cluster_profiles.csv")
    print("  14. acquirer_cluster_profiles.csv")
    
    print("\n📁 Visualizations:")
    print("  15. customer_clusters.png")
    print("  16. merchant_clusters.png")
    print("  17. issuer_clusters.png")
    print("  18. acquirer_clusters.png")
    
    print("\n" + "=" * 80)
    print("✅ ALL DONE! Ready for analysis.")
    print("=" * 80)