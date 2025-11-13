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
# PART 1: GENERATE SYNTHETIC TRANSACTION DATA WITH CUSTOMER ARCHETYPES
# ============================================================================

def generate_transaction_data(n_transactions=100000, n_customers=5000, n_merchants=500, n_issuers=50, n_acquirers=30):
    """
    Generate synthetic transaction data with distinct customer behavior patterns
    """
    
    # Define customer archetypes with distinct behaviors
    customer_archetypes = {
        'high_value_frequent': {
            'proportion': 0.10,
            'avg_amount_range': (150, 300),
            'freq_per_day_range': (2, 5),
            'online_prob': 0.4,
            'international_prob': 0.15,
            'fraud_score_range': (10, 30),
            'merchant_diversity': (0.7, 0.9),
            'weekend_prob': 0.3
        },
        'digital_native': {
            'proportion': 0.15,
            'avg_amount_range': (50, 100),
            'freq_per_day_range': (1, 3),
            'online_prob': 0.85,
            'international_prob': 0.1,
            'fraud_score_range': (15, 35),
            'merchant_diversity': (0.6, 0.8),
            'weekend_prob': 0.5
        },
        'international_traveler': {
            'proportion': 0.12,
            'avg_amount_range': (80, 150),
            'freq_per_day_range': (1.5, 3),
            'online_prob': 0.3,
            'international_prob': 0.6,
            'fraud_score_range': (20, 40),
            'merchant_diversity': (0.8, 0.95),
            'weekend_prob': 0.4
        },
        'everyday_shopper': {
            'proportion': 0.25,
            'avg_amount_range': (30, 60),
            'freq_per_day_range': (2.5, 4),
            'online_prob': 0.25,
            'international_prob': 0.05,
            'fraud_score_range': (10, 25),
            'merchant_diversity': (0.4, 0.6),
            'weekend_prob': 0.35
        },
        'occasional_shopper': {
            'proportion': 0.20,
            'avg_amount_range': (40, 80),
            'freq_per_day_range': (0.3, 0.8),
            'online_prob': 0.35,
            'international_prob': 0.08,
            'fraud_score_range': (15, 30),
            'merchant_diversity': (0.5, 0.7),
            'weekend_prob': 0.45
        },
        'high_risk': {
            'proportion': 0.08,
            'avg_amount_range': (60, 120),
            'freq_per_day_range': (1, 2.5),
            'online_prob': 0.55,
            'international_prob': 0.25,
            'fraud_score_range': (60, 90),
            'merchant_diversity': (0.65, 0.85),
            'weekend_prob': 0.4
        },
        'big_ticket_buyer': {
            'proportion': 0.10,
            'avg_amount_range': (200, 500),
            'freq_per_day_range': (0.2, 0.6),
            'online_prob': 0.3,
            'international_prob': 0.12,
            'fraud_score_range': (12, 28),
            'merchant_diversity': (0.3, 0.5),
            'weekend_prob': 0.5
        }
    }
    
    # Merchant categories
    # merchant_categories = {
    #     '5411': {'name': 'Grocery Stores', 'avg_amount': 75, 'std': 30},
    #     '5812': {'name': 'Restaurants', 'avg_amount': 45, 'std': 25},
    #     '5541': {'name': 'Gas Stations', 'avg_amount': 60, 'std': 20},
    #     '5311': {'name': 'Department Stores', 'avg_amount': 120, 'std': 60},
    #     '5912': {'name': 'Pharmacies', 'avg_amount': 35, 'std': 15},
    #     '5999': {'name': 'E-commerce', 'avg_amount': 85, 'std': 50},
    #     '4121': {'name': 'Taxi/Rideshare', 'avg_amount': 25, 'std': 10},
    #     '7011': {'name': 'Hotels', 'avg_amount': 200, 'std': 80},
    # }
    merchant_categories = {
        '2847': {'name': 'Cathay Pacific', 'avg_amount': 303.32, 'std': 170.7},
        '7851': {'name': 'Carrefour', 'avg_amount': 422.35, 'std': 89.16},
        '8211': {'name': 'Walmart', 'avg_amount': 279.72, 'std': 92.3},
        '8865': {'name': 'Panera Bread', 'avg_amount': 172.42, 'std': 71.66},
        '4317': {'name': 'Airbnb', 'avg_amount': 251.62, 'std': 74.93},
        '7811': {'name': 'Dunkin', 'avg_amount': 308.93, 'std': 159.94},
        '1164': {'name': 'Lidl', 'avg_amount': 169.58, 'std': 53.67},
        '8060': {'name': 'Walgreens', 'avg_amount': 112.97, 'std': 24.5},
        '4057': {'name': 'Shake Shack', 'avg_amount': 211.49, 'std': 114.2},
        '5428': {'name': 'Air France-KLM', 'avg_amount': 426.7, 'std': 202.73},
        '1402': {'name': 'Emirates', 'avg_amount': 440.31, 'std': 126.25},
        '9605': {'name': 'Amazon', 'avg_amount': 400.15, 'std': 213.64},
        '9236': {'name': 'Saks Fifth Avenue', 'avg_amount': 252.25, 'std': 139.14},
        '7349': {'name': 'Taco Bell', 'avg_amount': 182.82, 'std': 108.84},
        '5157': {'name': 'Google', 'avg_amount': 328.75, 'std': 167.16},
        '1867': {'name': 'AMC Theatres', 'avg_amount': 362.23, 'std': 98.65},
        '9210': {'name': 'Verizon', 'avg_amount': 360.49, 'std': 202.44},
        '1364': {'name': 'Kroger', 'avg_amount': 293.23, 'std': 167.3},
        '8467': {'name': 'The Home Depot', 'avg_amount': 412.5, 'std': 96.15},
        '7571': {'name': 'Disney', 'avg_amount': 326.53, 'std': 99.35},
        '1579': {'name': 'Regal Cinemas', 'avg_amount': 329.79, 'std': 139.46},
        '8674': {'name': 'Nike', 'avg_amount': 412.25, 'std': 156.83},
        '9737': {'name': 'Williams-Sonoma', 'avg_amount': 426.2, 'std': 118.65},
        '7468': {'name': 'United Airlines', 'avg_amount': 133.58, 'std': 74.06},
        '5150': {'name': 'Lululemon', 'avg_amount': 40.12, 'std': 18.19},
        '4039': {'name': 'T-Mobile', 'avg_amount': 403.87, 'std': 95.38},
        '5137': {'name': 'Uniqlo', 'avg_amount': 132.64, 'std': 49.0},
        '1452': {'name': 'American Airlines', 'avg_amount': 77.59, 'std': 26.53},
        '7537': {'name': 'Ford', 'avg_amount': 21.33, 'std': 6.68},
        '2381': {'name': 'Hyatt', 'avg_amount': 98.5, 'std': 41.29},
        '9154': {'name': 'InterContinental Hotels','avg_amount': 425.73,'std': 222.56},
        '2467': {'name': 'eBay', 'avg_amount': 386.62, 'std': 175.62},
        '3738': {'name': 'Expedia', 'avg_amount': 483.41, 'std': 207.03},
        '8572': {'name': 'Shopify', 'avg_amount': 432.27, 'std': 217.99},
        '7747': {'name': 'Tesla', 'avg_amount': 467.42, 'std': 141.78},
        '1880': {'name': 'Microsoft', 'avg_amount': 306.44, 'std': 129.03},
        '4149': {'name': 'ExxonMobil', 'avg_amount': 478.02, 'std': 258.26},
        '9276': {'name': 'Singapore Airlines', 'avg_amount': 119.25, 'std': 32.64},
        '8892': {'name': 'Burger King', 'avg_amount': 439.94, 'std': 94.16},
        '5144': {'name': 'TJX Companies', 'avg_amount': 48.18, 'std': 19.5},
        '2168': {'name': 'Universal Parks', 'avg_amount': 175.37, 'std': 47.76},
        '4129': {'name': 'Aldi', 'avg_amount': 201.47, 'std': 61.49},
        '4645': {'name': 'BP', 'avg_amount': 470.52, 'std': 163.11},
        '3526': {'name': 'CVS Health', 'avg_amount': 340.53, 'std': 173.55},
        '4516': {'name': 'Marriott International','avg_amount': 448.41,'std': 201.17},
        '4367': {'name': 'Lufthansa', 'avg_amount': 320.03, 'std': 110.71},
        '8338': {'name': 'Zara', 'avg_amount': 137.04, 'std': 74.62},
        '4555': {'name': 'Ross Stores', 'avg_amount': 419.75, 'std': 160.56},
        '6111': {'name': 'Gap Inc', 'avg_amount': 43.85, 'std': 15.16},
        '7752': {'name': 'Toyota', 'avg_amount': 372.5, 'std': 76.07},
        '1445': {'name': 'Netflix', 'avg_amount': 322.14, 'std': 116.83},
        '7364': {'name': 'Papa Johns', 'avg_amount': 328.43, 'std': 183.65},
        '4626': {'name': 'Booking.com', 'avg_amount': 248.11, 'std': 82.99},
        '5016': {'name': 'Bed Bath & Beyond', 'avg_amount': 290.84, 'std': 113.28},
        '4875': {'name': 'Dominos Pizza', 'avg_amount': 402.96, 'std': 224.27},
        '6497': {'name': 'Apple', 'avg_amount': 186.35, 'std': 37.85},
        '7238': {'name': 'Wayfair', 'avg_amount': 376.97, 'std': 89.56},
        '1261': {'name': 'H&M', 'avg_amount': 346.77, 'std': 196.22},
        '5992': {'name': 'Lowes', 'avg_amount': 242.01, 'std': 100.24},
        '5025': {'name': 'Five Guys', 'avg_amount': 445.54, 'std': 208.79},
        '9283': {'name': 'Nordstrom', 'avg_amount': 431.39, 'std': 137.59},
        '5222': {'name': 'General Motors', 'avg_amount': 386.1, 'std': 104.33},
        '5175': {'name': 'Chipotle', 'avg_amount': 89.47, 'std': 40.15},
        '4443': {'name': 'Starbucks', 'avg_amount': 181.54, 'std': 106.84},
        '6011': {'name': 'Shell', 'avg_amount': 228.46, 'std': 125.56},
        '1693': {'name': 'Target', 'avg_amount': 270.36, 'std': 139.8},
        '9347': {'name': 'Pizza Hut', 'avg_amount': 206.01, 'std': 51.58},
        '2812': {'name': 'Bloomingdales', 'avg_amount': 77.43, 'std': 39.8},
        '6724': {'name': 'AT&T', 'avg_amount': 68.24, 'std': 21.49},
        '2407': {'name': 'Chick-fil-A', 'avg_amount': 132.11, 'std': 42.85},
        '9032': {'name': 'Uber', 'avg_amount': 416.22, 'std': 132.23},
        '2410': {'name': 'Comcast', 'avg_amount': 396.8, 'std': 215.5},
        '4497': {'name': 'Chevron', 'avg_amount': 371.3, 'std': 145.18},
        '5372': {'name': 'Safeway', 'avg_amount': 333.91, 'std': 153.07},
        '8394': {'name': 'Macys', 'avg_amount': 116.64, 'std': 53.61},
        '1551': {'name': 'Neiman Marcus', 'avg_amount': 81.45, 'std': 25.4},
        '5332': {'name': 'McDonalds', 'avg_amount': 308.71, 'std': 122.6},
        '1997': {'name': 'Delta Air Lines', 'avg_amount': 304.78, 'std': 81.12},
        '5076': {'name': 'Subway', 'avg_amount': 37.94, 'std': 13.08},
        '7099': {'name': 'Hilton', 'avg_amount': 242.55, 'std': 82.35},
        '6555': {'name': 'Charter Communications','avg_amount': 310.23,'std': 148.67},
        '5052': {'name': 'British Airways', 'avg_amount': 157.58, 'std': 41.63},
        '6303': {'name': 'Wendys', 'avg_amount': 182.61, 'std': 70.55},
        '9118': {'name': 'Best Buy', 'avg_amount': 466.42, 'std': 195.29},
        '7102': {'name': 'Etsy', 'avg_amount': 461.04, 'std': 103.49},
        '6490': {'name': 'PayPal', 'avg_amount': 177.03, 'std': 79.05},
        '9564': {'name': 'Kohls', 'avg_amount': 206.61, 'std': 88.91},
        '6616': {'name': 'KFC', 'avg_amount': 381.74, 'std': 216.13},
        '3248': {'name': 'Spotify', 'avg_amount': 412.69, 'std': 119.68},
        '1848': {'name': 'Whole Foods', 'avg_amount': 255.64, 'std': 103.31},
        '7354': {'name': 'IKEA', 'avg_amount': 209.28, 'std': 74.57},
        '9695': {'name': 'Tesco', 'avg_amount': 296.73, 'std': 159.71},
        '9070': {'name': 'Trader Joes', 'avg_amount': 222.61, 'std': 95.69},
        '9894': {'name': 'Adidas', 'avg_amount': 397.15, 'std': 126.8},
        '4625': {'name': 'Costco', 'avg_amount': 234.52, 'std': 107.74},
        '1314': {'name': 'Southwest Airlines', 'avg_amount': 407.46, 'std': 167.53},
        '3376': {'name': 'Publix', 'avg_amount': 397.98, 'std': 205.9},
        '5378': {'name': 'Ticketmaster', 'avg_amount': 405.52, 'std': 125.91}
        }
    
    # Assign archetypes to customers
    customer_archetype_map = {}
    current_customer = 1
    for archetype, config in customer_archetypes.items():
        n_customers_archetype = int(n_customers * config['proportion'])
        for _ in range(n_customers_archetype):
            if current_customer <= n_customers:
                customer_archetype_map[f"CUST_{current_customer:06d}"] = archetype
                current_customer += 1
    
    # Fill remaining customers with random archetypes
    while current_customer <= n_customers:
        archetype = np.random.choice(list(customer_archetypes.keys()))
        customer_archetype_map[f"CUST_{current_customer:06d}"] = archetype
        current_customer += 1
    
    transactions = []
    start_date = datetime.now() - timedelta(days=90)
    
    # Calculate approximate transactions per customer based on archetypes
    transactions_per_customer = {}
    for customer_id, archetype in customer_archetype_map.items():
        config = customer_archetypes[archetype]
        avg_freq = np.mean(config['freq_per_day_range'])
        n_txns = int(avg_freq * 90 * np.random.uniform(0.8, 1.2))
        transactions_per_customer[customer_id] = max(1, n_txns)
    
    # Generate transactions based on customer behavior
    for customer_id, archetype in customer_archetype_map.items():
        config = customer_archetypes[archetype]
        n_customer_txns = transactions_per_customer[customer_id]
        
        # Customer-level attributes
        card_number = f"****{np.random.randint(1000, 9999)}"
        issuer_id = f"ISSUER_{np.random.randint(1, n_issuers+1):03d}"
        issuer_bin = f"{np.random.randint(400000, 499999)}"
        device_id = f"DEV_{hash(customer_id) % 10000:06d}"
        cardholder_email = f"{customer_id.lower()}@email.com"
        
        # Determine transaction amount range for this customer
        amt_min, amt_max = config['avg_amount_range']
        customer_avg_amount = np.random.uniform(amt_min, amt_max)
        
        # Determine merchant pool for this customer (based on diversity)
        div_min, div_max = config['merchant_diversity']
        customer_diversity = np.random.uniform(div_min, div_max)
        n_unique_merchants = max(1, int(n_customer_txns * customer_diversity))
        customer_merchants = [f"MERCH_{np.random.randint(1, n_merchants+1):05d}" 
                            for _ in range(n_unique_merchants)]
        
        for _ in range(n_customer_txns):
            # Merchant selection
            merchant_id = np.random.choice(customer_merchants)
            acquirer_id = f"ACQ_{np.random.randint(1, n_acquirers+1):03d}"
            mcc = np.random.choice(list(merchant_categories.keys()))
            merchant_name = merchant_categories[mcc]['name']
            terminal_id = f"TERM_{np.random.randint(10000, 99999)}"
            acquirer_bin = f"{np.random.randint(500000, 599999)}"
            
            # Transaction amount (customer-specific with variance)
            transaction_amount = max(5, np.random.normal(customer_avg_amount, customer_avg_amount * 0.3))
            
            # International transaction
            is_international = np.random.random() < config['international_prob']
            if is_international:
                merchant_country = np.random.choice(['UK', 'CA', 'DE', 'FR', 'IN', 'AU'])
                transaction_currency = np.random.choice(['GBP', 'EUR', 'CAD', 'INR'])
            else:
                merchant_country = 'US'
                transaction_currency = 'USD'
            
            # Timestamp
            days_ago = np.random.randint(0, 90)
            is_weekend = np.random.random() < config['weekend_prob']
            if is_weekend:
                hour = int(np.random.beta(3, 2) * 24)
            else:
                hour = int(np.random.beta(5, 2) * 24)
            transaction_timestamp = start_date + timedelta(days=days_ago, hours=hour, 
                                                           minutes=np.random.randint(0, 60))
            
            # Entry mode (online vs offline)
            is_online = np.random.random() < config['online_prob']
            if is_online:
                entry_mode = 'e-commerce'
            else:
                entry_mode = np.random.choice(['chip', 'contactless', 'swipe'], p=[0.5, 0.35, 0.15])
            
            transaction_type = np.random.choice(['purchase', 'refund', 'cash_advance'],
                                               p=[0.93, 0.05, 0.02])
            
            # Risk scoring (archetype-specific)
            fraud_min, fraud_max = config['fraud_score_range']
            fraud_score = np.random.uniform(fraud_min, fraud_max)
            
            # Account attributes
            account_balance = np.random.uniform(100, 50000)
            available_credit = np.random.uniform(1000, 20000)
            spending_limit_daily = np.random.choice([500, 1000, 2500, 5000, 10000])
            account_age_days = np.random.randint(30, 3650)
            
            velocity_1h = np.random.poisson(0.5)
            velocity_24h = np.random.poisson(3)
            cardholder_risk_score = np.random.beta(2, 8) * 100
            
            # Authorization decision
            decline_probability = 0.01 + (0.15 if fraud_score > 70 else 0) + \
                                (0.05 if transaction_amount > spending_limit_daily else 0)
            authorization_decision = 'approved' if np.random.random() > decline_probability else 'declined'
            
            decline_reason_codes = ['05', '51', '61', '65', '54', '57', '62', '63']
            response_code = '00' if authorization_decision == 'approved' else np.random.choice(decline_reason_codes)
            authorization_code = f"AUTH{np.random.randint(100000, 999999)}" if authorization_decision == 'approved' else None
            decline_reason = None
            if response_code != '00':
                decline_reasons = {
                    '05': 'Do not honor', '51': 'Insufficient funds',
                    '61': 'Exceeds withdrawal limit', '65': 'Exceeds withdrawal frequency',
                    '54': 'Expired card', '57': 'Transaction not permitted',
                    '62': 'Restricted card', '63': 'Security violation'
                }
                decline_reason = decline_reasons.get(response_code, 'Other')
            
            # Network variables
            network_transaction_id = f"NET{len(transactions):012d}"
            network = 'MASTERCARD'
            interchange_rate = np.random.uniform(1.5, 3.0)
            network_fee = transaction_amount * (interchange_rate / 100)
            cross_border = 1 if is_international else 0
            
            ip_address = f"{np.random.randint(1,255)}.{np.random.randint(1,255)}.{np.random.randint(1,255)}.{np.random.randint(1,255)}"
            
            # 3DS authentication
            auth_3ds_result = None
            if entry_mode == 'e-commerce':
                auth_3ds_result = np.random.choice(['success', 'failed', 'not_attempted'], p=[0.85, 0.05, 0.1])
            
            chargeback = 1 if np.random.random() < 0.005 else 0
            
            transaction = {
                'customer_id': customer_id,
                'card_number': card_number,
                'cardholder_email': cardholder_email,
                'device_id': device_id,
                'ip_address': ip_address,
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
        'customer_id': lambda x: x.nunique(),
        'card_number': lambda x: x.nunique(),
        'transaction_amount': ['sum', 'mean', 'count'],
        'available_credit': 'mean',
        'spending_limit_daily': 'mean',
        'account_age_days': 'mean',
        'fraud_score': 'mean',
        'cardholder_risk_score': 'mean',
        'authorization_decision': lambda x: (x == 'declined').sum() / len(x),
        'response_code': lambda x: (x == '51').sum() / len(x),
        'chargeback': 'sum',
        'cross_border_indicator': 'mean',
        'entry_mode': lambda x: (x == 'e-commerce').sum() / len(x),
        'transaction_type': lambda x: (x == 'cash_advance').sum() / len(x),
        'auth_3ds_result': lambda x: (x == 'success').sum() / x.notna().sum() if x.notna().sum() > 0 else 0,
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
    
    issuer_agg['transactions_per_customer'] = issuer_agg['transaction_count'] / issuer_agg['active_customers']
    issuer_agg['volume_per_customer'] = issuer_agg['total_transaction_volume'] / issuer_agg['active_customers']
    issuer_agg['chargeback_rate'] = issuer_agg['chargeback_count'] / issuer_agg['transaction_count']
    issuer_agg['approval_rate'] = 1 - issuer_agg['decline_rate']
    issuer_agg['cards_per_customer'] = issuer_agg['active_cards'] / issuer_agg['active_customers']
    
    return issuer_agg

def create_acquirer_aggregates(df):
    """Create aggregated acquirer-level features for clustering"""
    
    acquirer_agg = df.groupby('acquirer_id').agg({
        'merchant_id': lambda x: x.nunique(),
        'merchant_category_code': lambda x: x.nunique(),
        'transaction_amount': ['sum', 'mean', 'count'],
        'network_fee': 'sum',
        'customer_id': lambda x: x.nunique(),
        'merchant_country': lambda x: x.nunique(),
        'fraud_score': 'mean',
        'authorization_decision': lambda x: (x == 'declined').sum() / len(x),
        'chargeback': 'sum',
        'cross_border_indicator': 'mean',
        'entry_mode': lambda x: (x == 'e-commerce').sum() / len(x),
        'transaction_type': lambda x: (x == 'refund').sum() / len(x),
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
    
    id_cols = ['customer_id', 'merchant_id', 'issuer_id', 'acquirer_id', 
               'mcc', 'merchant_name', 'primary_network']
    feature_cols = [col for col in df.columns if col not in id_cols]
    
    X = df[feature_cols].fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    df['cluster'] = clusters
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['pca_1'] = X_pca[:, 0]
    df['pca_2'] = X_pca[:, 1]
    
    return df, kmeans, scaler, pca, feature_cols

def name_and_tag_clusters(df, entity_type='customer'):
    """Improved cluster naming with multi-dimensional analysis"""
    
    cluster_profiles = []
    
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id]
        
        profile = {
            'cluster_id': cluster_id,
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(df) * 100
        }
        
        if entity_type == 'customer':
            # Calculate key metrics
            avg_amount = cluster_data['avg_transaction_amount'].mean()
            frequency = cluster_data['transaction_frequency'].mean()
            online_ratio = cluster_data['online_transaction_ratio'].mean()
            international_ratio = cluster_data['international_ratio'].mean()
            decline_rate = cluster_data['decline_rate'].mean()
            fraud_score = cluster_data['avg_fraud_score'].mean()
            merchant_diversity = cluster_data['merchant_diversity'].mean()
            
            # Calculate percentiles for comparison
            amount_pct = (df['avg_transaction_amount'] < avg_amount).mean()
            freq_pct = (df['transaction_frequency'] < frequency).mean()
            online_pct = (df['online_transaction_ratio'] < online_ratio).mean()
            intl_pct = (df['international_ratio'] < international_ratio).mean()
            risk_pct = (df['avg_fraud_score'] < fraud_score).mean()
            
            profile.update({
                'avg_transaction_amount': avg_amount,
                'transaction_frequency': frequency,
                'online_ratio': online_ratio,
                'international_ratio': international_ratio,
                'decline_rate': decline_rate,
                'avg_fraud_score': fraud_score,
                'unique_merchants': cluster_data['unique_merchants'].mean(),
                'merchant_diversity': merchant_diversity
            })
            
            # Improved naming logic with priority hierarchy
            tags = []
            
            # 1. Risk-based classification (highest priority)
            if risk_pct > 0.85 and decline_rate > 0.05:
                name = "High-Risk Customers"
                tags = ['high-risk', 'fraud-prone', 'monitoring-required', 'elevated-decline']
            
            # 2. Value + Frequency combinations
            elif amount_pct > 0.75 and freq_pct > 0.75:
                name = "Premium Frequent Shoppers"
                tags = ['vip', 'high-value', 'frequent', 'premium', 'top-tier']
            
            elif amount_pct > 0.80 and freq_pct < 0.40:
                name = "Big Ticket Buyers"
                tags = ['high-value', 'occasional', 'large-purchases', 'premium-items']
            
            # 3. Channel-based classification
            elif online_pct > 0.80:
                name = "Digital-First Customers"
                tags = ['e-commerce', 'online', 'digital-native', 'tech-savvy', 'app-users']
            
            # 4. Geographic behavior
            elif intl_pct > 0.75:
                name = "International Travelers"
                tags = ['global', 'traveler', 'cross-border', 'international', 'multi-currency']
            
            # 5. Frequency-based
            elif freq_pct > 0.75 and amount_pct < 0.60:
                name = "Everyday Shoppers"
                tags = ['frequent', 'daily-use', 'groceries', 'routine', 'loyal']
            
            # 6. Low engagement
            elif freq_pct < 0.30 and amount_pct < 0.50:
                name = "Occasional Shoppers"
                tags = ['infrequent', 'low-engagement', 'dormant-risk', 'casual']
            
            # 7. Moderate/balanced
            else:
                name = "Balanced Customers"
                tags = ['moderate', 'balanced', 'stable', 'regular']
            
            # Add behavioral modifiers
            if merchant_diversity > 0.7:
                tags.append('diverse-spending')
            if online_ratio > 0.5 and online_pct < 0.80:
                tags.append('omnichannel')
            if international_ratio > 0.2 and intl_pct < 0.75:
                tags.append('occasional-traveler')
                
        elif entity_type == 'merchant':
            profile.update({
                'avg_ticket_size': cluster_data['avg_ticket_size'].mean(),
                'transaction_volume': cluster_data['transaction_count'].mean(),
                'unique_customers': cluster_data['unique_customers'].mean(),
                'decline_rate': cluster_data['decline_rate'].mean(),
                'refund_rate': cluster_data['refund_rate'].mean(),
            })
            
            if profile['avg_ticket_size'] > df['avg_ticket_size'].quantile(0.6):
                name = "Premium Merchants"
                tags = ['high-value', 'premium', 'luxury']
            elif profile['transaction_volume'] > df['transaction_count'].quantile(0.78):
                name = "High-Volume Merchants"
                tags = ['busy', 'popular', 'high-traffic']
            elif profile['decline_rate'] > df['decline_rate'].quantile(0.55):
                name = "High-Risk Merchants"
                tags = ['risky', 'high-decline', 'problematic']
            elif profile['refund_rate'] > df['refund_rate'].quantile(0.5):
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