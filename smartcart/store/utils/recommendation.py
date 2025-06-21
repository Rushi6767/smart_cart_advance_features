import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import re
import os
import pickle
from django.conf import settings

# Global variables to store the data and models
df = None
vectorizer = None
svd = None
reduced_matrix = None

def load_data():
    """Load pre-trained recommendation model and data"""
    global df, vectorizer, svd, reduced_matrix
    
    if df is not None:
        return df
    
    try:
        # Get the Django project root directory
        project_root = settings.BASE_DIR
        
        # Load the pre-trained model
        model_path = os.path.join(project_root, 'store', 'utils', 'recommendation_model.pkl')
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Extract the components from the saved model
        df = model_data.get('df')
        vectorizer = model_data.get('vectorizer')
        svd = model_data.get('svd')
        reduced_matrix = model_data.get('reduced_matrix')
        
        print(f"Pre-trained model loaded successfully. Data shape: {df.shape}")
        return df
        
    except Exception as e:
        print(f"Error loading pre-trained model: {e}")
        return None

def preprocess_text(text):
    if pd.isna(text):
        return ''
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    text = ' '.join(text.split())
    return text

def extract_gender_and_type(query):
    df = load_data()
    if df is None:
        return None, None
    
    query = query.lower()
    gender = None
    for g in ['men', 'women', 'boys', 'girls', 'unisex']:
        if g in query:
            gender = g.capitalize()
            break
    article_type = None
    for at in df['articleType_proc'].unique():
        if at in query:
            article_type = at
            break
    return gender, article_type

def recommend_products(product_name, top_n=5):
    """Get recommendations based on product name using pre-trained model - exactly like notebook"""
    df = load_data()
    if df is None:
        print("Could not load pre-trained model")
        return []
    
    # Preprocess query
    query_proc = preprocess_text(product_name)
    print(f"Looking for product: {product_name} (processed: {query_proc})")
    
    # Extract gender and articleType from query
    gender, article_type = extract_gender_and_type(query_proc)
    print(f"Extracted gender: {gender}, article_type: {article_type}")
    
    # Strictly filter by gender and articleType
    filtered = df.copy()
    if gender:
        filtered = filtered[filtered['gender'].str.lower() == gender.lower()]
    if article_type:
        filtered = filtered[filtered['articleType_proc'] == article_type]
    
    print(f"Filtered dataset size: {len(filtered)}")
    
    if len(filtered) == 0:
        print("❌ No products found for your query.")
        return []
    
    # Fuzzy match for product name in filtered set
    match = filtered[filtered['productDisplayName_proc'].str.contains(query_proc)]
    if not match.empty:
        product = match.iloc[0]
        print(f"Found exact match: {product['productDisplayName']}")
    else:
        # Fallback: just use the first in filtered
        product = filtered.iloc[0]
        print(f"Using fallback product: {product['productDisplayName']}")
    
    # Recommendation logic - exactly like notebook
    idx = filtered.index.get_loc(product.name)
    filtered_idx = filtered.index.tolist()
    product_vec = reduced_matrix[product.name].reshape(1, -1)
    filtered_vecs = reduced_matrix[filtered_idx]
    sims = cosine_similarity(product_vec, filtered_vecs).flatten()
    top_idx = sims.argsort()[-top_n-1:-1][::-1]
    results = filtered.iloc[top_idx]
    
    # Prepare output for template
    recommendations = []
    for _, row in results.iterrows():
        recommendations.append({
            'product_name': row['productDisplayName'],
            'category': row['articleType'],
            'color': row['baseColour'],
            'gender': row['gender'],
            'image_url': row['image_url'],
            'price': row.get('price', 'N/A'),
        })
    
    print(f"Found {len(recommendations)} recommendations")
    return recommendations 