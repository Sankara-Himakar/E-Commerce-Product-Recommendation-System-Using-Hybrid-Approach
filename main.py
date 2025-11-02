# main.py
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

# --- example: SVD-based item similarity from Amazon ratings (small subset) ---
amazon_ratings = pd.read_csv('dataset/ratings_Beauty.csv')
amazon_ratings = amazon_ratings.dropna(subset=['UserId', 'ProductId', 'Rating'])

# use a subset for speed (first 10k ratings)
amazon_ratings1 = amazon_ratings.head(10000)

# pivot utility matrix user x product
ratings_utility_matrix = amazon_ratings1.pivot_table(
    values='Rating', index='UserId', columns='ProductId', fill_value=0
)

# transpose to item x user (rows=items)
X = ratings_utility_matrix.T  # items x users

# SVD decomposition (items -> latent factors)
n_components = 10
svd = TruncatedSVD(n_components=n_components, random_state=42)
decomposed_matrix = svd.fit_transform(X)  # shape: items x n_components

# correlation matrix between items in latent space
correlation_matrix = np.corrcoef(decomposed_matrix)

# example: find recommendations for a given product id string
query_product_id = "6117036094"  # example from your original code
product_ids = list(X.index)

if query_product_id not in product_ids:
    print(f"{query_product_id} not found in this subset.")
else:
    product_ID = product_ids.index(query_product_id)
    correlation_product_ID = correlation_matrix[product_ID]
    min_confidence = 0.90
    recommend_mask = correlation_product_ID > min_confidence
    # remove self
    recommend_mask[product_ID] = False
    recommendations = [pid for pid, keep in zip(product_ids, recommend_mask) if keep]
    print("Top recommendations:", recommendations[:10])
