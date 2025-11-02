# contentBasedFiltering.py
import pandas as pd
from products import prodList
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

def cbf(product_name, dataset_dir="dataset", n_recs=12):
    """
    product_name: exact product_name string to base recommendations on.
    Returns DataFrame of recommended products with basic metadata.
    """
    products = prodList(dataset_dir=dataset_dir)
    products = pd.DataFrame(products, columns=['product_id', 'product_name', 'aisle_id', 'department_id', 'aisle', 'department'])
    productsCopy = products.copy()

    # Build a simple 'description' from aisle + department + product_name (if needed)
    products['description'] = (products['aisle'].fillna('') + ' ' + products['department'].fillna('') + ' ' + products['product_name'].fillna('')).str.strip()
    products['description'] = products['description'].fillna('')

    tfv = TfidfVectorizer(
        min_df=1, max_features=None,
        strip_accents='unicode', analyzer='word',
        token_pattern=r'\w{1,}', ngram_range=(1, 3),
        stop_words='english'
    )

    tfv_matrix = tfv.fit_transform(products['description'])
    sig = sigmoid_kernel(tfv_matrix, tfv_matrix)

    # mapping product_name -> index
    indices = pd.Series(products.index, index=products['product_name']).drop_duplicates()

    if product_name not in indices:
        raise KeyError(f"Product name '{product_name}' not found in products list.")

    idx = indices[product_name]

    sig_scores = list(enumerate(sig[idx]))
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
    # skip first (the product itself) and take next n_recs
    sig_scores = sig_scores[1:n_recs+1]
    product_indices = [i[0] for i in sig_scores]

    # return metadata rows for those indices
    recs = productsCopy.iloc[product_indices][['product_id', 'product_name', 'aisle', 'department']].reset_index(drop=True)
    return recs
