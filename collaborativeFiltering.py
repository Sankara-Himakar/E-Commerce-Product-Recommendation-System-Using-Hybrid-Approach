# collaborativeFiltering.py
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from products import prodList
from orders_products_df import ordersProducts

def cf(query_index, n_neighbors=10, dataset_dir="dataset"):
    """
    query_index: integer index into the pivot table rows (product index to base recommendations on)
    Returns a DataFrame of recommended products (product_id, product_name, aisle_id, department_id, aisle, department)
    """
    # get products metadata
    products = prodList(dataset_dir=dataset_dir)
    products = pd.DataFrame(products, columns=['product_id', 'product_name', 'aisle_id', 'department_id', 'aisle', 'department'])

    orders_products_df = ordersProducts(dataset_dir=dataset_dir)
    # NearestNeighbors expects samples as rows; here rows are product_name
    orders_products_df_matrix = csr_matrix(orders_products_df.values)

    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(orders_products_df_matrix)

    # make sure query_index is valid
    if query_index < 0 or query_index >= orders_products_df.shape[0]:
        raise IndexError("query_index out of range for orders_products_df")

    distances, indices = model_knn.kneighbors(
        orders_products_df.iloc[query_index, :].values.reshape(1, -1),
        n_neighbors=min(n_neighbors, orders_products_df.shape[0])
    )

    # retrieve product names for recommended indices
    reco_names = list(orders_products_df.index[indices.flatten()])
    # create DataFrame and merge with metadata
    reco_df = pd.DataFrame(reco_names, columns=['product_name'])
    reco_df = reco_df.merge(products, on='product_name', how='left')

    # ensure columns order
    swap_list = ["product_id", "product_name", "aisle_id", "department_id", "aisle", "department"]
    # for products not found in metadata, merge will leave NaNs
    return reco_df.reindex(columns=[c for c in swap_list if c in reco_df.columns])
