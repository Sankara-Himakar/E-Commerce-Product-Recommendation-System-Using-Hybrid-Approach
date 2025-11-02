
import pandas as pd

def prodList(limit=1000, dataset_dir="dataset"):
    """
    Returns a list of products (as lists) with columns:
    [product_id, product_name, aisle_id, department_id, aisle, department]
    """
    products = pd.read_csv(f"{dataset_dir}/products.csv")
    aisles = pd.read_csv(f"{dataset_dir}/aisles.csv")
    departments = pd.read_csv(f"{dataset_dir}/departments.csv")

    products = products.merge(aisles, on='aisle_id', how='left') \
                       .merge(departments, on='department_id', how='left')
    products = products.sort_values('product_id')
    products = products.head(limit)
    # to_numpy().tolist() so it's easy to create a DataFrame from
    return products.to_numpy().tolist()
