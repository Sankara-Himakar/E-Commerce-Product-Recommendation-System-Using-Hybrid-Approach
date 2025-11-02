# orders_products_df.py
import pandas as pd

def ordersProducts(dataset_dir="dataset", rating_threshold_min=0, rating_threshold_max=999999):
    """
    Builds a pivot table with index=product_name, columns=order_id, values=rating (add_to_cart_order).
    Filters products between rating_threshold_min and rating_threshold_max (by count of occurrences).
    Returns: DataFrame (product_name x order_id) filled with 0s for missing ratings.
    """
    order_products_train = pd.read_csv(f"{dataset_dir}/order_products__train.csv")
    products = pd.read_csv(f"{dataset_dir}/products.csv")
    aisles = pd.read_csv(f"{dataset_dir}/aisles.csv")
    departments = pd.read_csv(f"{dataset_dir}/departments.csv")

    products = products.merge(aisles, on='aisle_id', how='left') \
                       .merge(departments, on='department_id', how='left')
    # merge to get product_name into order_products
    orders_products = order_products_train.merge(products, on='product_id', how='left')

    # Use add_to_cart_order as a proxy "rating" (as original code)
    orders_products = orders_products.rename(columns={'add_to_cart_order': 'rating'})

    # drop rows with missing product_name
    orders_products = orders_products.dropna(subset=['product_name'])

    # total count per product
    ratingCount = orders_products.groupby('product_name')['rating'] \
                                 .count() \
                                 .reset_index() \
                                 .rename(columns={'rating': 'totalRatingCount'}) \
                                 [['product_name', 'totalRatingCount']]

    # merge counts back
    orders_products_with_totalRatingCount = orders_products.merge(
        ratingCount, on='product_name', how='left'
    )

    # filter by thresholds inclusive
    rating_popular_products = orders_products_with_totalRatingCount.query(
        'totalRatingCount >= @rating_threshold_min and totalRatingCount <= @rating_threshold_max'
    )

    # build pivot: index product_name, columns order_id, values rating; missing -> 0
    orders_products_df = rating_popular_products.pivot_table(
        index='product_name',
        columns='order_id',
        values='rating',
        aggfunc='first'  # should be single value per (product_name, order_id)
    ).fillna(0)

    return orders_products_df
