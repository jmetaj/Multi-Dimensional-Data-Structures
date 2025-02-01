from shared_utils import preprocess_data, filter_data, prepare_lsh, query_lsh
from scipy.spatial import KDTree
import pandas as pd

# Build KD-Tree
def build_kd_tree(filtered_data, selected_attributes):
    kd_tree_data = filtered_data[selected_attributes].to_numpy()
    kd_tree = KDTree(kd_tree_data)
    return kd_tree

# Query KD-Tree
def query_kd_tree(kd_tree, filtered_data, query_point, n_results, selected_attributes):
    kd_tree_data = filtered_data[selected_attributes].to_numpy()
    radius = 0.2
    indices = kd_tree.query_ball_point(query_point, r=radius)
    valid_indices = [i for i in indices if i < len(filtered_data)]

    if not valid_indices:
        print("No results found within the radius. Using k-nearest neighbors.")
        distances, indices = kd_tree.query([query_point], k=n_results)
        valid_indices = [i for i in indices[0] if i < len(filtered_data)]

    if not valid_indices:
        return pd.DataFrame()

    return filtered_data.iloc[valid_indices]

# Main Interactive Query System
def interactive_query_system_kd_tree(
    file_path, start_year, end_year, min_rating, min_price, max_price, country, n_results, query_text
):
    coffee_data, scaler = preprocess_data(file_path)

    selected_attributes = ['100g_USD', 'rating', 'review_year']

    # Filter the dataset
    filtered_data = filter_data(coffee_data, start_year, end_year, min_rating, min_price, max_price, country)
    if filtered_data.empty:
        print("No results match your KD-Tree filter criteria.")
        return pd.DataFrame()

    print("\nWelcome to the Coffee Reviews Query System with KD-Tree + LSH!")

    # Normalize the query point for KD-Tree
    query_point = scaler.transform(pd.DataFrame(
        [[min_price + (max_price - min_price) / 2, min_rating, (start_year + end_year) / 2]],
        columns=['original_100g_USD', 'original_rating', 'review_year']
    ))[0]

    print("\nDebug: KD-Tree query point (normalized):", query_point)

    # Build KD-Tree using filtered data
    kd_tree = build_kd_tree(filtered_data, selected_attributes)

    # Query KD-Tree
    print("\nQuerying KD-Tree...")
    kd_results = query_kd_tree(kd_tree, filtered_data, query_point, n_results, selected_attributes)
    if kd_results.empty:
        print("No KD-Tree results found.")
        return pd.DataFrame()
    
    # Include the original review year in the output
    kd_results['review_year'] = kd_results['original_review_year'] 

    print("\nKD-Tree Results:")
    print(kd_results[['name', 'original_100g_USD', 'original_rating',  'review_year', 'loc_country', 'review']].to_string(index=False))

    # Prepare LSH
    print("\nPreparing LSH on KD-Tree filtered data...")
    vectorizer, nbrs = prepare_lsh(kd_results)

    # Query LSH
    print("\nQuerying LSH...")
    lsh_results = query_lsh(vectorizer, nbrs, query_text, kd_results, n_results)

    if lsh_results.empty:
        print("No LSH results found within KD-Tree filtered results.")
        return pd.DataFrame()
    
    # Include the original review year in the LSH results output
    lsh_results['review_year'] = lsh_results['original_review_year']

    print("\nLSH Results:")
    print(lsh_results[['name', 'original_100g_USD', 'original_rating','review_year', 'loc_country', 'review']].to_string(index=False))

    return lsh_results
