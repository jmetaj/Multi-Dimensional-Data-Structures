import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import KDTree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Step 1: Preprocess the Dataset
def preprocess_data(file_path):
    coffee_data = pd.read_csv(file_path)

    # Convert review_date to datetime and extract year
    coffee_data['review_date'] = pd.to_datetime(coffee_data['review_date'], format='%B %Y', errors='coerce')
    coffee_data['original_review_year'] = coffee_data['review_date'].dt.year 
    coffee_data['review_year'] = coffee_data['original_review_year'] 

   
    coffee_data = coffee_data.dropna(subset=['review_date'])

    # Preserve original values before normalization
    coffee_data['original_100g_USD'] = coffee_data['100g_USD']
    coffee_data['original_rating'] = coffee_data['rating']

    # Normalize numerical attributes for KD-Tree
    scaler = MinMaxScaler()
    coffee_data[['100g_USD', 'rating', 'review_year']] = scaler.fit_transform(
        coffee_data[['original_100g_USD', 'original_rating', 'review_year']]
    )
    return coffee_data, scaler

# Step 2: Build KD-Tree for Numerical Attributes
def build_kd_tree(filtered_data, selected_attributes):
    kd_tree_data = filtered_data[selected_attributes].to_numpy()
    kd_tree = KDTree(kd_tree_data)
    return kd_tree

# Step 3: Prepare LSH for Textual Attributes
def prepare_lsh(filtered_data, text_column='review'):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(filtered_data[text_column])
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine').fit(tfidf_matrix)
    return vectorizer, nbrs

# Query LSH
def query_lsh(vectorizer, nbrs, query_text, filtered_data, n_results):
    query_vector = vectorizer.transform([query_text])
    distances, indices = nbrs.kneighbors(query_vector, n_neighbors=min(n_results, len(filtered_data)))
    return filtered_data.iloc[indices[0]]

# Query KD-Tree
def query_kd_tree(kd_tree, filtered_data, query_point, n_results, selected_attributes):
    kd_tree_data = filtered_data[selected_attributes].to_numpy()

    # Try radius-based search
    radius = 0.2
    indices = kd_tree.query_ball_point(query_point, r=radius)
    valid_indices = [i for i in indices if i < len(filtered_data)]

    # Fallback to k-nearest neighbors if no results are found
    if not valid_indices:
        print("No results found within the radius. Using k-nearest neighbors.")
        distances, indices = kd_tree.query([query_point], k=n_results)
        valid_indices = [i for i in indices[0] if i < len(filtered_data)]

    if not valid_indices:
        return pd.DataFrame()

    return filtered_data.iloc[valid_indices]


# Main Interactive Query System
def interactive_query_system(file_path):
    coffee_data, scaler = preprocess_data(file_path)

    # Allow user to select attributes for KD-Tree indexing
    selected_attributes = ['100g_USD', 'rating', 'review_year'] 

    # Build KD-Tree
    kd_tree = build_kd_tree(coffee_data, selected_attributes)

    print("\nWelcome to the Coffee Reviews Query System with KD-Tree + LSH!")

    # User Input for KD-Tree Filtering
    start_year = int(input("Enter the start year: "))
    end_year = int(input("Enter the end year: "))
    min_rating = float(input("Enter the minimum review rating: "))
    min_price = float(input("Enter the minimum price per 100g: "))
    max_price = float(input("Enter the maximum price per 100g: "))
    country = input("Enter the country of origin: ")
    n_results = int(input("Enter the number of top results to return: "))

    # Filter the dataset
    filtered_data = coffee_data[
        (coffee_data['original_review_year'] >= start_year) &
        (coffee_data['original_review_year'] <= end_year) &
        (coffee_data['original_rating'] >= min_rating) &
        (coffee_data['original_100g_USD'] >= min_price) &
        (coffee_data['original_100g_USD'] <= max_price) &
        (coffee_data['loc_country'] == country)
    ].reset_index(drop=True)

    print("\nDebug: Filtered data:")
    print(filtered_data)

    if filtered_data.empty:
        print("No results match your KD-Tree filter criteria. Please try again.")
        return

    # Normalize the query point for KD-Tree
    query_point = scaler.transform(pd.DataFrame(
        [[min_price + (max_price - min_price) / 2, min_rating, (start_year + end_year) / 2]],
        columns=['original_100g_USD', 'original_rating', 'review_year']
    ))[0]

    print("\nDebug: KD-Tree query point (normalized):", query_point)

     # Build KD-Tree using filtered data
    kd_tree = build_kd_tree(filtered_data, selected_attributes)


    # Query KD-Tree with timing
    print("\nQuerying KD-Tree...")
    import time
    start_time = time.time()
    kd_results = query_kd_tree(kd_tree, filtered_data, query_point, n_results, selected_attributes)
    kd_time = time.time() - start_time
    print(f"KD-Tree Query Time: {kd_time:.4f} seconds")

    if kd_results.empty:
        print("No KD-Tree results found.")
        return

    print("\nKD-Tree Results:")
    print(kd_results[['name', 'original_100g_USD', 'original_rating', 'original_review_year', 'loc_country', 'review']].to_string(index=False))

    # Prepare LSH using KD-Tree results
    print("\nPreparing LSH on KD-Tree filtered data...")
    vectorizer, nbrs = prepare_lsh(kd_results)

    # User Input for LSH Query
    print("\nEnter the LSH query parameters:")
    query_text = input("Enter a phrase to find similar reviews: ")
    lsh_results = query_lsh(vectorizer, nbrs, query_text, kd_results, n_results)

    if lsh_results.empty:
        print("No LSH results found within KD-Tree filtered results.")
        return

    print("\nLSH Results:")
    print(lsh_results[['name', 'original_100g_USD', 'original_rating', 'original_review_year', 'loc_country', 'review']].to_string(index=False))


# Main Execution
if __name__ == "__main__":
    file_path = "simplified_coffee.csv" 
    interactive_query_system(file_path)
