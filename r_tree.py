import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from rtree import index  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# Step 1: Preprocess the Dataset
def preprocess_data(file_path):
    coffee_data = pd.read_csv(file_path)

    # Convert review_date to datetime and extract year
    coffee_data['review_date'] = pd.to_datetime(coffee_data['review_date'], format='%B %Y', errors='coerce')
    coffee_data['original_review_year'] = coffee_data['review_date'].dt.year
    coffee_data['review_year'] = coffee_data['original_review_year']

    # Drop rows with missing dates
    coffee_data = coffee_data.dropna(subset=['review_date'])

    # Preserve original values before normalization
    coffee_data['original_100g_USD'] = coffee_data['100g_USD']
    coffee_data['original_rating'] = coffee_data['rating']

    # Normalize numerical attributes for R-Tree
    scaler = MinMaxScaler()
    coffee_data[['100g_USD', 'rating']] = scaler.fit_transform(
        coffee_data[['original_100g_USD', 'original_rating']]
    )

    return coffee_data, scaler


# Step 2: Build R-Tree
def build_rtree(filtered_data):
    rtree_idx = index.Index()
    for idx, row in filtered_data.iterrows():
       
        x = row['100g_USD']
        y = row['rating']
        rtree_idx.insert(idx, (x, y, x, y))  
    return rtree_idx


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


# Step 4: Query R-Tree
def query_rtree(rtree_idx, filtered_data, query_point, query_radius):
    x, y = query_point
    # Define a bounding box for the query radius
    query_bbox = (x - query_radius, y - query_radius, x + query_radius, y + query_radius)
    indices = list(rtree_idx.intersection(query_bbox))

    if not indices:
        return pd.DataFrame()  

    return filtered_data.iloc[indices]


# Interactive Query System for R-Tree + LSH
def interactive_query_system_rtree(file_path):
    coffee_data, scaler = preprocess_data(file_path)

    print("\nWelcome to the Coffee Reviews Query System with R-Tree + LSH!")

    # User Input for R-Tree Filtering
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
        print("No results match your R-Tree filter criteria. Please try again.")
        return

    # Build R-Tree
    print("\nBuilding R-Tree...")
    rtree_idx = build_rtree(filtered_data)

    # Define query point (normalized)
    query_point = scaler.transform(pd.DataFrame(
        [[min_price + (max_price - min_price) / 2, min_rating]],
        columns=['original_100g_USD', 'original_rating']
    ))[0]

    print("\nDebug: R-Tree query point (normalized):", query_point)

    # Query R-Tree
    query_radius = 0.1  # Define a suitable query radius
    print("\nQuerying R-Tree...")
    rtree_results = query_rtree(rtree_idx, filtered_data, query_point, query_radius)

    if rtree_results.empty:
        print("No R-Tree results found.")
        return

    print("\nR-Tree Results:")
    print(rtree_results[['name', 'original_100g_USD', 'original_rating', 'original_review_year', 'loc_country', 'review']].to_string(index=False))

    # Prepare LSH using R-Tree results
    print("\nPreparing LSH on R-Tree filtered data...")
    vectorizer, nbrs = prepare_lsh(rtree_results)

    # User Input for LSH Query
    print("\nEnter the LSH query parameters:")
    query_text = input("Enter a phrase to find similar reviews: ")
    lsh_results = query_lsh(vectorizer, nbrs, query_text, rtree_results, n_results)

    if lsh_results.empty:
        print("No LSH results found within R-Tree filtered results.")
        return

    print("\nLSH Results:")
    print(lsh_results[['name', 'original_100g_USD', 'original_rating', 'original_review_year', 'loc_country', 'review']].to_string(index=False))



# Main 
if __name__ == "__main__":
    file_path = "simplified_coffee.csv"
    interactive_query_system_rtree(file_path)
