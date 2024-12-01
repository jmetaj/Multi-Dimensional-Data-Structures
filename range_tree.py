import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# Range Tree Node
class RangeTreeNode:
    def __init__(self, points, dim):
        self.points = points  
        self.dim = dim  
        self.left = None  
        self.right = None 


# Step 1: Preprocess the Dataset
def preprocess_data(file_path):
    coffee_data = pd.read_csv(file_path)

    # Convert review_date to datetime and extract year
    coffee_data['review_date'] = pd.to_datetime(coffee_data['review_date'], format='%B %Y', errors='coerce')
    coffee_data['original_review_year'] = coffee_data['review_date'].dt.year
    coffee_data['review_year'] = coffee_data['original_review_year']

    # Drop rows with missing dates
    coffee_data = coffee_data.dropna(subset=['review_date'])

    
    coffee_data['original_100g_USD'] = coffee_data['100g_USD']
    coffee_data['original_rating'] = coffee_data['rating']

    # Normalize numerical attributes for Range Tree
    scaler = MinMaxScaler()
    coffee_data[['100g_USD', 'rating']] = scaler.fit_transform(
        coffee_data[['original_100g_USD', 'original_rating']]
    )

    return coffee_data, scaler


# Step 2: Build Range Tree
def build_range_tree(points, dim=0, max_dim=2):
    if not points:
        return None

    # Sort points by current dimension
    points = sorted(points, key=lambda x: x[dim])

    # Find median
    median_idx = len(points) // 2
    median_point = points[median_idx]

    # Create the current node
    node = RangeTreeNode(points, dim)

    # Build left and right subtrees
    next_dim = (dim + 1) % max_dim
    node.left = build_range_tree(points[:median_idx], next_dim, max_dim)
    node.right = build_range_tree(points[median_idx + 1:], next_dim, max_dim)

    return node


# Step 3: Query Range Tree
def range_tree_query(node, query_range, dim=0, max_dim=2):
    if not node:
        return []

    points_in_range = []
    current_dim = dim

    for point in node.points:
        if query_range[current_dim][0] <= point[current_dim] <= query_range[current_dim][1]:
            points_in_range.append(point)

    next_dim = (current_dim + 1) % max_dim
    if node.left:
        points_in_range += range_tree_query(node.left, query_range, next_dim, max_dim)
    if node.right:
        points_in_range += range_tree_query(node.right, query_range, next_dim, max_dim)

    return points_in_range


# Step 4: Prepare LSH for Textual Attributes
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


# Interactive Query System for Range Tree + LSH
def interactive_query_system_range(file_path):
    coffee_data, scaler = preprocess_data(file_path)

    print("\nWelcome to the Coffee Reviews Query System with Range Tree + LSH!")

    # User Input for Range Tree Filtering
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
        print("No results match your Range Tree filter criteria. Please try again.")
        return

    # Prepare points for Range Tree
    points = filtered_data[['100g_USD', 'rating']].to_numpy()
    points = [(point[0], point[1], idx) for idx, point in enumerate(points)]

    # Build Range Tree
    print("\nBuilding Range Tree...")
    range_tree = build_range_tree(points)

    # Define query range (normalized)
    query_range = [
        (scaler.transform([[min_price, 0]])[0][0], scaler.transform([[max_price, 0]])[0][0]),  # Price range
        (scaler.transform([[0, min_rating]])[0][1], scaler.transform([[0, max_price]])[0][1])  # Rating range
    ]

    # Query Range Tree
    print("\nQuerying Range Tree...")
    range_tree_results = range_tree_query(range_tree, query_range)

    if not range_tree_results:
        print("No Range Tree results found.")
        return

    # Extract relevant indices and corresponding data
    result_indices = [point[2] for point in range_tree_results]
    range_tree_data = filtered_data.iloc[result_indices]

    print("\nRange Tree Results:")
    print(range_tree_data[['name', 'original_100g_USD', 'original_rating', 'original_review_year', 'loc_country', 'review']].to_string(index=False))

    # Prepare LSH using Range Tree results
    print("\nPreparing LSH on Range Tree filtered data...")
    vectorizer, nbrs = prepare_lsh(range_tree_data)

    # User Input for LSH Query
    print("\nEnter the LSH query parameters:")
    query_text = input("Enter a phrase to find similar reviews: ")
    lsh_results = query_lsh(vectorizer, nbrs, query_text, range_tree_data, n_results)

    if lsh_results.empty:
        print("No LSH results found within Range Tree filtered results.")
        return

    print("\nLSH Results:")
    print(lsh_results[['name', 'original_100g_USD', 'original_rating', 'original_review_year', 'loc_country', 'review']].to_string(index=False))


# Main 
if __name__ == "__main__":
    file_path = "simplified_coffee.csv"
    interactive_query_system_range(file_path)
