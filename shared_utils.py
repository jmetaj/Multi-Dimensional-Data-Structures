# shared_utils.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Preprocess the dataset
def preprocess_data(file_path):
    coffee_data = pd.read_csv(file_path)

    coffee_data['review_date'] = pd.to_datetime(coffee_data['review_date'], format='%B %Y', errors='coerce')
    coffee_data['original_review_year'] = coffee_data['review_date'].dt.year
    coffee_data['review_year'] = coffee_data['original_review_year']
    coffee_data = coffee_data.dropna(subset=['review_date'])

    coffee_data['original_100g_USD'] = coffee_data['100g_USD']
    coffee_data['original_rating'] = coffee_data['rating']

    scaler = MinMaxScaler()
    coffee_data[['100g_USD', 'rating', 'review_year']] = scaler.fit_transform(
        coffee_data[['original_100g_USD', 'original_rating', 'review_year']]
    )

    return coffee_data, scaler

# Filter the dataset based on user criteria
def filter_data(coffee_data, start_year, end_year, min_rating, min_price, max_price, country):
    return coffee_data[
        (coffee_data['original_review_year'] >= start_year) &
        (coffee_data['original_review_year'] <= end_year) &
        (coffee_data['original_rating'] >= min_rating) &
        (coffee_data['original_100g_USD'] >= min_price) &
        (coffee_data['original_100g_USD'] <= max_price) &
        (coffee_data['loc_country'] == country)
    ].reset_index(drop=True)

# Prepare LSH for textual attributes
def prepare_lsh(filtered_data, text_column='review'):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(filtered_data[text_column])
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine').fit(tfidf_matrix)
    return vectorizer, nbrs

# Query LSH for similar reviews
def query_lsh(vectorizer, nbrs, query_text, filtered_data, n_results):
    query_vector = vectorizer.transform([query_text])
    distances, indices = nbrs.kneighbors(query_vector, n_neighbors=min(n_results, len(filtered_data)))
    return filtered_data.iloc[indices[0]]
