import os
import json
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from query_cleanup import search_related_keywords

# Load environment variables
load_dotenv()

def handle_query_by_vector_space_model(query):
    dataset_path = os.getenv('DATASET_PATH')
    keyword_length = int(os.getenv('KEYWORD_LENGTH'))

    try:
        with open(dataset_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
    except FileNotFoundError:
        return f"Error: '{dataset_path}' file not found."
    except json.JSONDecodeError as e:
        return f"Error: Could not parse JSON in '{dataset_path}'. {e}"

    # Extract keywords from the query
    extracted_keywords = [item[0] for item in search_related_keywords(query, keyword_length)]

    # Search for documents based on vector space model
    matched_documents = search_by_vector_space_model(query, json_data, extracted_keywords)

    return matched_documents

def search_by_vector_space_model(query, json_data, extracted_keywords):
    """
    Performs search using vector space model based on TF-IDF matrix, allowing for n-grams.

    Args:
        query (str): The user's query input.
        json_data (list): List of dictionaries containing dataset content.
        tfidf_matrix (sparse matrix): TF-IDF matrix of dataset content.
        feature_names (list or ndarray): List of feature names (terms) in TF-IDF matrix.
        extracted_keywords (list): List of extracted keywords from the query.

    Returns:
        list: List of tuples containing matched title and content.
    """


    vector_db_path = os.getenv('VECTOR_DB_PATH')
    try:
        with open(vector_db_path, 'r', encoding='utf-8') as file:
            vector_db_data = json.load(file)
    except FileNotFoundError:
        return f"Error: '{vector_db_path}' file not found."
    except json.JSONDecodeError as e:
        return f"Error: Could not parse JSON in '{vector_db_path}'. {e}"

    feature_names, tfidf_matrix = vector_db_data["feature_names"], vector_db_data["tfidf_matrix"]


    # Initialize query vector
    query_vector = np.zeros((1, len(feature_names)))

    # Convert feature_names to list if it's a NumPy ndarray
    if isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()

    # Create query vector based on extracted keywords
    for term in extracted_keywords:
        if term in feature_names:
            query_vector[0, feature_names.index(term)] += 1


    # Compute cosine similarities between query vector and documents
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    document_scores = [
        (
            obj.get('product_id'),
            obj.get('product_name'),
            obj.get('category'),
            obj.get('price_of_each_unit'),
            obj.get('description'),
            obj.get('available_units'),
            score
        )
        for obj, score in zip(json_data, cosine_similarities)
    ]

    document_scores.sort(key=lambda x: x[6], reverse=True)

    # Filter documents with nonzero similarity scores
    matched_documents = [
        (product_id, product_name, category, price, description, units, score)
        for product_id, product_name, category, price, description, units, score in document_scores
        if score > 0
    ]

    return matched_documents
