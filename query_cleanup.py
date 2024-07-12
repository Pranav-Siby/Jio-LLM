from nltk.corpus import stopwords
import json
import string

english_stopwords = stopwords.words('english')  # A list of stopwords

def search_related_keywords(query, keyword_length):
    """
    Searches for related keywords in the user query and returns them along with their scores.
    Stopwords and punctuation are removed from the query before searching.

    Parameters:
        query (str): User query as input.
        keyword_length (int): Maximum length of keyword phrases to search for.

    Returns:
        list: List of related keywords found in the query.
    """

    try:
        try:
            with open('keywords.json', 'r', encoding='utf-8') as file:
                keywords_dict = json.load(file)
        except FileNotFoundError:
            print("Error: 'keywords.json' file not found.")
            return "Error: 'keywords.json' file not found."

        # Get English stopwords
        stop_words = set(stopwords.words('english'))

        # Create a translation table to remove punctuation
        translator = str.maketrans('', '', string.punctuation)

        # Split query into words, remove stopwords and punctuation
        query_words = [word.translate(translator) for word in query.lower().split() if word not in stop_words]

        related_keywords = []
        for i in range(len(query_words)):
            for j in range(i + 1, min(i + keyword_length, len(query_words)) + 1):
                phrase = ' '.join(query_words[i:j])
                if phrase in keywords_dict:
                    related_keywords.append([phrase, keywords_dict[phrase]])

        # Sort related keywords by their scores
        related_keywords = sorted(related_keywords, key=lambda x: x[1])

        return related_keywords

    except Exception as e:
        print(f"Error during keyword extraction: {e}")
        return ["Error during keyword extraction"]
