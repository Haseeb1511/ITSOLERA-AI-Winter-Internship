import os
import re
import ast
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)


# path to data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # root dir
MOVIES_CSV = os.path.join(BASE_DIR, "data", "tmdb_5000_movies.csv")  # csv frile from root dir
CREDITS_CSV = os.path.join(BASE_DIR, "data", "tmdb_5000_credits.csv")



def extract_names(obj, key="name", top_n=None, job_filter=None):
    if pd.isna(obj):
        return []
    try:
        data = ast.literal_eval(obj)
    except Exception:
        return []

    results = []
    for item in data:
        if job_filter:
            if item.get("job") == job_filter:
                results.append(item.get(key))
                break
        else:
            results.append(item.get(key))
        if top_n and len(results) >= top_n:
            break
    return results


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocessing(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = text.lower()
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)



def build_model():
    """Load data, process, and compute similarity matrix. Returns (final_df, similarity)."""
    print("Loading datasets...")
    movies = pd.read_csv(MOVIES_CSV)
    credit = pd.read_csv(CREDITS_CSV)

    # Merge
    merged_df = movies.merge(credit, left_on="id", right_on="movie_id")
    merged_df = merged_df.drop(columns=["movie_id"])

    # Drop unnecessary columns
    columns_to_drop = [
        "homepage", "title_x", "title_y", "status",
        "production_countries", "spoken_languages",
        "revenue", "release_date", "runtime",
        "vote_average", "vote_count", "budget", "tagline",
    ]
    merged_df = merged_df.drop(columns=columns_to_drop)

    # Drop missing
    merged_df.dropna(inplace=True)

    # Extract features
    merged_df["genres"] = merged_df["genres"].apply(extract_names)
    merged_df["keywords"] = merged_df["keywords"].apply(extract_names)
    merged_df["cast"] = merged_df["cast"].apply(lambda x: extract_names(x, top_n=3))
    merged_df["crew"] = merged_df["crew"].apply(lambda x: extract_names(x, job_filter="Director"))

    # Process overview & remove spaces in names
    merged_df["overview"] = merged_df["overview"].fillna("").apply(lambda x: x.split())
    for col in ["genres", "keywords", "cast", "crew"]:
        merged_df[col] = merged_df[col].apply(lambda x: [i.replace(" ", "") for i in x])

    # Combine tags
    merged_df["tags"] = (
        merged_df["overview"]
        + merged_df["genres"]
        + merged_df["keywords"]
        + merged_df["cast"]
        + merged_df["crew"]
    )

    # Build final dataframe
    final_df = merged_df[["id", "original_title", "tags"]].copy()
    final_df["tags"] = final_df["tags"].apply(lambda x: " ".join(x))
    final_df["tags"] = final_df["tags"].apply(preprocessing)
    final_df = final_df.reset_index(drop=True)

    # TF-IDF + Cosine Similarity
    print("Building TF-IDF vectors...")
    tfidf = TfidfVectorizer(max_features=6000)
    vectors = tfidf.fit_transform(final_df["tags"]).toarray()

    print("Computing cosine similarity...")
    similarity = cosine_similarity(vectors)

    print(f"Model ready! {len(final_df)} movies loaded.")
    return final_df, similarity


# Initialize model on import
final_df, similarity = build_model()


def get_all_movies():
    """Return list of all movie titles."""
    return final_df["original_title"].tolist()




def recommend(movie: str, top_k: int = 5):
    """
    Given a movie title, return top_k similar movies.
    Returns list of dicts with 'id' and 'title'.
    """
    movie_lower = movie.lower()
    titles_lower = final_df["original_title"].str.lower()

    if movie_lower not in titles_lower.values:
        return None

    index = final_df[titles_lower == movie_lower].index[0]
    distances = similarity[index]
    movie_indices = distances.argsort()[::-1][1 : top_k + 1]

    results = []
    for idx in movie_indices:
        results.append({
            "id": int(final_df.iloc[idx]["id"]),
            "title": final_df.iloc[idx]["original_title"],
        })
    return results
