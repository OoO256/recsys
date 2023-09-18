from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

from load_data import load_data
from embedding import w2v_average_embedding, sentence_embedding

def get_recommendations(data, idx, similarity_matrix, k=10):
    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:k+1]

    # Get the movie indices
    indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return data.iloc[indices]

tf_idf_vectorizer = TfidfVectorizer(stop_words='english')

def tf_idf_filtering(query, k=10):
    movies = load_data()
    movies['overview'] = movies['overview'].fillna('')

    tfidf_vectors = tf_idf_vectorizer.fit_transform(movies['overview'])
    cosine_sim = cosine_similarity(tfidf_vectors, tfidf_vectors)

    title_to_index = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    idx_query = title_to_index[query]
    recommendations = get_recommendations(movies, idx_query, cosine_sim, k)
    return recommendations[['title', 'overview', 'vote_average']]

def w2v_tf_idf_filtering(query, k=10):
    movies = load_data()
    movies = movies[movies['overview'].notna()]

    word_embeddings = w2v_average_embedding(movies['overview'])
    cosine_sim = cosine_similarity(word_embeddings, word_embeddings)

    title_to_index = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    idx_query = title_to_index[query]
    recommendations = get_recommendations(movies, idx_query, cosine_sim, k)
    return recommendations[['title', 'overview', 'vote_average']]


def sentence_filtering(query, k=10):
    movies = load_data()
    movies = movies[movies['overview'].notna()]

    sentence_embeddings = sentence_embedding(movies['overview'])
    cosine_sim = cosine_similarity(sentence_embeddings, sentence_embeddings)

    title_to_index = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    idx_query = title_to_index[query]
    recommendations = get_recommendations(movies, idx_query, cosine_sim, k)
    return recommendations[['title', 'overview', 'vote_average']]

if __name__ == "__main__":
    # print(tf_idf_filtering('The Dark Knight Rises'))
    print(w2v_tf_idf_filtering('The Dark Knight Rises'))
    # print(sentence_filtering('The Dark Knight Rises'))
    # print(tf_idf_filtering('Inception'))
    # print(w2v_tf_idf_filtering('Inception'))
    # print(sentence_filtering('Inception'))