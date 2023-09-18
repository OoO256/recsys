import pandas as pd
import os 

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)

def load_data():
    movie=pd.read_csv(f"{ROOT_DIR}/datasets/tmdb/5000/tmdb_5000_movies.csv")
    credits=pd.read_csv(f"{ROOT_DIR}/datasets/tmdb/5000/tmdb_5000_credits.csv")

    movie=movie[['id','title','overview','genres','keywords', 'vote_average', 'vote_count']]
    credits=credits[['movie_id','cast','crew']]
    movie=movie.merge(credits, left_on='id', right_on='movie_id', how='left')
    
    return movie
