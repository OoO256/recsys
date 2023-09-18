from load_data import load_data

def demographic_filtering():
    '''
    using IMDB's weighted rating (wr)
    '''

    movies=load_data()

    rate_avg = movies['vote_average'].mean()
    min_rate_count= movies['vote_count'].quantile(0.9)
    movies_recommended = movies.copy().loc[movies['vote_count'] >= min_rate_count]

    def weighted_rating(x):
        vote_count = x['vote_count']
        vote_average = x['vote_average']
        return (vote_count/(vote_count+min_rate_count) * vote_average) + (min_rate_count/(min_rate_count+vote_count) * rate_avg)
    
    movies_recommended['weighted_rating'] = movies_recommended.apply(weighted_rating, axis=1)
    movies_recommended = movies_recommended.sort_values('weighted_rating', ascending=False)
    return movies_recommended


