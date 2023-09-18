from demographic_filtering import demographic_filtering
from content_based_filtering import tf_idf_filtering, w2v_tf_idf_filtering

demographic_filtering()[['title', 'weighted_rating', 'vote_average', 'vote_count']].head(10)
tf_idf_filtering('The Dark Knight Rises')
w2v_tf_idf_filtering('The Dark Knight Rises')