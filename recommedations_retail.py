# -*- coding: utf-8 -*-
"""
Created on Fri May 15 12:27:30 2026

@author: 424646
"""

# ===========================================
# Retail Recommendation System (Collaborative Filtering)
# ===========================================

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import numpy as np

# -------------------------------
# STEP 1: Load Data
# -------------------------------
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# -------------------------------
# STEP 2: Create User-Item Matrix
# -------------------------------
def create_user_item_matrix(df):
    user_item_matrix = df.pivot_table(
        index='customer_id',
        columns='product_id',
        values='rating',
        fill_value=0
    )
    return user_item_matrix

# -------------------------------
# STEP 3: Collaborative Filtering (User-based)
# -------------------------------
def user_similarity_recommendations(user_id, user_item_matrix, top_n=5):
    similarity = cosine_similarity(user_item_matrix)
    similarity_df = pd.DataFrame(similarity, 
                                 index=user_item_matrix.index,
                                 columns=user_item_matrix.index)

    similar_users = similarity_df[user_id].sort_values(ascending=False)[1:6]

    candidate_items = pd.Series(dtype=float)

    for sim_user in similar_users.index:
        candidate_items = candidate_items.add(
            user_item_matrix.loc[sim_user] * similar_users[sim_user],
            fill_value=0
        )

    already_purchased = user_item_matrix.loc[user_id]
    recommendations = candidate_items[already_purchased == 0]

    return recommendations.sort_values(ascending=False).head(top_n)

# -------------------------------
# STEP 4: Matrix Factorization (SVD)
# -------------------------------
#def svd_recommendations(user_id, user_item_matrix, top_n=5):
    #matrix = user_item_matrix.values
    #user_ratings_mean = np.mean(matrix, axis=1)
    
    #matrix_demeaned = matrix - user_ratings_mean.reshape(-1, 1)

    #U, sigma, Vt = svds(matrix_demeaned, k=50)
    #k = min(3, min(matrix_demeaned.shape) - 1)
    #U, sigma, Vt = svds(matrix_demeaned, k=k)

    #sigma = np.diag(sigma)

    #predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    #preds_df = pd.DataFrame(predicted_ratings, 
    #                        index=user_item_matrix.index,
    #                        columns=user_item_matrix.columns)

    #user_predictions = preds_df.loc[user_id]
    #already_purchased = user_item_matrix.loc[user_id]

    #recommendations = user_predictions[already_purchased == 0]
    #return recommendations.sort_values(ascending=False).head(top_n)
def svd_recommendations(user_id, user_item_matrix, top_n=5):
    import numpy as np
    from scipy.sparse.linalg import svds

    matrix = user_item_matrix.values
    user_ratings_mean = np.mean(matrix, axis=1)

    matrix_demeaned = matrix - user_ratings_mean.reshape(-1, 1)

    # ✅ FIXED k logic
    k = min(3, min(matrix_demeaned.shape) - 1)
    
    U, sigma, Vt = svds(matrix_demeaned, k=k)
    sigma = np.diag(sigma)

    predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

    preds_df = pd.DataFrame(predicted_ratings,
                            index=user_item_matrix.index,
                            columns=user_item_matrix.columns)

    user_predictions = preds_df.loc[user_id]
    already_purchased = user_item_matrix.loc[user_id]

    recommendations = user_predictions[already_purchased == 0]

    return recommendations.sort_values(ascending=False).head(top_n)

# -------------------------------
# STEP 5: Execute
# -------------------------------
if __name__ == "__main__":
    df = pd.read_excel("retail_data.xlsx")    
    #df = load_data(file_path)
    user_item_matrix = create_user_item_matrix(df)

    user_id = "C001"  # Example user
    
    print("\n--- User-Based Recommendations ---")
    print(user_similarity_recommendations(user_id, user_item_matrix))

    print("\n--- SVD Recommendations ---")
    print(svd_recommendations(user_id, user_item_matrix))