# -*- coding: utf-8 -*-
"""
Retail Product Recommendation System
Created: May 15, 2026

This module implements collaborative filtering algorithms for product recommendations:
1. User-based collaborative filtering (similarity-based)
2. Matrix factorization using SVD

@author: madhurigopal1
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import warnings

warnings.filterwarnings('ignore')


class RecommendationEngine:
    """
    A recommendation engine using collaborative filtering techniques.
    
    Attributes:
        df (pd.DataFrame): The input data containing customer-product-rating information
        user_item_matrix (pd.DataFrame): User-item matrix for recommendations
    """
    
    def __init__(self, file_path=None, df=None):
        """
        Initialize the recommendation engine with data.
        
        Args:
            file_path (str): Path to Excel file with customer ratings (optional)
            df (pd.DataFrame): Dataframe with customer ratings (optional)
            
        Raises:
            ValueError: If neither file_path nor df is provided
        """
        if file_path:
            self.df = self._load_data(file_path)
        elif df is not None:
            self.df = df
        else:
            raise ValueError("Either file_path or df must be provided")
        
        # Validate data structure
        required_columns = ['customer_id', 'product_id', 'rating']
        missing_cols = [col for col in required_columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        self.user_item_matrix = self._create_user_item_matrix()
    
    @staticmethod
    def _load_data(file_path):
        """
        Load data from Excel file.
        
        Args:
            file_path (str): Path to Excel file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            df = pd.read_excel(file_path)
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error loading file: {str(e)}")
    
    def _create_user_item_matrix(self):
        """
        Create user-item matrix from data.
        
        Returns:
            pd.DataFrame: User-item matrix with ratings
        """
        user_item_matrix = self.df.pivot_table(
            index='customer_id',
            columns='product_id',
            values='rating',
            fill_value=0
        )
        return user_item_matrix
    
    def get_user_similarity_recommendations(self, user_id, top_n=5, num_similar_users=5):
        """
        Get recommendations using user-based collaborative filtering.
        
        This method:
        1. Computes cosine similarity between the target user and all other users
        2. Finds the most similar users
        3. Aggregates their ratings (weighted by similarity)
        4. Recommends products not yet rated by the target user
        
        Args:
            user_id (str): Target user ID
            top_n (int): Number of recommendations to return (default: 5)
            num_similar_users (int): Number of similar users to consider (default: 5)
            
        Returns:
            pd.Series: Product recommendations with predicted scores
            
        Raises:
            ValueError: If user_id not found in data
        """
        if user_id not in self.user_item_matrix.index:
            raise ValueError(f"User {user_id} not found in data")
        
        # Compute similarity matrix
        similarity = cosine_similarity(self.user_item_matrix)
        similarity_df = pd.DataFrame(
            similarity,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        
        # Get similar users (excluding the user themselves)
        similar_users = similarity_df[user_id].sort_values(ascending=False)[1:num_similar_users+1]
        
        # Aggregate recommendations from similar users
        candidate_items = pd.Series(dtype=float)
        
        for sim_user in similar_users.index:
            candidate_items = candidate_items.add(
                self.user_item_matrix.loc[sim_user] * similar_users[sim_user],
                fill_value=0
            )
        
        # Filter out already purchased items
        already_purchased = self.user_item_matrix.loc[user_id]
        recommendations = candidate_items[already_purchased == 0]
        
        return recommendations.sort_values(ascending=False).head(top_n)
    
    def get_svd_recommendations(self, user_id, top_n=5, n_factors=3):
        """
        Get recommendations using matrix factorization (SVD).
        
        This method:
        1. Applies SVD to decompose the user-item matrix
        2. Predicts ratings for all items
        3. Recommends highest-rated unseen items
        
        Args:
            user_id (str): Target user ID
            top_n (int): Number of recommendations to return (default: 5)
            n_factors (int): Number of latent factors for SVD (default: 3)
            
        Returns:
            pd.Series: Product recommendations with predicted ratings
            
        Raises:
            ValueError: If user_id not found in data
        """
        if user_id not in self.user_item_matrix.index:
            raise ValueError(f"User {user_id} not found in data")
        
        matrix = self.user_item_matrix.values
        user_ratings_mean = np.mean(matrix, axis=1)
        
        # Mean-center the matrix
        matrix_demeaned = matrix - user_ratings_mean.reshape(-1, 1)
        
        # Determine optimal k for SVD
        k = min(n_factors, min(matrix_demeaned.shape) - 1)
        
        # Apply SVD
        U, sigma, Vt = svds(matrix_demeaned, k=k)
        sigma = np.diag(sigma)
        
        # Reconstruct predicted ratings
        predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
        
        # Create predictions dataframe
        preds_df = pd.DataFrame(
            predicted_ratings,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.columns
        )
        
        # Get predictions for target user
        user_predictions = preds_df.loc[user_id]
        already_purchased = self.user_item_matrix.loc[user_id]
        
        # Filter out already purchased items
        recommendations = user_predictions[already_purchased == 0]
        
        return recommendations.sort_values(ascending=False).head(top_n)
    
    def get_ensemble_recommendations(self, user_id, top_n=5, weight_user_based=0.5):
        """
        Get recommendations using an ensemble of both methods.
        
        Combines user-based and SVD recommendations with weighted averaging.
        
        Args:
            user_id (str): Target user ID
            top_n (int): Number of recommendations to return (default: 5)
            weight_user_based (float): Weight for user-based method (0-1)
            
        Returns:
            pd.Series: Ensemble recommendations with combined scores
        """
        # Get both recommendations
        user_based = self.get_user_similarity_recommendations(user_id, top_n=top_n*2)
        svd_based = self.get_svd_recommendations(user_id, top_n=top_n*2)
        
        # Normalize scores to 0-1 range
        if len(user_based) > 0:
            user_based_norm = (user_based - user_based.min()) / (user_based.max() - user_based.min() + 1e-8)
        else:
            user_based_norm = pd.Series()
        
        if len(svd_based) > 0:
            svd_based_norm = (svd_based - svd_based.min()) / (svd_based.max() - svd_based.min() + 1e-8)
        else:
            svd_based_norm = pd.Series()
        
        # Combine recommendations
        ensemble = pd.Series(dtype=float)
        all_items = set(user_based_norm.index) | set(svd_based_norm.index)
        
        for item in all_items:
            score = 0
            if item in user_based_norm.index:
                score += weight_user_based * user_based_norm[item]
            if item in svd_based_norm.index:
                score += (1 - weight_user_based) * svd_based_norm[item]
            ensemble[item] = score
        
        return ensemble.sort_values(ascending=False).head(top_n)
    
    def get_user_stats(self, user_id):
        """
        Get statistics for a specific user.
        
        Args:
            user_id (str): Target user ID
            
        Returns:
            dict: User statistics including number of items rated, average rating, etc.
        """
        if user_id not in self.user_item_matrix.index:
            raise ValueError(f"User {user_id} not found in data")
        
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0]
        
        return {
            'user_id': user_id,
            'items_rated': len(rated_items),
            'avg_rating': float(rated_items.mean()) if len(rated_items) > 0 else 0,
            'min_rating': float(rated_items.min()) if len(rated_items) > 0 else 0,
            'max_rating': float(rated_items.max()) if len(rated_items) > 0 else 0,
            'items_not_rated': len(user_ratings[user_ratings == 0])
        }
    
    def get_all_users(self):
        """Get list of all users in the dataset."""
        return list(self.user_item_matrix.index)
    
    def get_all_products(self):
        """Get list of all products in the dataset."""
        return list(self.user_item_matrix.columns)


# =====================================
# Main Execution
# =====================================
if __name__ == "__main__":
    # Initialize engine with sample data
    engine = RecommendationEngine(file_path="retail_data.xlsx")
    
    # Get list of users
    users = engine.get_all_users()
    example_user = users[0] if users else "C001"
    
    print("=" * 60)
    print("RETAIL PRODUCT RECOMMENDATION SYSTEM")
    print("=" * 60)
    
    # User statistics
    print(f"\nUser Statistics for {example_user}:")
    stats = engine.get_user_stats(example_user)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # User-based recommendations
    print(f"\n--- User-Based Collaborative Filtering Recommendations ---")
    user_recs = engine.get_user_similarity_recommendations(example_user, top_n=5)
    print(f"\nTop 5 Recommendations for {example_user}:")
    for product, score in user_recs.items():
        print(f"  {product}: {score:.4f}")
    
    # SVD recommendations
    print(f"\n--- Matrix Factorization (SVD) Recommendations ---")
    svd_recs = engine.get_svd_recommendations(example_user, top_n=5)
    print(f"\nTop 5 Recommendations for {example_user}:")
    for product, score in svd_recs.items():
        print(f"  {product}: {score:.4f}")
    
    # Ensemble recommendations
    print(f"\n--- Ensemble Recommendations ---")
    ensemble_recs = engine.get_ensemble_recommendations(example_user, top_n=5, weight_user_based=0.5)
    print(f"\nTop 5 Ensemble Recommendations for {example_user}:")
    for product, score in ensemble_recs.items():
        print(f"  {product}: {score:.4f}")
    
    print("\n" + "=" * 60)
