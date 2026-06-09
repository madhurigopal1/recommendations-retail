# -*- coding: utf-8 -*-
"""
Basic usage examples for the Recommendation Engine
"""

import sys
sys.path.insert(0, '..')

from recommendations_retail import RecommendationEngine


def example_1_simple_recommendations():
    """Example 1: Get simple recommendations for a user"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Simple Recommendations")
    print("="*60)
    
    # Initialize engine
    engine = RecommendationEngine(file_path='../retail_data.xlsx')
    
    # Get a user
    users = engine.get_all_users()
    user_id = users[0]
    
    # Get user-based recommendations
    recs = engine.get_user_similarity_recommendations(user_id, top_n=5)
    print(f"\nUser-based recommendations for {user_id}:")
    print(recs)


def example_2_multiple_methods():
    """Example 2: Compare different recommendation methods"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Comparing Recommendation Methods")
    print("="*60)
    
    engine = RecommendationEngine(file_path='../retail_data.xlsx')
    users = engine.get_all_users()
    user_id = users[0]
    
    print(f"\nRecommendations for user: {user_id}")
    
    # User-based collaborative filtering
    print("\n1. User-Based Collaborative Filtering:")
    user_based = engine.get_user_similarity_recommendations(user_id, top_n=5)
    for product, score in user_based.items():
        print(f"   {product}: {score:.4f}")
    
    # SVD-based
    print("\n2. Matrix Factorization (SVD):")
    svd_based = engine.get_svd_recommendations(user_id, top_n=5)
    for product, score in svd_based.items():
        print(f"   {product}: {score:.4f}")
    
    # Ensemble
    print("\n3. Ensemble (50-50 blend):")
    ensemble = engine.get_ensemble_recommendations(user_id, top_n=5, weight_user_based=0.5)
    for product, score in ensemble.items():
        print(f"   {product}: {score:.4f}")


def example_3_user_statistics():
    """Example 3: Get user statistics"""
    print("\n" + "="*60)
    print("EXAMPLE 3: User Statistics")
    print("="*60)
    
    engine = RecommendationEngine(file_path='../retail_data.xlsx')
    users = engine.get_all_users()
    
    print(f"\nStatistics for first 3 users:")
    for user_id in users[:3]:
        stats = engine.get_user_stats(user_id)
        print(f"\n{user_id}:")
        print(f"  Items rated: {stats['items_rated']}")
        print(f"  Average rating: {stats['avg_rating']:.2f}")
        print(f"  Rating range: {stats['min_rating']:.0f} - {stats['max_rating']:.0f}")
        print(f"  Items not yet rated: {stats['items_not_rated']}")


def example_4_batch_recommendations():
    """Example 4: Get recommendations for multiple users"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Batch Recommendations")
    print("="*60)
    
    engine = RecommendationEngine(file_path='../retail_data.xlsx')
    users = engine.get_all_users()[:3]
    
    results = {}
    for user_id in users:
        recs = engine.get_user_similarity_recommendations(user_id, top_n=3)
        results[user_id] = recs.to_dict()
        print(f"\n{user_id}: {list(recs.index)[:3]}")


def example_5_custom_parameters():
    """Example 5: Use custom parameters"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Custom Parameters")
    print("="*60)
    
    engine = RecommendationEngine(file_path='../retail_data.xlsx')
    users = engine.get_all_users()
    user_id = users[0]
    
    print(f"\nRecommendations for {user_id} with different parameters:")
    
    # Different number of similar users
    print("\nUsing 10 similar users:")
    recs = engine.get_user_similarity_recommendations(user_id, top_n=5, num_similar_users=10)
    print(list(recs.index))
    
    # Different number of factors for SVD
    print("\nUsing 5 factors for SVD:")
    recs = engine.get_svd_recommendations(user_id, top_n=5, n_factors=5)
    print(list(recs.index))
    
    # Different ensemble weights
    print("\nWeighting user-based more (70%):")
    recs = engine.get_ensemble_recommendations(user_id, top_n=5, weight_user_based=0.7)
    print(list(recs.index))


if __name__ == "__main__":
    example_1_simple_recommendations()
    example_2_multiple_methods()
    example_3_user_statistics()
    example_4_batch_recommendations()
    example_5_custom_parameters()
    
    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60)
