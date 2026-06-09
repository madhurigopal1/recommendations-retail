# -*- coding: utf-8 -*-
"""
Unit tests for the Recommendation Engine
"""

import sys
import unittest
import pandas as pd
import numpy as np

sys.path.insert(0, '..')

from recommendations_retail import RecommendationEngine


class TestRecommendationEngine(unittest.TestCase):
    """Test cases for RecommendationEngine class"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        # Create sample data
        data = {
            'customer_id': ['C001', 'C001', 'C001', 'C002', 'C002', 'C002', 'C003', 'C003', 'C003'],
            'product_id': ['P001', 'P002', 'P003', 'P001', 'P002', 'P004', 'P002', 'P003', 'P004'],
            'rating': [5, 4, 3, 5, 5, 2, 4, 3, 5]
        }
        cls.test_df = pd.DataFrame(data)
    
    def setUp(self):
        """Set up for each test"""
        self.engine = RecommendationEngine(df=self.test_df)
    
    def test_initialization(self):
        """Test engine initialization"""
        self.assertIsNotNone(self.engine.df)
        self.assertIsNotNone(self.engine.user_item_matrix)
        self.assertEqual(len(self.engine.user_item_matrix), 3)  # 3 users
    
    def test_missing_columns(self):
        """Test error handling for missing columns"""
        bad_df = pd.DataFrame({
            'customer_id': ['C001'],
            'product_id': ['P001']
            # Missing 'rating' column
        })
        with self.assertRaises(ValueError):
            RecommendationEngine(df=bad_df)
    
    def test_user_similarity_recommendations(self):
        """Test user-based collaborative filtering"""
        recs = self.engine.get_user_similarity_recommendations('C001', top_n=2)
        
        # Should return recommendations
        self.assertGreater(len(recs), 0)
        self.assertLessEqual(len(recs), 2)
        
        # All scores should be non-negative
        self.assertTrue((recs >= 0).all())
    
    def test_svd_recommendations(self):
        """Test SVD-based recommendations"""
        recs = self.engine.get_svd_recommendations('C001', top_n=2)
        
        # Should return recommendations
        self.assertGreater(len(recs), 0)
        self.assertLessEqual(len(recs), 2)
    
    def test_ensemble_recommendations(self):
        """Test ensemble recommendations"""
        recs = self.engine.get_ensemble_recommendations('C001', top_n=2, weight_user_based=0.5)
        
        # Should return recommendations
        self.assertGreater(len(recs), 0)
        self.assertLessEqual(len(recs), 2)
        
        # All scores should be in [0, 1]
        self.assertTrue((recs >= 0).all())
        self.assertTrue((recs <= 1).all())
    
    def test_invalid_user(self):
        """Test handling of invalid user ID"""
        with self.assertRaises(ValueError):
            self.engine.get_user_similarity_recommendations('INVALID_USER')
    
    def test_user_stats(self):
        """Test user statistics"""
        stats = self.engine.get_user_stats('C001')
        
        self.assertEqual(stats['user_id'], 'C001')
        self.assertEqual(stats['items_rated'], 3)
        self.assertEqual(stats['avg_rating'], 4.0)
        self.assertEqual(stats['min_rating'], 3.0)
        self.assertEqual(stats['max_rating'], 5.0)
    
    def test_get_all_users(self):
        """Test getting all users"""
        users = self.engine.get_all_users()
        
        self.assertEqual(len(users), 3)
        self.assertIn('C001', users)
        self.assertIn('C002', users)
        self.assertIn('C003', users)
    
    def test_get_all_products(self):
        """Test getting all products"""
        products = self.engine.get_all_products()
        
        self.assertEqual(len(products), 4)
        self.assertIn('P001', products)
        self.assertIn('P002', products)
        self.assertIn('P003', products)
        self.assertIn('P004', products)
    
    def test_top_n_parameter(self):
        """Test that top_n parameter is respected"""
        for n in [1, 2, 3]:
            recs = self.engine.get_user_similarity_recommendations('C001', top_n=n)
            self.assertLessEqual(len(recs), n)
    
    def test_recommendations_are_sorted(self):
        """Test that recommendations are sorted in descending order"""
        recs = self.engine.get_user_similarity_recommendations('C001', top_n=5)
        
        # Check if sorted in descending order
        for i in range(len(recs) - 1):
            self.assertGreaterEqual(recs.iloc[i], recs.iloc[i + 1])
    
    def test_no_duplicate_recommendations(self):
        """Test that no duplicate recommendations are returned"""
        recs = self.engine.get_user_similarity_recommendations('C001', top_n=5)
        
        # All products should be unique
        self.assertEqual(len(recs), len(set(recs.index)))
    
    def test_different_weights(self):
        """Test ensemble with different weights"""
        recs_50_50 = self.engine.get_ensemble_recommendations('C001', weight_user_based=0.5)
        recs_70_30 = self.engine.get_ensemble_recommendations('C001', weight_user_based=0.7)
        recs_30_70 = self.engine.get_ensemble_recommendations('C001', weight_user_based=0.3)
        
        # All should return recommendations
        self.assertGreater(len(recs_50_50), 0)
        self.assertGreater(len(recs_70_30), 0)
        self.assertGreater(len(recs_30_70), 0)


class TestDataValidation(unittest.TestCase):
    """Test data validation"""
    
    def test_no_data_provided(self):
        """Test error when no data is provided"""
        with self.assertRaises(ValueError):
            RecommendationEngine()
    
    def test_file_not_found(self):
        """Test error when file is not found"""
        with self.assertRaises(FileNotFoundError):
            RecommendationEngine(file_path='nonexistent_file.xlsx')


class TestRecommendationQuality(unittest.TestCase):
    """Test recommendation quality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        # Create a larger sample dataset
        np.random.seed(42)
        n_users = 10
        n_products = 20
        
        data = []
        for user_id in range(n_users):
            # Each user rates 5-10 random products
            n_ratings = np.random.randint(5, 11)
            product_ids = np.random.choice(n_products, n_ratings, replace=False)
            ratings = np.random.randint(1, 6, n_ratings)
            
            for product_id, rating in zip(product_ids, ratings):
                data.append({
                    'customer_id': f'C{user_id:03d}',
                    'product_id': f'P{product_id:03d}',
                    'rating': rating
                })
        
        cls.test_df = pd.DataFrame(data)
    
    def setUp(self):
        """Set up for each test"""
        self.engine = RecommendationEngine(df=self.test_df)
    
    def test_recommendations_exist(self):
        """Test that recommendations are generated"""
        users = self.engine.get_all_users()
        
        for user_id in users[:3]:
            recs = self.engine.get_user_similarity_recommendations(user_id, top_n=5)
            self.assertGreater(len(recs), 0, f"No recommendations for {user_id}")
    
    def test_no_already_rated_items(self):
        """Test that already-rated items are not recommended"""
        user_id = self.engine.get_all_users()[0]
        
        # Get user's already rated items
        user_ratings = self.engine.user_item_matrix.loc[user_id]
        already_rated = set(user_ratings[user_ratings > 0].index)
        
        # Get recommendations
        recs = self.engine.get_user_similarity_recommendations(user_id, top_n=10)
        recommended_items = set(recs.index)
        
        # Check no overlap
        overlap = already_rated & recommended_items
        self.assertEqual(len(overlap), 0, f"Already-rated items recommended: {overlap}")


if __name__ == '__main__':
    unittest.main()
