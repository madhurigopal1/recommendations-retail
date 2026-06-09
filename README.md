# Retail Product Recommendation System

A Python-based collaborative filtering recommendation engine that suggests products to customers using matrix factorization and user-based similarity analysis.

## Features

- **User-Based Collaborative Filtering**: Finds similar customers and recommends their preferred products
- **Matrix Factorization (SVD)**: Predicts ratings for unseen products using dimensionality reduction
- **Ensemble Method**: Combines multiple algorithms for better recommendations
- **Data Preprocessing**: Handles customer data and product ratings efficiently
- **Flexible Configuration**: Customizable number of recommendations and similarity metrics

## Technology Stack

- **Python 3.x**
- **pandas**: Data manipulation and pivot table operations
- **scikit-learn**: Cosine similarity computation
- **scipy**: SVD decomposition for matrix factorization
- **numpy**: Numerical computations

## Project Structure

```
recommendations-retail/
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── recommendations_retail.py      # Main recommendation engine
├── retail_data.xlsx              # Sample dataset
├── examples/                      # Usage examples
│   └── basic_usage.py
└── tests/                        # Unit tests
    └── test_recommendations.py
```

## Dataset Requirements

The input data should have the following structure:

| Column | Type | Description |
|--------|------|-------------|
| `customer_id` | String | Unique customer identifier |
| `product_id` | String | Unique product identifier |
| `rating` | Numeric | Customer's rating (typically 1-5) |

### Example Data Format

```
customer_id, product_id, rating
C001, P001, 5
C001, P002, 4
C002, P001, 3
...
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/madhurigopal1/recommendations-retail.git
cd recommendations-retail
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

```python
from recommendations_retail import RecommendationEngine

# Initialize the engine
engine = RecommendationEngine(file_path='retail_data.xlsx')

# Get user-based recommendations
user_recs = engine.get_user_similarity_recommendations(user_id='C001', top_n=5)
print(user_recs)

# Get SVD-based recommendations
svd_recs = engine.get_svd_recommendations(user_id='C001', top_n=5)
print(svd_recs)
```

### Advanced Usage

```python
# Get recommendations with scores
engine = RecommendationEngine(file_path='retail_data.xlsx')

# User-based with similarity details
recommendations = engine.get_user_similarity_recommendations(
    user_id='C001',
    top_n=10,
    num_similar_users=5
)

# SVD-based with custom parameters
svd_recommendations = engine.get_svd_recommendations(
    user_id='C001',
    top_n=10,
    n_factors=3
)

# Ensemble approach
ensemble_recs = engine.get_ensemble_recommendations(
    user_id='C001',
    top_n=5,
    weight_user_based=0.5
)
```

## Algorithm Details

### User-Based Collaborative Filtering

1. Compute cosine similarity between all users
2. Find the K most similar users to the target user
3. Aggregate ratings from similar users (weighted by similarity)
4. Exclude items already rated by the target user
5. Return top-N products with highest predicted scores

**Complexity**: O(m² × n) where m = users, n = items

### Matrix Factorization (SVD)

1. Create user-item rating matrix
2. Mean-center the ratings
3. Apply Singular Value Decomposition (SVD)
4. Reconstruct predicted ratings
5. Filter already-purchased items
6. Return top-N products by predicted rating

**Complexity**: O(k × m × n) where k = number of factors

## Performance Considerations

- For large datasets (>100K ratings), consider sparse matrix operations
- SVD with k=3 works well for most use cases
- User-based CF works better with dense rating matrices
- Cache similarity matrices for repeated queries

## Evaluation Metrics

Common metrics for recommendation systems:

- **Mean Absolute Error (MAE)**: Average difference between predicted and actual ratings
- **Root Mean Squared Error (RMSE)**: Penalizes larger prediction errors
- **Precision@K**: Fraction of recommended items that are relevant
- **Recall@K**: Fraction of relevant items that were recommended

## Known Limitations

- Cold-start problem: Cannot recommend for new users with few ratings
- Data sparsity: Works best with dense rating matrices
- Scalability: User-based CF doesn't scale well to millions of users
- No temporal dynamics: Doesn't account for changing preferences

## Future Enhancements

- [ ] Content-based filtering (product attributes)
- [ ] Hybrid recommendation system
- [ ] Deep learning models (neural collaborative filtering)
- [ ] Real-time prediction API
- [ ] A/B testing framework
- [ ] Model evaluation and comparison utilities

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/enhancement`)
3. Commit changes (`git commit -am 'Add enhancement'`)
4. Push to branch (`git push origin feature/enhancement`)
5. Open a Pull Request

## License

This project is open source and available under the MIT License.

## Author

Created by: @madhurigopal1

## References

- [Collaborative Filtering - Wikipedia](https://en.wikipedia.org/wiki/Collaborative_filtering)
- [Matrix Factorization Techniques for Recommender Systems](https://datajobs.com/data-science-repo/Recommender-Systems-(Netflix).pdf)
- [scikit-learn Documentation](https://scikit-learn.org/)
