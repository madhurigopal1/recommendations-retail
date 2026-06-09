# Retail Product Recommendation System

[![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: PEP 8](https://img.shields.io/badge/code%20style-PEP%208-green.svg)](https://www.python.org/dev/peps/pep-0008/)
[![GitHub stars](https://img.shields.io/github/stars/madhurigopal1/recommendations-retail.svg?style=social)](https://github.com/madhurigopal1/recommendations-retail)

A Python-based collaborative filtering recommendation engine that suggests products to customers using matrix factorization and user-based similarity analysis. Perfect for e-commerce platforms, retail businesses, and recommendation-driven applications.

## 🎯 What This Project Does

Build a production-ready recommendation system that predicts customer preferences and suggests products they'll likely purchase or enjoy. Implements multiple algorithms:
- **User-Based Collaborative Filtering**: Find similar customers and recommend their preferences
- **Matrix Factorization (SVD)**: Predict ratings using dimensionality reduction
- **Ensemble Methods**: Combine algorithms for improved accuracy

## ✨ Features

- **User-Based Collaborative Filtering**: Finds similar customers and recommends their preferred products
- **Matrix Factorization (SVD)**: Predicts ratings for unseen products using dimensionality reduction
- **Ensemble Method**: Combines multiple algorithms for better recommendations
- **Data Preprocessing**: Handles customer data and product ratings efficiently
- **Flexible Configuration**: Customizable number of recommendations and similarity metrics
- **Well-Documented**: Comprehensive examples and algorithm explanations
- **Production-Ready**: Handles edge cases and includes error handling

## 📊 Technology Stack

| Component | Library | Purpose |
|-----------|---------|----------|
| **Core** | Python 3.x | Programming language |
| **Data** | pandas | Data manipulation and pivot tables |
| **ML** | scikit-learn | Cosine similarity computation |
| **Math** | scipy, numpy | SVD decomposition, numerical operations |
| **Testing** | pytest | Unit test framework |

## 📁 Project Structure

```
recommendations-retail/
├── README.md                      # Project documentation
├── LICENSE                        # MIT License
├── CONTRIBUTING.md                # Contributing guidelines
├── CHANGELOG.md                   # Version history
├── requirements.txt               # Python dependencies
├── recommendations_retail.py      # Main recommendation engine
├── retail_data.xlsx              # Sample dataset
├── docs/
│   └── BLOG_POST_OUTLINE.md       # Dev.to blog post guide
├── examples/
│   └── basic_usage.py             # Quick start examples
└── tests/
    └── test_recommendations.py    # Unit tests
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/madhurigopal1/recommendations-retail.git
cd recommendations-retail

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from recommendations_retail import RecommendationEngine

# Initialize the engine
engine = RecommendationEngine(file_path='retail_data.xlsx')

# Get user-based recommendations
user_recs = engine.get_user_similarity_recommendations(user_id='C001', top_n=5)
print("User-Based Recommendations:")
print(user_recs)

# Get SVD-based recommendations
svd_recs = engine.get_svd_recommendations(user_id='C001', top_n=5)
print("\nSVD Recommendations:")
print(svd_recs)

# Get ensemble recommendations (best accuracy)
ensemble_recs = engine.get_ensemble_recommendations(user_id='C001', top_n=5)
print("\nEnsemble Recommendations:")
print(ensemble_recs)
```

## 📈 Advanced Usage

### Custom Parameters

```python
# User-based with more similar users
recommendations = engine.get_user_similarity_recommendations(
    user_id='C001',
    top_n=10,
    num_similar_users=5
)

# SVD with custom factors
svd_recommendations = engine.get_svd_recommendations(
    user_id='C001',
    top_n=10,
    n_factors=3
)

# Ensemble with custom weights
ensemble_recs = engine.get_ensemble_recommendations(
    user_id='C001',
    top_n=5,
    weight_user_based=0.5,
    weight_svd=0.5
)
```

## 📊 Dataset Requirements

The input data should have the following structure:

| Column | Type | Description |
|--------|------|-------------|
| `customer_id` | String | Unique customer identifier |
| `product_id` | String | Unique product identifier |
| `rating` | Numeric | Customer's rating (typically 1-5) |

### Example Data Format

```csv
customer_id, product_id, rating
C001, P001, 5
C001, P002, 4
C002, P001, 3
C002, P003, 4
C003, P002, 5
```

## 🧠 Algorithm Details

### User-Based Collaborative Filtering

**How it works:**
1. Compute cosine similarity between all users
2. Find K most similar users to target user
3. Aggregate ratings from similar users (weighted by similarity)
4. Exclude items already rated by target user
5. Return top-N products with highest predicted scores

**Complexity**: O(m² × n) where m = users, n = items

**Best for**: 
- Small to medium datasets (< 100K ratings)
- Dense rating matrices
- Fast computation needed

### Matrix Factorization (SVD)

**How it works:**
1. Create user-item rating matrix
2. Mean-center the ratings
3. Apply Singular Value Decomposition (SVD)
4. Reconstruct predicted ratings from latent factors
5. Filter already-purchased items
6. Return top-N products by predicted rating

**Complexity**: O(k × m × n) where k = number of factors

**Best for**:
- Large sparse datasets
- Capturing latent patterns
- Production systems

### Ensemble Method

Combines both algorithms for better accuracy:
```python
final_score = (weight_user * user_based_score) + (weight_svd * svd_score)
```

## ⚡ Performance Considerations

| Dataset Size | User-Based CF | SVD | Ensemble |
|--------------|---------------|-----|----------|
| 10K ratings | <100ms | <150ms | <250ms |
| 100K ratings | 500ms | 300ms | 800ms |
| 1M ratings | >5s | 1.5s | 6.5s |

**Optimization tips:**
- Use sparse matrix operations for datasets > 100K ratings
- Cache similarity matrices for repeated queries
- SVD with k=3 works well for most use cases
- User-based CF works best with dense rating matrices
- Consider distributed computing for 10M+ ratings

## 📊 Evaluation Metrics

Common metrics for recommendation systems:

| Metric | Formula | Use Case |
|--------|---------|----------|
| **MAE** | Σ\|predicted - actual\| / n | Overall accuracy |
| **RMSE** | √(Σ(predicted - actual)² / n) | Penalizes large errors |
| **Precision@K** | Relevant items in top-K / K | Recommendation quality |
| **Recall@K** | Relevant items in top-K / total relevant | Coverage |
| **NDCG@K** | Ranking quality metric | Ranking performance |

## ⚠️ Known Limitations

- **Cold-Start Problem**: Cannot recommend for new users with few ratings
- **Data Sparsity**: Works best with dense rating matrices
- **Scalability**: User-based CF doesn't scale well to millions of users
- **No Temporal Dynamics**: Doesn't account for changing preferences over time
- **No Content Features**: Ignores product attributes or categories

## 🔮 Future Enhancements

- [ ] Content-based filtering (product attributes & categories)
- [ ] Hybrid recommendation system
- [ ] Deep learning models (neural collaborative filtering)
- [ ] Real-time prediction API (FastAPI/Flask)
- [ ] A/B testing framework
- [ ] Model evaluation and comparison utilities
- [ ] Handling temporal dynamics (trending products)
- [ ] Distributed computing support (Spark)

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas for Contribution
- 🐛 Bug fixes and improvements
- ✨ New recommendation algorithms (content-based, hybrid)
- 📚 Documentation and tutorials
- 🧪 Test coverage expansion
- ⚡ Performance optimizations
- 🎨 Visualization and demo website

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

Created by: [@madhurigopal1](https://github.com/madhurigopal1)

## 📚 References & Resources

### Research Papers
- [Collaborative Filtering - Wikipedia](https://en.wikipedia.org/wiki/Collaborative_filtering)
- [Matrix Factorization Techniques for Recommender Systems](https://datajobs.com/data-science-repo/Recommender-Systems-(Netflix).pdf)
- [Netflix Prize Competition](https://www.kaggle.com/netflix-inc/netflix-prize-data)

### Documentation
- [scikit-learn Similarity Metrics](https://scikit-learn.org/)
- [SciPy SVD Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html)
- [Pandas Data Manipulation](https://pandas.pydata.org/docs/)

### Learning Resources
- [Recommendation Systems Handbook](https://www.springer.com/gp/book/9780387858197)
- [Coursera ML Specialization](https://www.coursera.org/specializations/machine-learning)
- [Fast.ai Collaborative Filtering](https://www.fast.ai/)

## 🌟 Support

If this project helped you, please consider:
- ⭐ Starring the repository
- 🔗 Sharing with your network
- 💬 Opening issues for bugs or feature requests
- 🤝 Contributing improvements

---

**Happy Recommending!** 🚀
