"""Tests for statistical analysis tools."""
import pytest
import numpy as np
from scipy import stats
from fractiverse.statistics.analysis import (
    TimeSeriesAnalysis,
    PatternAnalysis,
    CorrelationAnalysis,
    SignificanceTest
)

@pytest.fixture
def test_data():
    """Provide test data for statistical analysis."""
    np.random.seed(42)
    return {
        "time_series": np.random.normal(0, 1, 1000),
        "patterns": np.random.rand(100, 10),
        "categories": np.random.choice(['A', 'B', 'C'], 100)
    }

def test_time_series_analysis(test_data):
    """Test time series analysis functions."""
    analyzer = TimeSeriesAnalysis()
    
    # Test stationarity
    stationarity = analyzer.check_stationarity(test_data["time_series"])
    assert isinstance(stationarity.p_value, float)
    assert 0 <= stationarity.p_value <= 1
    
    # Test autocorrelation
    acf = analyzer.compute_autocorrelation(test_data["time_series"], max_lag=10)
    assert len(acf) == 11  # Including lag 0
    assert np.isclose(acf[0], 1.0)  # Correlation with self
    
    # Test trend detection
    trend = analyzer.detect_trend(test_data["time_series"])
    assert "slope" in trend
    assert "p_value" in trend
    assert np.isfinite(trend["slope"])

def test_pattern_analysis(test_data):
    """Test pattern analysis methods."""
    analyzer = PatternAnalysis()
    
    # Test clustering
    clusters = analyzer.find_clusters(test_data["patterns"], n_clusters=3)
    assert len(np.unique(clusters.labels)) == 3
    assert len(clusters.labels) == len(test_data["patterns"])
    
    # Test dimensionality reduction
    reduced = analyzer.reduce_dimensions(test_data["patterns"], n_components=2)
    assert reduced.shape == (100, 2)
    
    # Test pattern similarity
    similarity = analyzer.compute_similarity_matrix(test_data["patterns"])
    assert similarity.shape == (100, 100)
    assert np.allclose(similarity, similarity.T)  # Symmetric
    assert np.allclose(np.diag(similarity), 1.0)  # Self-similarity

def test_correlation_analysis(test_data):
    """Test correlation analysis tools."""
    analyzer = CorrelationAnalysis()
    
    # Generate correlated data
    x = np.random.normal(0, 1, 100)
    y = x + np.random.normal(0, 0.1, 100)  # Strong correlation
    z = np.random.normal(0, 1, 100)  # No correlation
    
    # Test Pearson correlation
    corr_strong = analyzer.pearson_correlation(x, y)
    corr_weak = analyzer.pearson_correlation(x, z)
    
    assert abs(corr_strong.coefficient) > 0.9  # Strong correlation
    assert abs(corr_weak.coefficient) < 0.2  # Weak correlation
    assert 0 <= corr_strong.p_value <= 1
    
    # Test correlation matrix
    matrix = analyzer.correlation_matrix(test_data["patterns"])
    assert matrix.shape == (10, 10)
    assert np.allclose(matrix, matrix.T)

def test_significance_testing(test_data):
    """Test statistical significance tests."""
    tester = SignificanceTest()
    
    # Split data into groups by category
    groups = [
        test_data["time_series"][test_data["categories"] == cat]
        for cat in ['A', 'B', 'C']
    ]
    
    # Test t-test
    t_test = tester.t_test(groups[0], groups[1])
    assert "statistic" in t_test
    assert "p_value" in t_test
    assert 0 <= t_test["p_value"] <= 1
    
    # Test ANOVA
    anova = tester.anova(groups)
    assert "f_statistic" in anova
    assert "p_value" in anova
    assert 0 <= anova["p_value"] <= 1
    
    # Test chi-square
    observed = np.array([len(g) for g in groups])
    chi_square = tester.chi_square_test(observed)
    assert "statistic" in chi_square
    assert "p_value" in chi_square

@pytest.mark.parametrize("test_type,alpha", [
    ("t-test", 0.05),
    ("chi-square", 0.01),
    ("anova", 0.001)
])
def test_statistical_tests(test_data, test_type, alpha):
    """Test various statistical test methods."""
    tester = SignificanceTest()
    
    if test_type == "t-test":
        # Split data into two groups
        result = tester.t_test(
            test_data["time_series"][:500],
            test_data["time_series"][500:],
            alpha=alpha
        )
    elif test_type == "chi-square":
        # Count category frequencies
        observed = np.bincount(np.searchsorted(
            ['A', 'B', 'C'], 
            test_data["categories"]
        ))
        result = tester.chi_square_test(observed, alpha=alpha)
    else:  # anova
        # Split into three groups
        groups = np.array_split(test_data["time_series"], 3)
        result = tester.anova(groups, alpha=alpha)
    
    assert "p_value" in result
    assert "reject_null" in result
    assert isinstance(result["reject_null"], bool)
    assert 0 <= result["p_value"] <= 1 