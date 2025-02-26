"""Statistical analysis tools module."""
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from typing import Dict, Any, List, Optional

class TimeSeriesAnalysis:
    """Time series analysis tools."""
    
    def check_stationarity(self, series):
        """Check time series stationarity using Augmented Dickey-Fuller test.
        
        Args:
            series (np.ndarray): Time series data
            
        Returns:
            dict: Test results
        """
        result = stats.adfuller(series)
        return {
            "statistic": result[0],
            "p_value": result[1],
            "is_stationary": result[1] < 0.05
        }
        
    def compute_autocorrelation(self, series, max_lag=None):
        """Compute autocorrelation function.
        
        Args:
            series (np.ndarray): Time series data
            max_lag (int, optional): Maximum lag to compute
            
        Returns:
            np.ndarray: Autocorrelation values
        """
        return stats.acf(series, nlags=max_lag)
        
    def detect_trend(self, series):
        """Detect linear trend in time series.
        
        Args:
            series (np.ndarray): Time series data
            
        Returns:
            dict: Trend analysis results
        """
        x = np.arange(len(series))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
        
        return {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_value**2,
            "p_value": p_value,
            "std_err": std_err
        }

class PatternAnalysis:
    """Pattern analysis tools."""
    
    def find_clusters(self, data, n_clusters=3, random_state=42):
        """Find clusters in data using K-means.
        
        Args:
            data (np.ndarray): Input data
            n_clusters (int): Number of clusters
            random_state (int): Random seed
            
        Returns:
            object: Clustering results
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels = kmeans.fit_predict(data)
        
        return type("ClusterResults", (), {
            "labels": labels,
            "centers": kmeans.cluster_centers_,
            "inertia": kmeans.inertia_
        })
        
    def reduce_dimensions(self, data, n_components=2):
        """Reduce data dimensionality using PCA.
        
        Args:
            data (np.ndarray): Input data
            n_components (int): Number of components
            
        Returns:
            np.ndarray: Reduced data
        """
        pca = PCA(n_components=n_components)
        return pca.fit_transform(data)
        
    def compute_similarity_matrix(self, data):
        """Compute pairwise similarity matrix.
        
        Args:
            data (np.ndarray): Input data
            
        Returns:
            np.ndarray: Similarity matrix
        """
        # Using correlation as similarity measure
        return np.corrcoef(data)

class CorrelationAnalysis:
    """Correlation analysis tools."""
    
    def pearson_correlation(self, x, y):
        """Compute Pearson correlation coefficient.
        
        Args:
            x (np.ndarray): First variable
            y (np.ndarray): Second variable
            
        Returns:
            object: Correlation results
        """
        coef, p_value = stats.pearsonr(x, y)
        return type("CorrelationResults", (), {
            "coefficient": coef,
            "p_value": p_value
        })
        
    def correlation_matrix(self, data):
        """Compute correlation matrix for multivariate data.
        
        Args:
            data (np.ndarray): Input data
            
        Returns:
            np.ndarray: Correlation matrix
        """
        return np.corrcoef(data.T)

class SignificanceTest:
    """Statistical significance testing tools."""
    
    def t_test(self, group1, group2, alpha=0.05):
        """Perform independent t-test.
        
        Args:
            group1 (np.ndarray): First group
            group2 (np.ndarray): Second group
            alpha (float): Significance level
            
        Returns:
            dict: Test results
        """
        t_stat, p_value = stats.ttest_ind(group1, group2)
        return {
            "statistic": t_stat,
            "p_value": p_value,
            "reject_null": p_value < alpha
        }
        
    def anova(self, groups, alpha=0.05):
        """Perform one-way ANOVA.
        
        Args:
            groups (list): List of groups
            alpha (float): Significance level
            
        Returns:
            dict: Test results
        """
        f_stat, p_value = stats.f_oneway(*groups)
        return {
            "f_statistic": f_stat,
            "p_value": p_value,
            "reject_null": p_value < alpha
        }
        
    def chi_square_test(self, observed, alpha=0.05):
        """Perform chi-square test.
        
        Args:
            observed (np.ndarray): Observed frequencies
            alpha (float): Significance level
            
        Returns:
            dict: Test results
        """
        # Assuming uniform expected frequencies
        n = len(observed)
        expected = np.full(n, np.sum(observed) / n)
        
        chi2_stat, p_value = stats.chisquare(observed, expected)
        return {
            "statistic": chi2_stat,
            "p_value": p_value,
            "reject_null": p_value < alpha
        } 