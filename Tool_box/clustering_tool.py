"""
Clustering Tool - Enhanced clustering algorithms for machine learning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering,
    Birch, MiniBatchKMeans, OPTICS, MeanShift, AffinityPropagation,
    GaussianMixture
)
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Dict, List, Optional, Union, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

class ClusteringTool:
    """An enhanced tool for various clustering algorithms and evaluation."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.cluster_models = {}
        self.cluster_labels = {}
        self.cluster_centers = {}
        self.cluster_metrics = {}

    def kmeans_clustering(self, X: pd.DataFrame, n_clusters: int = 3,
                         init: str = 'k-means++', n_init: int = 10,
                         max_iter: int = 300, **kwargs) -> Dict:
        """
        Perform K-Means clustering.

        Args:
            X: Feature matrix
            n_clusters: Number of clusters
            init: Initialization method
            n_init: Number of initializations
            max_iter: Maximum iterations
            **kwargs: Additional parameters

        Returns:
            Clustering results
        """
        try:
            kmeans = KMeans(
                n_clusters=n_clusters,
                init=init,
                n_init=n_init,
                max_iter=max_iter,
                random_state=self.random_state,
                **kwargs
            )

            labels = kmeans.fit_predict(X)
            centers = kmeans.cluster_centers_

            results = {
                'algorithm': 'kmeans',
                'n_clusters': n_clusters,
                'labels': labels,
                'centers': centers,
                'inertia': kmeans.inertia_,
                'n_iter': kmeans.n_iter_,
                'model': kmeans
            }

            self.cluster_models['kmeans'] = kmeans
            self.cluster_labels['kmeans'] = labels
            self.cluster_centers['kmeans'] = centers

            return results

        except Exception as e:
            return {'error': str(e)}

    def dbscan_clustering(self, X: pd.DataFrame, eps: float = 0.5,
                         min_samples: int = 5, **kwargs) -> Dict:
        """
        Perform DBSCAN clustering.

        Args:
            X: Feature matrix
            eps: Maximum distance between samples
            min_samples: Minimum samples in neighborhood
            **kwargs: Additional parameters

        Returns:
            Clustering results
        """
        try:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
            labels = dbscan.fit_predict(X)

            # Calculate number of clusters (excluding noise labeled as -1)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            results = {
                'algorithm': 'dbscan',
                'eps': eps,
                'min_samples': min_samples,
                'labels': labels,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'model': dbscan
            }

            self.cluster_models['dbscan'] = dbscan
            self.cluster_labels['dbscan'] = labels

            return results

        except Exception as e:
            return {'error': str(e)}

    def hierarchical_clustering(self, X: pd.DataFrame, n_clusters: int = 3,
                               linkage: str = 'ward', **kwargs) -> Dict:
        """
        Perform Hierarchical (Agglomerative) clustering.

        Args:
            X: Feature matrix
            n_clusters: Number of clusters
            linkage: Linkage criterion ('ward', 'complete', 'average', 'single')
            **kwargs: Additional parameters

        Returns:
            Clustering results
        """
        try:
            hierarchical = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage,
                **kwargs
            )

            labels = hierarchical.fit_predict(X)

            results = {
                'algorithm': 'hierarchical',
                'n_clusters': n_clusters,
                'linkage': linkage,
                'labels': labels,
                'model': hierarchical
            }

            self.cluster_models['hierarchical'] = hierarchical
            self.cluster_labels['hierarchical'] = labels

            return results

        except Exception as e:
            return {'error': str(e)}

    def spectral_clustering(self, X: pd.DataFrame, n_clusters: int = 3,
                          affinity: str = 'rbf', **kwargs) -> Dict:
        """
        Perform Spectral clustering.

        Args:
            X: Feature matrix
            n_clusters: Number of clusters
            affinity: Affinity matrix type
            **kwargs: Additional parameters

        Returns:
            Clustering results
        """
        try:
            spectral = SpectralClustering(
                n_clusters=n_clusters,
                affinity=affinity,
                random_state=self.random_state,
                **kwargs
            )

            labels = spectral.fit_predict(X)

            results = {
                'algorithm': 'spectral',
                'n_clusters': n_clusters,
                'affinity': affinity,
                'labels': labels,
                'model': spectral
            }

            self.cluster_models['spectral'] = spectral
            self.cluster_labels['spectral'] = labels

            return results

        except Exception as e:
            return {'error': str(e)}

    def birch_clustering(self, X: pd.DataFrame, n_clusters: Optional[int] = 3,
                        threshold: float = 0.5, branching_factor: int = 50,
                        **kwargs) -> Dict:
        """
        Perform BIRCH clustering.

        Args:
            X: Feature matrix
            n_clusters: Number of clusters (None for no global clustering)
            threshold: Radius threshold
            branching_factor: Maximum branching factor
            **kwargs: Additional parameters

        Returns:
            Clustering results
        """
        try:
            birch = Birch(
                n_clusters=n_clusters,
                threshold=threshold,
                branching_factor=branching_factor,
                **kwargs
            )

            labels = birch.fit_predict(X)

            results = {
                'algorithm': 'birch',
                'n_clusters': n_clusters,
                'threshold': threshold,
                'branching_factor': branching_factor,
                'labels': labels,
                'model': birch
            }

            self.cluster_models['birch'] = birch
            self.cluster_labels['birch'] = labels

            return results

        except Exception as e:
            return {'error': str(e)}

    def optics_clustering(self, X: pd.DataFrame, min_samples: int = 5,
                        max_eps: float = np.inf, **kwargs) -> Dict:
        """
        Perform OPTICS clustering.

        Args:
            X: Feature matrix
            min_samples: Minimum samples in neighborhood
            max_eps: Maximum distance
            **kwargs: Additional parameters

        Returns:
            Clustering results
        """
        try:
            optics = OPTICS(min_samples=min_samples, max_eps=max_eps, **kwargs)
            labels = optics.fit_predict(X)

            # Calculate number of clusters (excluding noise labeled as -1)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            results = {
                'algorithm': 'optics',
                'min_samples': min_samples,
                'max_eps': max_eps,
                'labels': labels,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'model': optics
            }

            self.cluster_models['optics'] = optics
            self.cluster_labels['optics'] = labels

            return results

        except Exception as e:
            return {'error': str(e)}

    def mean_shift_clustering(self, X: pd.DataFrame, bandwidth: Optional[float] = None,
                            **kwargs) -> Dict:
        """
        Perform Mean Shift clustering.

        Args:
            X: Feature matrix
            bandwidth: Kernel bandwidth
            **kwargs: Additional parameters

        Returns:
            Clustering results
        """
        try:
            mean_shift = MeanShift(bandwidth=bandwidth, **kwargs)
            labels = mean_shift.fit_predict(X)

            centers = mean_shift.cluster_centers_
            n_clusters = len(centers)

            results = {
                'algorithm': 'mean_shift',
                'bandwidth': bandwidth,
                'labels': labels,
                'centers': centers,
                'n_clusters': n_clusters,
                'model': mean_shift
            }

            self.cluster_models['mean_shift'] = mean_shift
            self.cluster_labels['mean_shift'] = labels
            self.cluster_centers['mean_shift'] = centers

            return results

        except Exception as e:
            return {'error': str(e)}

    def affinity_propagation_clustering(self, X: pd.DataFrame,
                                      damping: float = 0.5,
                                      preference: Optional[float] = None,
                                      **kwargs) -> Dict:
        """
        Perform Affinity Propagation clustering.

        Args:
            X: Feature matrix
            damping: Damping factor
            preference: Preference parameter
            **kwargs: Additional parameters

        Returns:
            Clustering results
        """
        try:
            affinity = AffinityPropagation(
                damping=damping,
                preference=preference,
                random_state=self.random_state,
                **kwargs
            )

            labels = affinity.fit_predict(X)

            centers_indices = affinity.cluster_centers_indices_
            centers = X.iloc[centers_indices].values if centers_indices is not None else None
            n_clusters = len(centers_indices) if centers_indices is not None else 0

            results = {
                'algorithm': 'affinity_propagation',
                'damping': damping,
                'preference': preference,
                'labels': labels,
                'centers_indices': centers_indices,
                'centers': centers,
                'n_clusters': n_clusters,
                'model': affinity
            }

            self.cluster_models['affinity_propagation'] = affinity
            self.cluster_labels['affinity_propagation'] = labels
            if centers is not None:
                self.cluster_centers['affinity_propagation'] = centers

            return results

        except Exception as e:
            return {'error': str(e)}

    def gaussian_mixture_clustering(self, X: pd.DataFrame, n_components: int = 3,
                                  covariance_type: str = 'full', **kwargs) -> Dict:
        """
        Perform Gaussian Mixture Model clustering.

        Args:
            X: Feature matrix
            n_components: Number of mixture components
            covariance_type: Covariance type ('full', 'tied', 'diag', 'spherical')
            **kwargs: Additional parameters

        Returns:
            Clustering results
        """
        try:
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type=covariance_type,
                random_state=self.random_state,
                **kwargs
            )

            labels = gmm.fit_predict(X)
            probabilities = gmm.predict_proba(X)
            means = gmm.means_
            covariances = gmm.covariances_

            results = {
                'algorithm': 'gaussian_mixture',
                'n_components': n_components,
                'covariance_type': covariance_type,
                'labels': labels,
                'probabilities': probabilities,
                'means': means,
                'covariances': covariances,
                'bic': gmm.bic(X),
                'aic': gmm.aic(X),
                'model': gmm
            }

            self.cluster_models['gaussian_mixture'] = gmm
            self.cluster_labels['gaussian_mixture'] = labels

            return results

        except Exception as e:
            return {'error': str(e)}

    def evaluate_clustering(self, X: pd.DataFrame, labels: np.ndarray,
                          metric: str = 'all') -> Dict:
        """
        Evaluate clustering quality using various metrics.

        Args:
            X: Feature matrix
            labels: Cluster labels
            metric: Evaluation metric ('silhouette', 'calinski_harabasz', 'davies_bouldin', 'all')

        Returns:
            Evaluation metrics
        """
        try:
            metrics = {}

            if metric in ['silhouette', 'all']:
                try:
                    metrics['silhouette'] = silhouette_score(X, labels)
                except:
                    metrics['silhouette'] = None

            if metric in ['calinski_harabasz', 'all']:
                try:
                    metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels)
                except:
                    metrics['calinski_harabasz'] = None

            if metric in ['davies_bouldin', 'all']:
                try:
                    metrics['davies_bouldin'] = davies_bouldin_score(X, labels)
                except:
                    metrics['davies_bouldin'] = None

            return metrics

        except Exception as e:
            return {'error': str(e)}

    def plot_clusters_2d(self, X: pd.DataFrame, labels: np.ndarray,
                        centers: Optional[np.ndarray] = None,
                        algorithm: str = 'Clustering', method: str = 'pca') -> None:
        """
        Plot clusters in 2D space.

        Args:
            X: Feature matrix
            labels: Cluster labels
            centers: Cluster centers (optional)
            algorithm: Algorithm name for plot title
            method: Dimensionality reduction method ('pca', 'tsne', 'original')
        """
        try:
            # Reduce dimensionality if needed
            if X.shape[1] > 2:
                if method == 'pca':
                    pca = PCA(n_components=2, random_state=self.random_state)
                    X_plot = pca.fit_transform(X)
                    xlabel, ylabel = 'PC1', 'PC2'
                elif method == 'tsne':
                    from sklearn.manifold import TSNE
                    tsne = TSNE(n_components=2, random_state=self.random_state)
                    X_plot = tsne.fit_transform(X)
                    xlabel, ylabel = 't-SNE 1', 't-SNE 2'
                else:
                    X_plot = X.iloc[:, :2].values
                    xlabel, ylabel = X.columns[0], X.columns[1]
            else:
                X_plot = X.values
                xlabel, ylabel = X.columns[0] if len(X.columns) > 0 else 'Feature 1', \
                               X.columns[1] if len(X.columns) > 1 else 'Feature 2'

            # Create plot
            plt.figure(figsize=(10, 8))
            unique_labels = set(labels)

            # Plot clusters
            for label in unique_labels:
                if label == -1:  # Noise points in DBSCAN/OPTICS
                    color = 'black'
                    marker = 'x'
                    label_name = 'Noise'
                else:
                    color = plt.cm.Set3(label / len(unique_labels))
                    marker = 'o'
                    label_name = f'Cluster {label}'

                mask = labels == label
                plt.scatter(X_plot[mask, 0], X_plot[mask, 1],
                          c=[color], marker=marker, label=label_name,
                          alpha=0.6, edgecolors='w', s=50)

            # Plot centers if available
            if centers is not None and len(centers) > 0:
                if centers.shape[1] > 2 and method == 'pca':
                    centers_plot = pca.transform(centers)
                elif centers.shape[1] > 2 and method == 'tsne':
                    # Note: This is approximate, centers should be transformed with the same method
                    centers_plot = centers[:, :2]
                else:
                    centers_plot = centers

                plt.scatter(centers_plot[:, 0], centers_plot[:, 1],
                          c='red', marker='x', s=200, linewidths=3,
                          label='Centroids')

            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(f'{algorithm} Clustering Results')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

        except Exception as e:
            print(f"Error plotting clusters: {str(e)}")

    def compare_algorithms(self, X: pd.DataFrame, algorithms: Optional[List[str]] = None,
                         n_clusters: int = 3) -> Dict:
        """
        Compare different clustering algorithms.

        Args:
            X: Feature matrix
            algorithms: List of algorithms to compare
            n_clusters: Number of clusters for algorithms that require it

        Returns:
            Comparison results
        """
        if algorithms is None:
            algorithms = ['kmeans', 'hierarchical', 'spectral', 'dbscan']

        comparison = {}

        for algorithm in algorithms:
            try:
                if algorithm == 'kmeans':
                    result = self.kmeans_clustering(X, n_clusters=n_clusters)
                elif algorithm == 'hierarchical':
                    result = self.hierarchical_clustering(X, n_clusters=n_clusters)
                elif algorithm == 'spectral':
                    result = self.spectral_clustering(X, n_clusters=n_clusters)
                elif algorithm == 'dbscan':
                    result = self.dbscan_clustering(X)
                elif algorithm == 'optics':
                    result = self.optics_clustering(X)
                elif algorithm == 'mean_shift':
                    result = self.mean_shift_clustering(X)
                elif algorithm == 'gaussian_mixture':
                    result = self.gaussian_mixture_clustering(X, n_components=n_clusters)
                else:
                    continue

                if 'error' not in result:
                    # Evaluate clustering quality
                    labels = result['labels']
                    metrics = self.evaluate_clustering(X, labels)

                    comparison[algorithm] = {
                        'n_clusters': result.get('n_clusters', 'N/A'),
                        'silhouette': metrics.get('silhouette'),
                        'calinski_harabasz': metrics.get('calinski_harabasz'),
                        'davies_bouldin': metrics.get('davies_bouldin')
                    }

            except Exception as e:
                comparison[algorithm] = {'error': str(e)}

        return comparison

    def get_cluster_labels(self, algorithm: str) -> Optional[np.ndarray]:
        """Get cluster labels for a specific algorithm."""
        return self.cluster_labels.get(algorithm)

    def get_cluster_centers(self, algorithm: str) -> Optional[np.ndarray]:
        """Get cluster centers for a specific algorithm."""
        return self.cluster_centers.get(algorithm)

    def get_clustering_models(self) -> Dict:
        """Get all clustering models."""
        return self.cluster_models.copy()

    def clear_results(self):
        """Clear all clustering results."""
        self.cluster_models = {}
        self.cluster_labels = {}
        self.cluster_centers = {}
        self.cluster_metrics = {}
