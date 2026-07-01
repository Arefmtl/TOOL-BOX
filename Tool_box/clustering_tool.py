"""
Clustering Tool - Enhanced clustering algorithms for machine learning.

Pipeline Step: Training (Clustering branch)

Supports 14 algorithms with evaluation metrics (Silhouette, CH, DB, Homogeneity, etc.)
and visualizations (Elbow Plot, Silhouette Plot, 2D Cluster Plot).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering,
    Birch, MiniBatchKMeans, OPTICS, MeanShift, AffinityPropagation
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    homogeneity_score, completeness_score, v_measure_score,
    adjusted_rand_score, adjusted_mutual_info_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Dict, List, Optional, Union, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

from .decorators import step

# ── Optional imports ─────────────────────────────────────────────

try:
    from hdbscan import HDBSCAN as HDBSCAN_
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

try:
    from sklearn_extra.cluster import KMedoids
    KMEDOIDS_AVAILABLE = True
except ImportError:
    KMEDOIDS_AVAILABLE = False

try:
    import skfuzzy as fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False


class ClusteringTool:
    """An enhanced tool for various clustering algorithms and evaluation.

    Supports: KMeans, MiniBatchKMeans, DBSCAN, HDBSCAN, Hierarchical, Spectral,
    BIRCH, OPTICS, MeanShift, AffinityPropagation, GMM, K-Medoids, Fuzzy C-Means.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.cluster_models = {}
        self.cluster_labels = {}
        self.cluster_centers = {}
        self.cluster_metrics = {}

    # ── Individual clustering methods ───────────────────────────────

    @step('K-Means Clustering')
    def kmeans_clustering(self, X: pd.DataFrame, n_clusters: int = 3,
                          init: str = 'k-means++', n_init: int = 10,
                          max_iter: int = 300, **kwargs) -> Dict:
        """Perform K-Means clustering."""
        try:
            kmeans = KMeans(
                n_clusters=n_clusters, init=init, n_init=n_init,
                max_iter=max_iter, random_state=self.random_state, **kwargs
            )
            labels = kmeans.fit_predict(X)
            results = {
                'algorithm': 'kmeans', 'n_clusters': n_clusters,
                'labels': labels, 'centers': kmeans.cluster_centers_,
                'inertia': kmeans.inertia_, 'model': kmeans
            }
            self.cluster_models['kmeans'] = kmeans
            self.cluster_labels['kmeans'] = labels
            self.cluster_centers['kmeans'] = kmeans.cluster_centers_
            return results
        except Exception as e:
            return {'error': str(e)}

    @step('MiniBatch K-Means')
    def mini_batch_kmeans(self, X: pd.DataFrame, n_clusters: int = 3,
                          batch_size: int = 1024, **kwargs) -> Dict:
        """Perform MiniBatch K-Means (fast for large datasets, ~10x faster)."""
        try:
            mbk = MiniBatchKMeans(
                n_clusters=n_clusters, batch_size=batch_size,
                random_state=self.random_state, **kwargs
            )
            labels = mbk.fit_predict(X)
            results = {
                'algorithm': 'mini_batch_kmeans', 'n_clusters': n_clusters,
                'batch_size': batch_size, 'labels': labels,
                'centers': mbk.cluster_centers_, 'inertia': mbk.inertia_,
                'model': mbk
            }
            self.cluster_models['mini_batch_kmeans'] = mbk
            self.cluster_labels['mini_batch_kmeans'] = labels
            return results
        except Exception as e:
            return {'error': str(e)}

    @step('DBSCAN Clustering')
    def dbscan_clustering(self, X: pd.DataFrame, eps: float = 0.5,
                          min_samples: int = 5, **kwargs) -> Dict:
        """Perform DBSCAN clustering."""
        try:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
            labels = dbscan.fit_predict(X)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            results = {
                'algorithm': 'dbscan', 'eps': eps, 'min_samples': min_samples,
                'labels': labels, 'n_clusters': n_clusters, 'n_noise': n_noise,
                'model': dbscan
            }
            self.cluster_models['dbscan'] = dbscan
            self.cluster_labels['dbscan'] = labels
            return results
        except Exception as e:
            return {'error': str(e)}

    @step('HDBSCAN Clustering')
    def hdbscan_clustering(self, X: pd.DataFrame, min_cluster_size: int = 5,
                           min_samples: Optional[int] = None, **kwargs) -> Dict:
        """Perform HDBSCAN clustering (no eps parameter needed).

        Args:
            X: Feature matrix
            min_cluster_size: Minimum cluster size
            min_samples: Minimum samples in neighborhood (None = min_cluster_size)
        """
        if not HDBSCAN_AVAILABLE:
            print("Warning: HDBSCAN not available. Install with: pip install hdbscan")
            return {'error': 'HDBSCAN not available'}
        try:
            hdbscan = HDBSCAN_(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                **kwargs
            )
            labels = hdbscan.fit_predict(X)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            results = {
                'algorithm': 'hdbscan', 'min_cluster_size': min_cluster_size,
                'labels': labels, 'n_clusters': n_clusters, 'n_noise': n_noise,
                'model': hdbscan
            }
            self.cluster_models['hdbscan'] = hdbscan
            self.cluster_labels['hdbscan'] = labels
            return results
        except Exception as e:
            return {'error': str(e)}

    @step('Hierarchical Clustering')
    def hierarchical_clustering(self, X: pd.DataFrame, n_clusters: int = 3,
                                linkage: str = 'ward', **kwargs) -> Dict:
        """Perform Hierarchical (Agglomerative) clustering."""
        try:
            hierarchical = AgglomerativeClustering(
                n_clusters=n_clusters, linkage=linkage, **kwargs
            )
            labels = hierarchical.fit_predict(X)
            results = {
                'algorithm': 'hierarchical', 'n_clusters': n_clusters,
                'linkage': linkage, 'labels': labels, 'model': hierarchical
            }
            self.cluster_models['hierarchical'] = hierarchical
            self.cluster_labels['hierarchical'] = labels
            return results
        except Exception as e:
            return {'error': str(e)}

    @step('Spectral Clustering')
    def spectral_clustering(self, X: pd.DataFrame, n_clusters: int = 3,
                            affinity: str = 'rbf', **kwargs) -> Dict:
        """Perform Spectral clustering."""
        try:
            spectral = SpectralClustering(
                n_clusters=n_clusters, affinity=affinity,
                random_state=self.random_state, **kwargs
            )
            labels = spectral.fit_predict(X)
            results = {
                'algorithm': 'spectral', 'n_clusters': n_clusters,
                'affinity': affinity, 'labels': labels, 'model': spectral
            }
            self.cluster_models['spectral'] = spectral
            self.cluster_labels['spectral'] = labels
            return results
        except Exception as e:
            return {'error': str(e)}

    @step('BIRCH Clustering')
    def birch_clustering(self, X: pd.DataFrame, n_clusters: int = 3,
                         threshold: float = 0.5, **kwargs) -> Dict:
        """Perform BIRCH clustering."""
        try:
            birch = Birch(n_clusters=n_clusters, threshold=threshold, **kwargs)
            labels = birch.fit_predict(X)
            results = {
                'algorithm': 'birch', 'n_clusters': n_clusters,
                'threshold': threshold, 'labels': labels, 'model': birch
            }
            self.cluster_models['birch'] = birch
            self.cluster_labels['birch'] = labels
            return results
        except Exception as e:
            return {'error': str(e)}

    @step('OPTICS Clustering')
    def optics_clustering(self, X: pd.DataFrame, min_samples: int = 5,
                          xi: float = 0.05, **kwargs) -> Dict:
        """Perform OPTICS clustering."""
        try:
            optics = OPTICS(min_samples=min_samples, xi=xi, **kwargs)
            labels = optics.fit_predict(X)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            results = {
                'algorithm': 'optics', 'min_samples': min_samples,
                'labels': labels, 'n_clusters': n_clusters, 'n_noise': n_noise,
                'model': optics
            }
            self.cluster_models['optics'] = optics
            self.cluster_labels['optics'] = labels
            return results
        except Exception as e:
            return {'error': str(e)}

    @step('Mean Shift Clustering')
    def mean_shift_clustering(self, X: pd.DataFrame, bandwidth: Optional[float] = None,
                              **kwargs) -> Dict:
        """Perform Mean Shift clustering."""
        try:
            meanshift = MeanShift(bandwidth=bandwidth, **kwargs)
            labels = meanshift.fit_predict(X)
            n_clusters = len(set(labels))
            results = {
                'algorithm': 'mean_shift', 'bandwidth': bandwidth,
                'labels': labels, 'n_clusters': n_clusters,
                'centers': meanshift.cluster_centers_, 'model': meanshift
            }
            self.cluster_models['mean_shift'] = meanshift
            self.cluster_labels['mean_shift'] = labels
            self.cluster_centers['mean_shift'] = meanshift.cluster_centers_
            return results
        except Exception as e:
            return {'error': str(e)}

    @step('Affinity Propagation')
    def affinity_propagation(self, X: pd.DataFrame, damping: float = 0.5,
                             **kwargs) -> Dict:
        """Perform Affinity Propagation clustering."""
        try:
            ap = AffinityPropagation(damping=damping, random_state=self.random_state, **kwargs)
            labels = ap.fit_predict(X)
            n_clusters = len(set(labels))
            results = {
                'algorithm': 'affinity_propagation', 'damping': damping,
                'labels': labels, 'n_clusters': n_clusters,
                'centers': ap.cluster_centers_indices_, 'model': ap
            }
            self.cluster_models['affinity_propagation'] = ap
            self.cluster_labels['affinity_propagation'] = labels
            return results
        except Exception as e:
            return {'error': str(e)}

    @step('GMM Clustering')
    def gmm_clustering(self, X: pd.DataFrame, n_clusters: int = 3,
                       covariance_type: str = 'full', **kwargs) -> Dict:
        """Perform Gaussian Mixture Model clustering (soft clustering)."""
        try:
            gmm = GaussianMixture(
                n_components=n_clusters, covariance_type=covariance_type,
                random_state=self.random_state, **kwargs
            )
            labels = gmm.fit_predict(X)
            probs = gmm.predict_proba(X)
            results = {
                'algorithm': 'gmm', 'n_clusters': n_clusters,
                'covariance_type': covariance_type, 'labels': labels,
                'probabilities': probs, 'model': gmm
            }
            self.cluster_models['gmm'] = gmm
            self.cluster_labels['gmm'] = labels
            return results
        except Exception as e:
            return {'error': str(e)}

    @step('K-Medoids Clustering')
    def kmedoids_clustering(self, X: pd.DataFrame, n_clusters: int = 3,
                            **kwargs) -> Dict:
        """Perform K-Medoids clustering (robust to outliers vs K-Means).

        Args:
            X: Feature matrix
            n_clusters: Number of clusters
        """
        if not KMEDOIDS_AVAILABLE:
            print("Warning: KMedoids not available. Install with: pip install scikit-learn-extra")
            return {'error': 'KMedoids not available'}
        try:
            kmedoids = KMedoids(
                n_clusters=n_clusters, random_state=self.random_state, **kwargs
            )
            labels = kmedoids.fit_predict(X)
            results = {
                'algorithm': 'kmedoids', 'n_clusters': n_clusters,
                'labels': labels, 'centers': kmedoids.cluster_centers_,
                'inertia': kmedoids.inertia_, 'model': kmedoids
            }
            self.cluster_models['kmedoids'] = kmedoids
            self.cluster_labels['kmedoids'] = labels
            self.cluster_centers['kmedoids'] = kmedoids.cluster_centers_
            return results
        except Exception as e:
            return {'error': str(e)}

    @step('Fuzzy C-Means')
    def fuzzy_cmeans(self, X: pd.DataFrame, n_clusters: int = 3,
                     m: float = 2.0, maxiter: int = 200, **kwargs) -> Dict:
        """Perform Fuzzy C-Means clustering (soft membership).

        Args:
            X: Feature matrix
            n_clusters: Number of clusters
            m: Fuzziness coefficient (default: 2.0)
            maxiter: Maximum iterations
        """
        if not FUZZY_AVAILABLE:
            print("Warning: Fuzzy C-Means not available. Install with: pip install scikit-fuzzy")
            return {'error': 'Fuzzy C-Means not available'}
        try:
            cntr, u, _, _, _, _, _ = fuzz.cmeans(
                X.T, n_clusters, m, error=0.005, maxiter=maxiter, **kwargs
            )
            labels = np.argmax(u, axis=0)
            results = {
                'algorithm': 'fuzzy_cmeans', 'n_clusters': n_clusters,
                'fuzziness': m, 'labels': labels,
                'membership': u, 'centers': cntr
            }
            self.cluster_models['fuzzy_cmeans'] = cntr
            self.cluster_labels['fuzzy_cmeans'] = labels
            self.cluster_centers['fuzzy_cmeans'] = cntr
            return results
        except Exception as e:
            return {'error': str(e)}

    # ── Evaluation ──────────────────────────────────────────────────

    @step('Evaluate Clustering')
    def evaluate_clustering(self, labels: np.ndarray, X: pd.DataFrame,
                            true_labels: Optional[np.ndarray] = None) -> Dict:
        """Evaluate clustering quality using multiple metrics.

        Args:
            labels: Cluster labels from algorithm
            X: Original feature matrix
            true_labels: Optional ground truth labels for supervised metrics

        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}

        # Internal metrics (no ground truth needed)
        n_unique = len(set(labels))
        if n_unique > 1 and n_unique < len(X):
            metrics['silhouette_score'] = float(silhouette_score(X, labels))
            metrics['calinski_harabasz_score'] = float(calinski_harabasz_score(X, labels))
            metrics['davies_bouldin_score'] = float(davies_bouldin_score(X, labels))

        # External metrics (need ground truth)
        if true_labels is not None:
            metrics['homogeneity_score'] = float(homogeneity_score(true_labels, labels))
            metrics['completeness_score'] = float(completeness_score(true_labels, labels))
            metrics['v_measure_score'] = float(v_measure_score(true_labels, labels))
            metrics['adjusted_rand_score'] = float(adjusted_rand_score(true_labels, labels))
            metrics['adjusted_mutual_info_score'] = float(adjusted_mutual_info_score(true_labels, labels))

        self.cluster_metrics = metrics
        return metrics

    # ── Compare algorithms ──────────────────────────────────────────

    @step('Compare Clustering Algorithms')
    def compare_algorithms(self, X: pd.DataFrame,
                           algorithms: Optional[List[str]] = None,
                           n_clusters_range: List[int] = [3, 5, 7],
                           **kwargs) -> Dict:
        """Compare multiple clustering algorithms on the same data.

        Args:
            X: Feature matrix
            algorithms: List of algorithms to compare (None = all available)
            n_clusters_range: Cluster numbers to try (for algorithms that need it)

        Returns:
            Dictionary of comparison results with best n_clusters per algorithm
        """
        if algorithms is None:
            algorithms = ['kmeans', 'mini_batch_kmeans', 'hierarchical', 'birch', 'gmm']

        results = {}
        needs_n_clusters = {'kmeans', 'mini_batch_kmeans', 'hierarchical', 'spectral', 'birch', 'gmm'}

        for algo in algorithms:
            try:
                if algo not in needs_n_clusters:
                    if algo == 'dbscan':
                        res = self.dbscan_clustering(X, **kwargs)
                    elif algo == 'hdbscan':
                        res = self.hdbscan_clustering(X, **kwargs)
                    else:
                        continue
                    if 'error' not in res:
                        metrics = self.evaluate_clustering(res['labels'], X)
                        results[algo] = {'metrics': metrics}
                    continue

                best_result = None
                best_score = -1
                for n_clusters in n_clusters_range:
                    if algo == 'kmeans':
                        res = self.kmeans_clustering(X, n_clusters=n_clusters, **kwargs)
                    elif algo == 'mini_batch_kmeans':
                        res = self.mini_batch_kmeans(X, n_clusters=n_clusters, **kwargs)
                    elif algo == 'hierarchical':
                        res = self.hierarchical_clustering(X, n_clusters=n_clusters, **kwargs)
                    elif algo == 'spectral':
                        res = self.spectral_clustering(X, n_clusters=n_clusters, **kwargs)
                    elif algo == 'birch':
                        res = self.birch_clustering(X, n_clusters=n_clusters, **kwargs)
                    elif algo == 'gmm':
                        res = self.gmm_clustering(X, n_clusters=n_clusters, **kwargs)
                    else:
                        continue

                    if 'error' not in res:
                        metrics = self.evaluate_clustering(res['labels'], X)
                        silhouette = metrics.get('silhouette', -1)
                        if silhouette > best_score:
                            best_score = silhouette
                            best_result = {'n_clusters': n_clusters, 'metrics': metrics}

                if best_result:
                    results[algo] = best_result
            except Exception as e:
                results[algo] = {'error': str(e)}

        return results

    # ── Visualizations ──────────────────────────────────────────────

    @step('Plot Clusters 2D')
    def plot_clusters_2d(self, X: pd.DataFrame, labels: np.ndarray,
                         title: str = 'Cluster Visualization',
                         save_path: Optional[str] = None) -> plt.Figure:
        """Plot clusters in 2D using PCA.

        Args:
            X: Feature matrix
            labels: Cluster labels
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        # Apply PCA for 2D visualization
        pca = PCA(n_components=2, random_state=self.random_state)
        X_pca = pca.fit_transform(X)

        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis',
                             alpha=0.7, edgecolors='k', s=50)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        plt.colorbar(scatter, ax=ax, label='Cluster')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    @step('Elbow Plot')
    def plot_elbow(self, X: pd.DataFrame, max_k: int = 10,
                   save_path: Optional[str] = None) -> plt.Figure:
        """Plot Elbow curve to find optimal K for K-Means.

        Args:
            X: Feature matrix
            max_k: Maximum K to try
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        inertias = []
        k_range = range(1, max_k + 1)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax.set_title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Clusters (K)')
        ax.set_ylabel('Inertia')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    @step('Silhouette Plot')
    def plot_silhouette(self, X: pd.DataFrame, n_clusters: int = 3,
                        save_path: Optional[str] = None) -> plt.Figure:
        """Plot Silhouette scores for K-Means clustering.

        Args:
            X: Feature matrix
            n_clusters: Number of clusters
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        from sklearn.metrics import silhouette_samples

        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        sil_samples = silhouette_samples(X, labels)

        fig, ax = plt.subplots(figsize=(10, 7))
        y_lower = 10
        for i in range(n_clusters):
            cluster_sil = sil_samples[labels == i]
            cluster_sil.sort()
            size = len(cluster_sil)
            y_upper = y_lower + size
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil, alpha=0.7)
            ax.text(-0.05, y_lower + 0.5 * size, str(i))
            y_lower = y_upper + 10

        ax.axvline(x=silhouette_score(X, labels), color='red', linestyle='--', linewidth=2)
        ax.set_title(f'Silhouette Plot (K={n_clusters})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Silhouette Coefficient')
        ax.set_ylabel('Cluster')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    @step('Find Optimal K')
    def find_optimal_k(self, X: pd.DataFrame, max_k: int = 10) -> Dict:
        """Find optimal number of clusters using Elbow + Silhouette.

        Args:
            X: Feature matrix
            max_k: Maximum K to try

        Returns:
            Dictionary with optimal K and per-K metrics
        """
        results = {}
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            sil = silhouette_score(X, labels) if len(set(labels)) > 1 else -1
            results[k] = {
                'inertia': kmeans.inertia_,
                'silhouette': float(sil) if sil != -1 else None
            }

        # Best K is the one with highest silhouette
        best_k = max(results, key=lambda k: results[k].get('silhouette', -1) or -1)
        return {'optimal_k': best_k, 'results': results}