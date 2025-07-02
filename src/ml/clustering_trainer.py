"""
Clustering trainer for unsupervised learning tasks
"""

import asyncio
import pickle
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import optuna
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import structlog

from ..core.config import settings
from ..models.schema import ClusteringResult
from typing import Optional

logger = structlog.get_logger()


class ClusteringTrainer:
    """Clustering trainer for unsupervised learning"""
    
    def __init__(self, run_id: str, session_id: str = None, websocket_manager=None):
        self.run_id = run_id
        self.session_id = session_id
        self.websocket_manager = websocket_manager
        self.start_time = time.time()
        self.max_time_seconds = 600  # 10 minutes max
    
    async def train_clustering_models(
        self,
        dataset_path: str,
        constraints: Dict[str, Any],
        selected_algorithms: List[str] = None
    ) -> ClusteringResult:
        """Train multiple clustering algorithms and return the best one"""
        
        logger.info("Starting clustering training", 
                   run_id=self.run_id, 
                   session_id=self.session_id,
                   selected_algorithms=selected_algorithms)
        
        # Load and prepare data
        df = pd.read_csv(dataset_path)
        X = self._prepare_data(df, constraints)
        
        # Send initial progress
        if self.websocket_manager and self.session_id:
            await self.websocket_manager.broadcast_training_status(self.session_id, {
                "event": "clustering_started",
                "message": "Preparing data for clustering",
                "progress": 10
            })
        
        # Define algorithms to train
        algorithms = selected_algorithms or ["kmeans", "dbscan", "hierarchical"]
        
        # Train algorithms
        results = []
        for i, algorithm in enumerate(algorithms):
            try:
                logger.info(f"Training {algorithm} clustering", algorithm=algorithm)
                
                result = await self._train_algorithm(algorithm, X, constraints)
                if result:
                    results.append(result)
                
                # Send progress update
                progress = 20 + int(70 * (i + 1) / len(algorithms))
                if self.websocket_manager and self.session_id:
                    await self.websocket_manager.broadcast_training_status(self.session_id, {
                        "event": "algorithm_completed",
                        "message": f"Completed {algorithm} clustering",
                        "algorithm": algorithm,
                        "progress": progress
                    })
                
            except Exception as e:
                logger.error(f"Failed to train {algorithm}", error=str(e))
                continue
        
        if not results:
            raise ValueError("All clustering algorithms failed")
        
        # Select best result based on silhouette score
        best_result = max(results, key=lambda x: x.silhouette_score)
        
        logger.info("Clustering completed", 
                   best_algorithm=best_result.algorithm,
                   silhouette_score=best_result.silhouette_score,
                   n_clusters=best_result.n_clusters)
        
        # Send completion notification
        if self.websocket_manager and self.session_id:
            await self.websocket_manager.broadcast_training_complete(self.session_id, {
                "clustering_results": True,
                "run_id": self.run_id,
                "best_algorithm": best_result.algorithm,
                "silhouette_score": best_result.silhouette_score,
                "n_clusters": best_result.n_clusters,
                "model_path": best_result.model_path
            })
        
        return best_result
    
    def _prepare_data(self, df: pd.DataFrame, constraints: Dict[str, Any]) -> np.ndarray:
        """Prepare data for clustering"""
        
        # Drop excluded columns
        exclude_cols = constraints.get("exclude_cols", [])
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols]
        
        # Create preprocessing pipeline
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        transformers = []
        
        if numeric_features:
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numeric_transformer, numeric_features))
        
        if categorical_features:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('cat', categorical_transformer, categorical_features))
        
        if transformers:
            preprocessor = ColumnTransformer(transformers=transformers)
            X_processed = preprocessor.fit_transform(X)
        else:
            X_processed = X.values
        
        # Apply dimensionality reduction if too many features
        if X_processed.shape[1] > 50:
            logger.info("Applying PCA for dimensionality reduction", 
                       original_features=X_processed.shape[1])
            pca = PCA(n_components=min(50, X_processed.shape[0] - 1))
            X_processed = pca.fit_transform(X_processed)
            logger.info("PCA completed", final_features=X_processed.shape[1])
        
        return X_processed
    
    async def _train_algorithm(self, algorithm: str, X: np.ndarray, constraints: Dict[str, Any]) -> Optional[ClusteringResult]:
        """Train a specific clustering algorithm"""
        
        try:
            if algorithm == "kmeans":
                return await self._train_kmeans(X)
            elif algorithm == "dbscan":
                return await self._train_dbscan(X)
            elif algorithm == "hierarchical":
                return await self._train_hierarchical(X)
            elif algorithm == "gaussian_mixture":
                return await self._train_gaussian_mixture(X)
            elif algorithm == "spectral":
                return await self._train_spectral(X)
            else:
                logger.warning(f"Unknown clustering algorithm: {algorithm}")
                return None
                
        except Exception as e:
            logger.error(f"Training failed for {algorithm}", error=str(e))
            return None
    
    async def _train_kmeans(self, X: np.ndarray) -> ClusteringResult:
        """Train K-means with optimal k selection"""
        
        # Test different k values
        max_k = min(10, len(X) // 2)
        best_score = -1
        best_model = None
        best_k = 3
        
        for k in range(2, max_k + 1):
            try:
                model = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
                labels = model.fit_predict(X)
                
                if len(np.unique(labels)) > 1:  # Must have more than 1 cluster
                    score = silhouette_score(X, labels)
                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_k = k
            except Exception as e:
                logger.warning(f"K-means failed for k={k}", error=str(e))
                continue
        
        if best_model is None:
            raise ValueError("K-means failed for all k values")
        
        # Calculate additional metrics
        labels = best_model.labels_
        silhouette = silhouette_score(X, labels)
        calinski_harabasz = calinski_harabasz_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        
        # Save model
        model_path = await self._save_model(best_model, "kmeans")
        
        return ClusteringResult(
            run_id=self.run_id,
            algorithm="kmeans",
            model_path=model_path,
            n_clusters=best_k,
            silhouette_score=silhouette,
            calinski_harabasz_score=calinski_harabasz,
            davies_bouldin_score=davies_bouldin,
            labels=labels.tolist(),
            parameters={"n_clusters": best_k, "init": "k-means++", "n_init": 10}
        )
    
    async def _train_dbscan(self, X: np.ndarray) -> ClusteringResult:
        """Train DBSCAN with parameter optimization"""
        
        # Use Optuna to optimize DBSCAN parameters
        def objective(trial):
            eps = trial.suggest_float('eps', 0.1, 2.0)
            min_samples = trial.suggest_int('min_samples', 2, min(20, len(X) // 10))
            
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            if n_clusters < 2:
                return -1  # Bad clustering
            
            # Calculate silhouette score (excluding noise points)
            if len(set(labels)) > 1:
                mask = labels != -1
                if np.sum(mask) > 1:
                    return silhouette_score(X[mask], labels[mask])
            
            return -1
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20, timeout=60)
        
        if study.best_value <= 0:
            raise ValueError("DBSCAN failed to find good clustering")
        
        # Train final model with best parameters
        best_params = study.best_params
        model = DBSCAN(eps=best_params['eps'], min_samples=best_params['min_samples'])
        labels = model.fit_predict(X)
        
        # Calculate metrics
        mask = labels != -1
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        if np.sum(mask) > 1:
            silhouette = silhouette_score(X[mask], labels[mask])
            calinski_harabasz = calinski_harabasz_score(X[mask], labels[mask])
            davies_bouldin = davies_bouldin_score(X[mask], labels[mask])
        else:
            silhouette = -1
            calinski_harabasz = 0
            davies_bouldin = float('inf')
        
        # Save model
        model_path = await self._save_model(model, "dbscan")
        
        return ClusteringResult(
            run_id=self.run_id,
            algorithm="dbscan",
            model_path=model_path,
            n_clusters=n_clusters,
            silhouette_score=silhouette,
            calinski_harabasz_score=calinski_harabasz,
            davies_bouldin_score=davies_bouldin,
            labels=labels.tolist(),
            parameters=best_params
        )
    
    async def _train_hierarchical(self, X: np.ndarray) -> ClusteringResult:
        """Train hierarchical clustering"""
        
        # Test different numbers of clusters
        max_clusters = min(10, len(X) // 2)
        best_score = -1
        best_model = None
        best_n_clusters = 3
        
        for n_clusters in range(2, max_clusters + 1):
            try:
                model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
                labels = model.fit_predict(X)
                
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_n_clusters = n_clusters
            except Exception as e:
                logger.warning(f"Hierarchical clustering failed for n_clusters={n_clusters}", error=str(e))
                continue
        
        if best_model is None:
            raise ValueError("Hierarchical clustering failed for all cluster numbers")
        
        # Calculate metrics
        labels = best_model.labels_
        silhouette = silhouette_score(X, labels)
        calinski_harabasz = calinski_harabasz_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        
        # Save model
        model_path = await self._save_model(best_model, "hierarchical")
        
        return ClusteringResult(
            run_id=self.run_id,
            algorithm="hierarchical",
            model_path=model_path,
            n_clusters=best_n_clusters,
            silhouette_score=silhouette,
            calinski_harabasz_score=calinski_harabasz,
            davies_bouldin_score=davies_bouldin,
            labels=labels.tolist(),
            parameters={"n_clusters": best_n_clusters, "linkage": "ward"}
        )
    
    async def _train_gaussian_mixture(self, X: np.ndarray) -> ClusteringResult:
        """Train Gaussian Mixture Model"""
        
        # Test different numbers of components
        max_components = min(10, len(X) // 2)
        best_score = -1
        best_model = None
        best_n_components = 3
        
        for n_components in range(2, max_components + 1):
            try:
                model = GaussianMixture(n_components=n_components, random_state=42)
                model.fit(X)
                labels = model.predict(X)
                
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_n_components = n_components
            except Exception as e:
                logger.warning(f"GMM failed for n_components={n_components}", error=str(e))
                continue
        
        if best_model is None:
            raise ValueError("Gaussian Mixture Model failed for all component numbers")
        
        # Calculate metrics
        labels = best_model.predict(X)
        silhouette = silhouette_score(X, labels)
        calinski_harabasz = calinski_harabasz_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        
        # Save model
        model_path = await self._save_model(best_model, "gaussian_mixture")
        
        return ClusteringResult(
            run_id=self.run_id,
            algorithm="gaussian_mixture",
            model_path=model_path,
            n_clusters=best_n_components,
            silhouette_score=silhouette,
            calinski_harabasz_score=calinski_harabasz,
            davies_bouldin_score=davies_bouldin,
            labels=labels.tolist(),
            parameters={"n_components": best_n_components, "covariance_type": "full"}
        )
    
    async def _train_spectral(self, X: np.ndarray) -> ClusteringResult:
        """Train Spectral clustering"""
        
        # Test different numbers of clusters
        max_clusters = min(8, len(X) // 2)  # Spectral clustering is more expensive
        best_score = -1
        best_model = None
        best_n_clusters = 3
        
        for n_clusters in range(2, max_clusters + 1):
            try:
                model = SpectralClustering(n_clusters=n_clusters, random_state=42)
                labels = model.fit_predict(X)
                
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_n_clusters = n_clusters
            except Exception as e:
                logger.warning(f"Spectral clustering failed for n_clusters={n_clusters}", error=str(e))
                continue
        
        if best_model is None:
            raise ValueError("Spectral clustering failed for all cluster numbers")
        
        # Calculate metrics
        labels = best_model.labels_
        silhouette = silhouette_score(X, labels)
        calinski_harabasz = calinski_harabasz_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        
        # Save model
        model_path = await self._save_model(best_model, "spectral")
        
        return ClusteringResult(
            run_id=self.run_id,
            algorithm="spectral",
            model_path=model_path,
            n_clusters=best_n_clusters,
            silhouette_score=silhouette,
            calinski_harabasz_score=calinski_harabasz,
            davies_bouldin_score=davies_bouldin,
            labels=labels.tolist(),
            parameters={"n_clusters": best_n_clusters, "gamma": 1.0}
        )
    
    async def _save_model(self, model, algorithm: str) -> str:
        """Save the trained model"""
        
        model_dir = Path(settings.MODELS_DIR) / self.run_id
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{algorithm}_model.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        return str(model_path) 