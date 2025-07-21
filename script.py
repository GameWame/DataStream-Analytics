import pandas as pd
import numpy as np
import os
import time
import requests
import gzip
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from river.cluster import CluStream, DenStream
from river.preprocessing import StandardScaler


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ExperimentSettings:

    RUN_STATIONARY_SCENARIO: bool = False   # if true downloads covertype dataset, if false searches local batches (GasSensor dataset)
    DATASET_FRACTION: float = 1.0  # How much of the dataset are we using (testing), 1.0 = 100%
    NUM_BATCHES: int = 10 # number of batches to generate
    

    # Testing parameters

    CLUSTREAM_MICRO_CLUSTERS: int = 100
    DENSTREAM_EPSILON: float = 0.8
    KMEANS_N_INIT: int = 10
    DRIFT_DATA_PATH: str = "."
    DRIFT_N_FEATURES: int = 128
    DRIFT_N_CLUSTERS: int = 6
    KMEANS_RETRAIN_INTERVAL: int = 3


class DataHandler:
    
    def __init__(self, settings: ExperimentSettings):
        self.settings = settings

    def get_data_batches(self) -> (List[pd.DataFrame], int):
      
        if self.settings.RUN_STATIONARY_SCENARIO:
            logging.info("Loading data for Stationary Scenario (Covertype).")
            df = self._download_and_cache_covertype()
            n_clusters = df['target'].nunique()
            batches = self._create_batches_from_df(df)
        else:
            logging.info("Loading data for Concept Drift Scenario (GasSensor).")
            batches, n_clusters = self._load_drift_data()
        
        # Reproducibility check (testing)
        #if batches:
        #    checksum = batches[0]['feature_1'].sum()
        #    logging.info(f"Checksum for the first batch: {checksum:.4f}")

        return batches, n_clusters

    def _download_and_cache_covertype(self, cache_dir: str = ".") -> pd.DataFrame:
        """Downloading Covertype Dataset in cache."""
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
        csv_filename = os.path.join(cache_dir, "covtype.csv")

        if not os.path.exists(csv_filename):
            logging.info("Covertype dataset not found in cache. Starting download...")
            gz_filename = os.path.join(cache_dir, "covtype.data.gz")
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(gz_filename, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logging.info("Decompressing...")
                with gzip.open(gz_filename, 'rb') as f_in, open(csv_filename, 'wb') as f_out:
                    f_out.writelines(f_in)
                os.remove(gz_filename)
            except Exception as e:
                logging.error(f"Error while downloading or decompressing: {e}")
                raise

        logging.info(f"Loading dataset from: {csv_filename}")
        df = pd.read_csv(csv_filename, header=None)
        n_features = len(df.columns) - 1
        df.columns = [f'feature_{i+1}' for i in range(n_features)] + ['target']
        return df

    def _create_batches_from_df(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        
        frac = self.settings.DATASET_FRACTION
        if frac < 1.0:
            logging.warning(f"Using only a fraction of the dataset for testing: ({frac*100:.0f}%).")
            df = df.sample(frac=frac, random_state=42) 
        
        df_shuffled = df.reset_index(drop=True)
        batches = np.array_split(df_shuffled, self.settings.NUM_BATCHES)
        logging.info(f"Created {len(batches)} batches of approximately {len(batches[0])} samples each.")
        return batches

    def _load_drift_data(self) -> (List[pd.DataFrame], int):
        
        batches = []
        for i in range(1, self.settings.NUM_BATCHES + 1):
            filename = f'batch{i}.dat'
            file_path = os.path.join(self.settings.DRIFT_DATA_PATH, filename)
            try:
                df = self._parse_svmlight_to_df(file_path, self.settings.DRIFT_N_FEATURES)
                batches.append(df)
            except FileNotFoundError:
                logging.error(f"File {filename} not found.")
                raise
        return batches, self.settings.DRIFT_N_CLUSTERS

    def _parse_svmlight_to_df(self, filepath: str, n_features: int) -> pd.DataFrame:

        # Parser for svmlight files

        labels, rows_data = [], []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip().replace(';', ' ')
                if not line: continue
                parts = line.split()
                try:
                    labels.append(int(float(parts[0])))
                    row_features = [0.0] * n_features
                    for part in parts[1:]:
                        if ':' in part:
                            index, value = part.split(':')
                            feature_index = int(index) - 1
                            if 0 <= feature_index < n_features:
                                row_features[feature_index] = float(value)
                    rows_data.append(row_features)
                except (ValueError, IndexError):
                    continue
        X = np.array(rows_data, dtype=float)
        y = np.array(labels, dtype=int)
        df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(n_features)])
        df['target'] = y
        return df


class ModelRunner(ABC):
    
    def __init__(self, model_name: str, settings: ExperimentSettings):
        self.model_name = model_name
        self.settings = settings
        logging.info(f"Initializing runner for {self.model_name} model")

    @abstractmethod
    def run_and_evaluate(self, all_batches_df: List[pd.DataFrame], n_clusters: int) -> pd.DataFrame:
        pass
    
    def _calculate_metrics(self, X_scaled: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        ari = adjusted_rand_score(y_true, y_pred)
        
        if len(np.unique(y_pred)) > 1:
            sil_score = silhouette_score(X_scaled, y_pred)
        else:
            sil_score = 0.0  # If clustering fails 
            
        return {"ari_score": ari, "silhouette_score": sil_score}

class KMeansRunner(ModelRunner):
    
    def __init__(self, model_name: str, settings: ExperimentSettings, with_retraining: bool = False):
        super().__init__(model_name, settings)
        self.with_retraining = with_retraining

    def run_and_evaluate(self, all_batches_df: List[pd.DataFrame], n_clusters: int) -> pd.DataFrame:
        scaler = StandardScaler()
        kmeans_model = KMeans(
            n_clusters=n_clusters, 
            random_state=42, 
            n_init=self.settings.KMEANS_N_INIT
        )
        results = []

        for i, batch_df in enumerate(all_batches_df):
            batch_name = f"batch{i+1}"
            X_batch_raw = batch_df.drop('target', axis=1)
            y_true = batch_df['target'].values
            
            if i == 0 or (self.with_retraining and i % self.settings.KMEANS_RETRAIN_INTERVAL == 0):
                logging.info(f"Training/Re-training {self.model_name} on {batch_name}...")
                scaler.learn_many(X_batch_raw)
                X_train_scaled = scaler.transform_many(X_batch_raw)
                kmeans_model.fit(X_train_scaled)

            # Evaluation
            X_test_scaled = scaler.transform_many(X_batch_raw)
            start_time = time.time()
            y_pred = kmeans_model.predict(X_test_scaled)
            end_time = time.time()
            processing_time = end_time - start_time
            throughput = len(batch_df) / processing_time if processing_time > 0 else float('inf')
            
            metrics = self._calculate_metrics(X_test_scaled, y_true, y_pred)
            metrics["throughput_points_s"] = throughput
            metrics["batch"] = batch_name
            
            logging.info(f"Evaluation on {batch_name}: ARI={metrics['ari_score']:.4f}, Silhouette={metrics['silhouette_score']:.4f}")
            results.append(metrics)
            
        return pd.DataFrame(results)

class RiverModelRunner(ModelRunner):
 
    def __init__(self, model_name: str, model_instance, settings: ExperimentSettings):
        super().__init__(model_name, settings)
        self.model_instance = model_instance

    def run_and_evaluate(self, all_batches_df: List[pd.DataFrame], n_clusters: int) -> pd.DataFrame:
        scaler = StandardScaler()
        online_model = self.model_instance
        results = []

        for i, batch_df in enumerate(all_batches_df):
            batch_name = f"batch{i+1}"
            X_batch_raw = batch_df.drop('target', axis=1)
            y_true_batch = batch_df['target'].values
            
            y_pred_batch = []
            
            start_time = time.time()

            # In order to "simulate" a DataStream we send the data one by one. This may be done in mini-batches,
            # improving the computing times of online algorithms, but the result may not be realistic.

            for _, row in X_batch_raw.iterrows(): 
                point_dict = row.to_dict()
                scaler.learn_one(point_dict)
                point_scaled = scaler.transform_one(point_dict)
                
                try:
                    pred_label = online_model.predict_one(point_scaled)
                except Exception:
                    pred_label = -1
                y_pred_batch.append(pred_label)
                online_model.learn_one(point_scaled)
            end_time = time.time()

            processing_time = end_time - start_time
            throughput = len(X_batch_raw) / processing_time if processing_time > 0 else float('inf')
            X_batch_scaled = scaler.transform_many(X_batch_raw)
            metrics = self._calculate_metrics(X_batch_scaled, y_true_batch, np.array(y_pred_batch))
            metrics["throughput_points_s"] = throughput
            metrics["batch"] = batch_name
            
            logging.info(f"Evaluation on {batch_name}: ARI={metrics['ari_score']:.4f}, Silhouette={metrics['silhouette_score']:.4f}")
            results.append(metrics)

        return pd.DataFrame(results)


# Class used to create graphic representations
class ResultsVisualizer:

    def __init__(self, results_data: List[Dict[str, Any]], scenario_title: str):
        self.results_data = results_data
        self.scenario_title = scenario_title

    def plot_summary(self):
        fig, axes = plt.subplots(3, 1, figsize=(14, 20), sharex=True)
        fig.suptitle(f'Metrics Summary - {self.scenario_title}', fontsize=20, y=0.95)

        
        metrics_config = [
            {'key': 'ari_score', 'title': 'Adjusted Rand Index (ARI)', 'ax': axes[0], 'ylim': (-0.1, 1.0)},
            {'key': 'silhouette_score', 'title': 'Silhouette Score', 'ax': axes[1], 'ylim': (-1.0, 1.0)},
            {'key': 'throughput_points_s', 'title': 'Throughput (points/s)', 'ax': axes[2], 'ylim': None} # automatic scaling
        ]

        for config in metrics_config:

            metric_key = config['key']
            ax = config['ax']
           
            for result in self.results_data:

                df = result['data']
                if metric_key in df.columns and not df[metric_key].isnull().all():

                    ax.plot(df['batch'], df[metric_key], marker='o', linestyle='-', label=result['label'])
            
            ax.set_title(config['title'], fontsize=16)
            ax.set_ylabel('Score' if config['ylim'] else 'Points / second')
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.legend()
            
            if config['ylim']:
                ax.set_ylim(config['ylim'])
            else:
                ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
                ax.ticklabel_format(style='plain', axis='y')

        axes[-1].set_xlabel('Temporal Batch', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        plt.show()

class ExperimentManager:

    def __init__(self, settings: ExperimentSettings):
        self.settings = settings
        self.scenario_title = (
            "Stationary Scenario (Covertype Dataset)" 
            if settings.RUN_STATIONARY_SCENARIO 
            else "Concept Drift Scenario"
        )

   # To test a specific model, comment out the others.
    def run(self):

        logging.info(f"--- STARTING SIMULATION: {self.scenario_title.upper()} ---")

        data_handler = DataHandler(self.settings)
        batches, n_clusters = data_handler.get_data_batches()
        all_results = []

        # --- K-Means ---
        kmeans_static_runner = KMeansRunner("K-Means (Static)", self.settings, with_retraining=False)
        kmeans_static_results = kmeans_static_runner.run_and_evaluate(batches, n_clusters)
        all_results.append({'label': kmeans_static_runner.model_name, 'data': kmeans_static_results})

        # --- K-Means (with Retraining) ---
        kmeans_retrain_runner = KMeansRunner("K-Means (with Retraining)", self.settings, with_retraining=True)
        kmeans_retrain_results = kmeans_retrain_runner.run_and_evaluate(batches, n_clusters)
        all_results.append({'label': kmeans_retrain_runner.model_name, 'data': kmeans_retrain_results})

        # --- CluStream ---
        clustream_runner = RiverModelRunner(
            "CluStream", 
            CluStream(n_macro_clusters=n_clusters, max_micro_clusters=self.settings.CLUSTREAM_MICRO_CLUSTERS, seed=42),
            self.settings
        )
        clustream_results = clustream_runner.run_and_evaluate(batches, n_clusters)
        all_results.append({'label': clustream_runner.model_name, 'data': clustream_results})

        # --- DenStream ---
        denstream_runner = RiverModelRunner(
            "DenStream",
            DenStream(decaying_factor=0.01, beta=0.5, mu=3, epsilon=self.settings.DENSTREAM_EPSILON),
            self.settings
        )

        denstream_results = denstream_runner.run_and_evaluate(batches, n_clusters)
        all_results.append({'label': denstream_runner.model_name, 'data': denstream_results})

        logging.info("Generating the summary chart...")
        visualizer = ResultsVisualizer(all_results, self.scenario_title)

        visualizer.plot_summary()
        logging.info("--- SIMULATION COMPLETED ---")


if __name__ == "__main__":

    settings = ExperimentSettings()
    
    manager = ExperimentManager(settings)
    manager.run()
