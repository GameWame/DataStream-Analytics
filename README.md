# 📊 Analysis of Clustering Algorithms on Data Streams

Project for the **IoT Data Analytics** course at the University of Salerno.

This project provides a framework to analyze and compare the performance of different clustering algorithms on simulated data streams. It evaluates how traditional batch models and specialized streaming algorithms perform in both stable (stationary) and changing (concept drift) environments.

---

## 🔬 The Scenarios

You can run the experiment in two different modes to test the models under different conditions.

### 🧮 Stationary Scenario

This mode simulates a stable environment where the underlying data distribution does not change over time.

-   **Dataset:** Uses the **Digits dataset**, which is included in the `scikit-learn` library and requires no download.
-   **Goal:** Evaluate the accuracy and efficiency of the algorithms in a predictable and well-defined environment.

### 💨 Concept Drift Scenario

This mode simulates a dynamic environment where the data properties change over time, mimicking real-world situations like sensor degradation or shifting user behavior.

- **Dataset:** Uses the **Gas Sensor Array Drift** dataset, which must be provided as a `.zip` archive.
- **Goal:** Test the adaptability of the algorithms and their ability to handle changes.

---

## 🤖 The Models Under Test

The project compares four distinct clustering strategies:

1.  **K-Means:** A classic K-Means model trained only once on the initial data. It's fast but not adaptive.
2.  **K-Means (with Retraining):** A K-Means model that is periodically retrained on new data. 
3.  **CluStream:** A streaming algorithm designed for high-speed data, using micro-clusters to summarize the stream.
4.  **DenStream:** A density-based streaming algorithm that can find clusters of arbitrary shapes and is robust to noise.

---

## 🚀 Getting Started: A 3-Step Guide

Follow these steps to set up your environment and run the analysis.

### Step 1: Set Up Your Environment

It is highly recommended to use a Python virtual environment to keep dependencies isolated.

### Step 2: Install Dependencies

With the environment activated, run this single command to install all required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib river requests
```

### Step 3: Configure and Run

1.  **Prepare the Datasets:**
    -   **Stationary Scenario:** No preparation required. The script will automatically download the dataset.
    -   **Concept Drift Scenario:** Place the `.zip` archive containing the **Gas Sensor Array Drift** dataset in the project folder. **You must extract the archive** to create the `batch*.dat` files needed by the script.

2.  **Choose your scenario:** Open the `script.py` file and edit the `RUN_STATIONARY_SCENARIO` flag in the `ExperimentSettings` class.

3.  **Run the script** 

---

## 📈 Understanding the Output

After the script finishes, a summary chart will be displayed. This chart allows you to compare the models across three key performance indicators:

1.  **Adjusted Rand Index (ARI):** Measures the accuracy of the clustering, a score of 1.0 is a perfect match while 0.0 is no better than random.
2.  **Silhouette Score:** To measure the quality of the clusters without looking at the true labels. It assesses how dense and well-separated the clusters are, using scores that range from -1 to 1, with higher values being better.
3.  **Throughput (points/s):** To measure the efficiency and speed of the algorithm. 

## 💡 Troubleshotting

Here are some solutions for common issues you might encounter.

- If you're having trouble with computational times, you can reduce the "DATASET_FRACTION" variable.

- The online algorithms are highly sensitive to their parameters, if you switch to a new dataset, you must tune their hyperparameters in ExperimentSettings to match the new data's scale and density.




