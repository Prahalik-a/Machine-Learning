import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


# -------------------- LOAD DATA --------------------
def load_data(file_path):
    if not os.path.exists(file_path):
        print("❌ File not found at:", file_path)
        file_path = input("👉 Please paste full correct file path here: ")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError("File still not found. Check path carefully.")
    
    df = pd.read_csv(file_path)
    print("✅ File loaded successfully!")
    return df


# -------------------- LINEAR REGRESSION --------------------
def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_regression(model, X, y):
    y_pred = model.predict(X)
    
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    
    # Avoid division by zero in MAPE
    y_safe = np.where(y == 0, 1e-10, y)
    mape = np.mean(np.abs((y - y_pred) / y_safe)) * 100
    
    r2 = r2_score(y, y_pred)
    
    return round(mse, 3), round(rmse, 3), round(mape, 3), round(r2, 3)


# -------------------- KMEANS --------------------
def perform_kmeans(X, k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    return kmeans


def evaluate_clustering(X, labels):
    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)
    
    return round(sil, 3), round(ch, 3), round(db, 3)


def kmeans_multiple_k(X, k_range):
    sil_scores = []
    distortions = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        
        sil_scores.append(silhouette_score(X, kmeans.labels_))
        distortions.append(kmeans.inertia_)

    return sil_scores, distortions


# -------------------- MAIN --------------------
if __name__ == "__main__":

    # 👇 Default path (change if needed)
    file_path = r"C:\Users\vijiamd\Documents\Water_Consumption_Data_July_2025_0.csv"
    
    df = load_data(file_path)

    df.columns = df.columns.str.strip()

    target_column = "Consumption in ML"

    # Drop non-numeric columns safely
    df_numeric = df.select_dtypes(include=[np.number])

    if target_column not in df_numeric.columns:
        raise ValueError(f"❌ Target column '{target_column}' not found!")

    X = df_numeric.drop(columns=[target_column])
    y = df_numeric[target_column]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------- Single Feature --------
    single_feature = X_train.columns[0]

    model_single = train_linear_regression(
        X_train[[single_feature]], y_train
    )

    print("\n--- Single Feature Regression ---")
    print("Train:", evaluate_regression(model_single, X_train[[single_feature]], y_train))
    print("Test :", evaluate_regression(model_single, X_test[[single_feature]], y_test))

    # -------- Multi Feature --------
    model_multi = train_linear_regression(X_train, y_train)

    print("\n--- Multi Feature Regression ---")
    print("Train:", evaluate_regression(model_multi, X_train, y_train))
    print("Test :", evaluate_regression(model_multi, X_test, y_test))

    # -------- KMeans (k=2) --------
    X_cluster = X

    kmeans_2 = perform_kmeans(X_cluster, 2)

    sil, ch, db = evaluate_clustering(X_cluster, kmeans_2.labels_)

    print("\n--- KMeans (k=2) ---")
    print("Silhouette:", sil)
    print("Calinski-Harabasz:", ch)
    print("Davies-Bouldin:", db)

    # -------- Multiple K --------
    k_range = range(2, 10)

    sil_scores, distortions = kmeans_multiple_k(X_cluster, k_range)

    # Silhouette Plot
    plt.figure()
    plt.plot(k_range, sil_scores)
    plt.title("Silhouette Score vs K")
    plt.xlabel("K")
    plt.ylabel("Silhouette Score")
    plt.show()

    # Elbow Plot
    plt.figure()
    plt.plot(k_range, distortions)
    plt.title("Elbow Plot")
    plt.xlabel("K")
    plt.ylabel("Inertia")
    plt.show()