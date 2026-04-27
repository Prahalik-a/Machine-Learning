# ============================
# IMPORTS
# ============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

import shap
from lime.lime_tabular import LimeTabularExplainer


# ============================
# LOAD DATA
# ============================
def load_data(path):
    df = pd.read_csv(path)
    print(df.columns)
    # Drop non-useful columns
    if 'Ward Name' in df.columns:
        df = df.drop(columns=['Ward Name'])

    # Optional: drop Ward NO (categorical ID)
    if 'Ward NO' in df.columns:
        df = df.drop(columns=['Ward NO'])

    # Keep only numeric data
    df = df.select_dtypes(include=[np.number])

    target_column = 'Consumption in ML'

    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return X, y


# ============================
# A1: CORRELATION HEATMAP
# ============================
def plot_correlation(X):
    plt.figure(figsize=(8,6))
    sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
    plt.title("Feature Correlation Heatmap")
    plt.show()


# ============================
# PCA FUNCTION (A2, A3)
# ============================
def apply_pca(X, variance):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=variance)
    X_pca = pca.fit_transform(X_scaled)

    return X_pca


# ============================
# MODEL TRAINING
# ============================
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return model, r2, rmse, X_train, X_test


# ============================
# A4: SEQUENTIAL FEATURE SELECTION
# ============================
def sequential_fs(X, y, k_features):
    model = RandomForestRegressor()

    sfs = SequentialFeatureSelector(
        model,
        n_features_to_select=k_features,
        direction='forward'
    )

    X_new = sfs.fit_transform(X, y)
    return X_new


# ============================
# A5: SHAP
# ============================
def shap_explain(model, X_train):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    shap.summary_plot(shap_values, X_train)


# ============================
# A5: LIME
# ============================
def lime_explain(model, X_train, X_test):
    explainer = LimeTabularExplainer(
        training_data=X_train,
        mode='regression'
    )

    exp = explainer.explain_instance(
        X_test[0],
        model.predict
    )

    print(exp.as_list())


# ============================
# MAIN
# ============================
if __name__ == "__main__":

    file_path = "Household_Profile_Bengaluru_ason_01-03-2011.csv"

    X, y = load_data(file_path)

    # A1
    plot_correlation(X)

    # A2: PCA 99%
    X_pca_99 = apply_pca(X, 0.99)
    model_99, r2_99, rmse_99, Xtr, Xte = train_model(X_pca_99, y)

    # A3: PCA 95%
    X_pca_95 = apply_pca(X, 0.95)
    model_95, r2_95, rmse_95, _, _ = train_model(X_pca_95, y)

    # A4: Sequential Feature Selection
    k = min(2, X.shape[1] - 1)
    X_sfs = sequential_fs(X, y, k_features=k)
    model_sfs, r2_sfs, rmse_sfs, _, _ = train_model(X_sfs, y)

    # A5: Explainability
    shap_explain(model_99, Xtr)
    lime_explain(model_99, Xtr, Xte)

    # RESULTS
    print("----- RESULTS -----")
    print("PCA 99% -> R2:", r2_99, "RMSE:", rmse_99)
    print("PCA 95% -> R2:", r2_95, "RMSE:", rmse_95)
    print("SFS -> R2:", r2_sfs, "RMSE:", rmse_sfs)