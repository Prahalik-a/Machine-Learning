import pandas as pd
import numpy as np

# Model selection & evaluation
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score

# Classification models
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Regression models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Clustering models
from sklearn.cluster import AgglomerativeClustering, DBSCAN


# ---------------------------------------------------------
# Function: Tune SVM using Randomized Search
# ---------------------------------------------------------
def tune_svm_model(X_train, y_train):
    
    # Trying a few combinations instead of all (faster)
    param_dist = {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"]
    }

    # Randomized search for best parameters
    search = RandomizedSearchCV(SVC(), param_dist, n_iter=2)
    search.fit(X_train, y_train)

    # Return the best version of SVM
    return search.best_estimator_


# ---------------------------------------------------------
# Function: Run multiple classification models
# ---------------------------------------------------------
def evaluate_classifiers(X_train, X_test, y_train, y_test, tuned_svm):

    # Different models to compare
    model_dict = {
        "SVM": tuned_svm,
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "Naive Bayes": GaussianNB(),
        "MLP": MLPClassifier(max_iter=200)
    }

    results_list = []

    # Train + test each model
    for model_name, clf in model_dict.items():

        clf.fit(X_train, y_train)

        # Predictions
        train_pred = clf.predict(X_train)
        test_pred = clf.predict(X_test)

        # Store metrics
        results_list.append([
            model_name,
            accuracy_score(y_train, train_pred),
            accuracy_score(y_test, test_pred),
            precision_score(y_test, test_pred, average='weighted'),
            recall_score(y_test, test_pred, average='weighted'),
            f1_score(y_test, test_pred, average='weighted')
        ])

    # Convert results into DataFrame
    return pd.DataFrame(results_list, columns=[
        "Model", "Train Accuracy", "Test Accuracy", "Precision", "Recall", "F1 Score"
    ])


# ---------------------------------------------------------
# Function: Run regression models
# ---------------------------------------------------------
def evaluate_regressors(X_train, X_test, y_train, y_test):

    reg_models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor()
    }

    reg_results = []

    for name, reg in reg_models.items():

        reg.fit(X_train, y_train)

        # Predict values
        predictions = reg.predict(X_test)

        # Calculate errors
        mse_val = mean_squared_error(y_test, predictions)
        r2_val = r2_score(y_test, predictions)

        reg_results.append([name, mse_val, r2_val])

    return pd.DataFrame(reg_results, columns=["Model", "MSE", "R2 Score"])


# ---------------------------------------------------------
# Function: Perform clustering
# ---------------------------------------------------------
def perform_clustering(X):

    # Hierarchical clustering
    agg_model = AgglomerativeClustering(n_clusters=3)
    agg_result = agg_model.fit_predict(X)

    # Density-based clustering
    db_model = DBSCAN(eps=0.5, min_samples=5)
    db_result = db_model.fit_predict(X)

    return agg_result, db_result


# ---------------------------------------------------------
# Main execution block
# ---------------------------------------------------------
def main():

    # Step 1: Load dataset
    data = pd.read_csv(r"C:\Users\sahil\Downloads\final_all_data\Water_Consumption_Data_July_2025_0.csv")

    # Just making sure column names are clean (sometimes spaces cause issues)
    data.columns = data.columns.str.strip()

    print("Available Columns:", data.columns.tolist())

    # Removing columns that are not useful for ML
    data = data.drop(columns=["Ward NO", "Ward Name"], errors='ignore')

    # Filling missing values (simple approach)
    data = data.fillna(0)

    target = "Consumption in ML"

    # Creating classification labels (low, medium, high)
    y_class = pd.qcut(data[target], q=3, labels=[0, 1, 2])

    # Regression target (actual values)
    y_reg = data[target]

    # Feature matrix
    X = data.drop(columns=[target]).values

    # ----------------------------
    # Random Search (SVM tuning)
    # ----------------------------
    print("\nRunning SVM Hyperparameter Tuning...")

    X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2)

    best_svm_model = tune_svm_model(X_train, y_train)

    # ----------------------------
    # Classification
    # ----------------------------
    print("\nRunning Classification Models...")

    clf_results = evaluate_classifiers(X_train, X_test, y_train, y_test, best_svm_model)

    print("\n=== Classification Results ===")
    print(clf_results)

    # ----------------------------
    # Regression
    # ----------------------------
    print("\nRunning Regression Models...")

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2)

    reg_results = evaluate_regressors(X_train_r, X_test_r, y_train_r, y_test_r)

    print("\n=== Regression Results ===")
    print(reg_results)

    # ----------------------------
    # Clustering
    # ----------------------------
    print("\nRunning Clustering Models...")

    agg_clusters, db_clusters = perform_clustering(X)

    print("\n=== Clustering Output ===")
    print("Agglomerative Clusters:", np.unique(agg_clusters))
    print("DBSCAN Clusters:", np.unique(db_clusters))


# Run the program
if __name__ == "__main__":
    main()