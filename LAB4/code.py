import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV



#A1
def compute_classification_metrics(y_true, y_pred): #computing our confusion matrix using our dataset
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred)#this directly checks for precision score
    recall = recall_score(y_true, y_pred)#this directly checks for recall score
    f1 = f1_score(y_true, y_pred) # this directly checks for f1 score
    return cm, precision, recall, f1

# A2
def compute_regression_metrics(actual, predicted): #already done in lab2
    mse = mean_squared_error(actual, predicted)#calculates the mean squared error
    rmse = np.sqrt(mse)#calculates the root mean square error
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100 #calculates the mean absolute percentage error
    r2 = r2_score(actual, predicted)# r squared
    return mse, rmse, mape, r2

#A3
def generate_training_data(num_points=20, low=1, high=10):
    X = np.random.uniform(low, high, (num_points, 2))
    y = np.array([0 if x[0] + x[1] < (low + high) else 1 for x in X])
    return X, y

def plot_training_data(X, y):
    colors = ['blue' if label == 0 else 'red' for label in y]
    plt.scatter(X[:, 0], X[:, 1], c=colors)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Training Data Scatter Plot (A3)")
    plt.show()

def generate_test_data(start=0, end=10, step=0.1):
    gx, gy = np.meshgrid(
        np.arange(start, end, step),
        np.arange(start, end, step)
    )
    return np.c_[gx.ravel(), gy.ravel()]

def plot_test_classification(X_test, preds, k):
    colors = ['blue' if label == 0 else 'red' for label in preds]
    plt.scatter(X_test[:, 0], X_test[:, 1], c=colors, s=2)
    plt.title(f"KNN Classification (k={k})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

X, y = generate_training_data()
test_data = generate_test_data()

for k in [1, 3, 7]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    preds = knn.predict(test_data)
    plot_test_classification(test_data, preds, k)



#DATA PREP  

def prepare_features_labels(df):
    numeric_df = df.select_dtypes(include=[np.number])

    # Ensure at least 3 numeric columns
    if numeric_df.shape[1] < 3:
        raise ValueError("Dataset does not have enough numeric columns")

    # Drop rows with missing values in required columns
    numeric_df = numeric_df.dropna(subset=numeric_df.columns[:3])

    X = numeric_df.iloc[:, :2].values
    target = numeric_df.iloc[:, 2].values

    median_val = np.median(target)
    y = np.array([1 if val >= median_val else 0 for val in target])

    return X, y

# A7 Hyperparameter Tuning 

param_grid = {
    'n_neighbors': list(range(1, 21))   # k from 1 to 20
}

knn = KNeighborsClassifier()

grid = GridSearchCV(
    knn,
    param_grid,
    cv=5,
    scoring='accuracy'
)

grid.fit(X, y)

best_k = grid.best_params_['n_neighbors']
best_score = grid.best_score_

print("Best k value:", best_k)
print("Best cross-validation accuracy:", best_score)


#MAIN 

if __name__ == "__main__":
    csv_path = r"C:\Users\vijiamd\Downloads\Unemployment_Rate_Bengaluru_ason_01-03-2011_1.csv" #our datasets

    df = pd.read_csv(csv_path)

    X, y = prepare_features_labels(df)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)
    predictions = knn.predict(X)

    cm, precision, recall, f1 = compute_classification_metrics(y, predictions)

    print("Confusion Matrix:\n", cm)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

# A2 – Price Prediction Evaluation (Lab 02)
actual_prices = np.array([200, 250, 300, 400, 500])
predicted_prices = np.array([210, 240, 310, 390, 480])

mse, rmse, mape, r2 = compute_regression_metrics(actual_prices, predicted_prices)

print("\nA2 – Regression Metrics")
print("MSE:", mse)
print("RMSE:", rmse)
print("MAPE:", mape)
print("R2 Score:", r2)


X_train, y_train = generate_training_data()
plot_training_data(X_train, y_train)

test_data = generate_test_data()
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
preds = knn.predict(test_data)

plot_test_classification(test_data, preds, 3)