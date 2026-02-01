import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import minkowski
import pandas as pd

#a1
def dot_product(A, B):
    result = 0
    for i in range(len(A)):
        result += A[i] * B[i] # for each element in A multiply it with B 
    return result #add to Result

def euclidean_norm(A):
    
    sum_sq = 0
    for i in range(len(A)):
        sum_sq += A[i] ** 2 # for each element in A square it and to sum
    return np.sqrt(sum_sq) # root the of the sum

#a2
def mean_vector(data):
    return np.sum(data, axis=0) / data.shape[0] # axis 0 means row wise sum then didvied by the total number of samples 

def variance_vector(data):
    mean = mean_vector(data)
    return np.sum((data - mean) ** 2, axis=0) / data.shape[0] # (data(i)-mean(i))^2 for each element then divided by total number of samples

def std_vector(data):
    return np.sqrt(variance_vector(data)) # square root of variance

#a4
def minkowski_distance(A, B, p):
    total = 0
    for i in range(len(A)):
        total += abs(A[i] - B[i]) ** p # take the mod value of the distance of each element of A and B then power p and add to total
    return total ** (1 / p) # calculates the pth root of the total

#a10
def custom_knn(train_X, train_y, test_point, k):
    distances = []
    for i in range(len(train_X)): #go through all training samples
        dist = euclidean_norm(train_X[i] - test_point) # calculate euclidean dist between training and the test point
        distances.append((dist, train_y[i])) # add it to list distances and the label of the training sample

    distances.sort(key=lambda x: x[0])# sort in ascending order 
    neighbors = distances[:k] # get the k nearest neighbors

    class_votes = {}
    for _, label in neighbors: # for each neighbour get the label
        class_votes[label] = class_votes.get(label, 0) + 1 #count the votes for each class label

    return max(class_votes, key=class_votes.get) # returns class with max votes

#confusion matrix

def confusion_matrix(true, pred): # true VS predicted labels 
    TP = TN = FP = FN = 0   #initialize
    for t, p in zip(true, pred): 
        if t == 1 and p == 1: TP += 1
        elif t == 0 and p == 0: TN += 1
        elif t == 0 and p == 1: FP += 1
        elif t == 1 and p == 0: FN += 1
    return TP, TN, FP, FN

def accuracy(TP, TN, FP, FN):
    return (TP + TN) / (TP + TN + FP + FN)

def precision(TP, FP):
    return TP / (TP + FP) if (TP + FP) != 0 else 0

def recall(TP, FN):
    return TP / (TP + FN) if (TP + FN) != 0 else 0

def f1_score(P, R):
    return 2 * P * R / (P + R) if (P + R) != 0 else 0

#main
#dataset
file_path = r"C:\Users\preet\Downloads\LAB3\Household_Profile_Bengaluru_ason_01-03-2011.csv"

df = pd.read_csv(file_path)

df_numeric = df.select_dtypes(include=[np.number])#takes the numeric data
df_numeric = df_numeric.dropna()

target_column = df_numeric.columns[0]
df_numeric["Class"] = (df_numeric[target_column] > df_numeric[target_column].median()).astype(int) # make data above median as 1 and below median as 0

X = df_numeric.drop(columns=["Class"]).values
y = df_numeric["Class"].values

#a1

A = X[0]
B = X[1]

print("Dot Product (Custom):", dot_product(A, B))
print("Dot Product (NumPy):", np.dot(A, B))

print("Euclidean Norm (Custom):", euclidean_norm(A))
print("Euclidean Norm (NumPy):", np.linalg.norm(A))

#a2

class1 = X[y == 0]
class2 = X[y == 1]

mean1 = mean_vector(class1)
mean2 = mean_vector(class2)

std1 = std_vector(class1)
std2 = std_vector(class2)

interclass_distance = euclidean_norm(mean1 - mean2) # tells how seperated the data is

print("\nInterclass Distance:", interclass_distance)

#a3 histogram

feature = X[:, 0]

plt.hist(feature, bins=10)
plt.title("Histogram of Selected Feature")
plt.xlabel("Feature Value")
plt.ylabel("Frequency")
plt.show()

print("Mean of Feature:", np.mean(feature))
print("Variance of Feature:", np.var(feature))

# A4minkowski distance

p_values = range(1, 11)
distances = [minkowski_distance(A, B, p) for p in p_values]

plt.plot(p_values, distances, marker='o')
plt.xlabel("p value")
plt.ylabel("Distance")
plt.title("Minkowski Distance vs p")
plt.show()

#a5
print("Custom Minkowski (p=3):", minkowski_distance(A, B, 3))
print("SciPy Minkowski (p=3):", minkowski(A, B, 3))

#a6 train test split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42)
#training data is 70% and testing is 30%
#a7
knn = KNeighborsClassifier(n_neighbors=3) #k=3
knn.fit(X_train, y_train)

#a8
print("\nkNN Accuracy:", knn.score(X_test, y_test))

#a9
predictions = knn.predict(X_test)
print("Sample Predictions:", predictions[:5]) #displays first 5 predictions


#a10
custom_predictions = []
for test_point in X_test:
    custom_predictions.append(custom_knn(X_train, y_train, test_point, 3))

#a11
k_values = range(1, 12)
accuracies = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    accuracies.append(model.score(X_test, y_test))

plt.plot(k_values, accuracies, marker='o')
plt.xlabel("k value")
plt.ylabel("Accuracy")
plt.title("Accuracy vs k")
plt.show()

#a12 13 confusion matrix
TP, TN, FP, FN = confusion_matrix(y_test, predictions)

acc = accuracy(TP, TN, FP, FN)
prec = precision(TP, FP)
rec = recall(TP, FN)
f1 = f1_score(prec, rec)

print("\nConfusion Matrix:")
print("TP:", TP, "TN:", TN, "FP:", FP, "FN:", FN)

print("\nPerformance Metrics:")
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)
