import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree



#A1 : Entropy Calculation


def calculate_entropy(labels):
   
   # Computes entropy of the target variable.
    unique_vals, counts = np.unique(labels, return_counts=True)

    probs = counts / counts.sum()

    ent = -np.sum(probs * np.log2(probs))

    return ent



#A2 : Gini Index Calculation


def calculate_gini(labels):
  # Computes the Gini Index which is another impurity measure.Lower value -> purer node

    unique_vals, counts = np.unique(labels, return_counts=True)

    probs = counts / counts.sum()

    gini_value = 1 - np.sum(probs ** 2)

    return gini_value


#A3 : Information Gain


def compute_information_gain(features, target, column_index):
   
# Calculates Information Gain for a particular feature.Used for deciding the root node of the tree.
   

    parent_entropy = calculate_entropy(target)

    unique_values = np.unique(features[:, column_index])

    weighted_entropy = 0

    for val in unique_values:

        subset_target = target[features[:, column_index] == val]

        weight = len(subset_target) / len(target)

        weighted_entropy += weight * calculate_entropy(subset_target)

    info_gain = parent_entropy - weighted_entropy

    return info_gain


def find_best_feature(features, target):
   
    #Checks information gain for all features and returns the feature index with highest gain
   

    gain_list = []

    for i in range(features.shape[1]):

        gain = compute_information_gain(features, target, i)

        gain_list.append(gain)

    best_index = np.argmax(gain_list)

    return best_index


#A4 : Equal Width Binning

def equal_width_bins(data_column, number_of_bins=10):
   
   # Converts continuous values into bins of equal width. 
  

    min_value = np.min(data_column)
    max_value = np.max(data_column)

    bin_width = (max_value - min_value) / number_of_bins

    bin_result = np.floor((data_column - min_value) / bin_width)

    return bin_result.astype(int)



#A5 : Simple Decision Tree Structure


def create_simple_tree(features, target, feature_names):
   
   # Creates a simple tree structure by identifying
   
    root_index = find_best_feature(features, target)

    tree_structure = {
        "root_index": root_index,
        "root_feature": feature_names[root_index]
    }

    return tree_structure



#A6 : Decision Tree Visualization


def plot_decision_tree(features, target, feature_names, class_labels):

    dt_model = DecisionTreeClassifier()

    dt_model.fit(features, target)

    plt.figure(figsize=(12, 8))

    plot_tree(
        dt_model,
        feature_names=feature_names,
        class_names=class_labels,
        filled=True
    )

    plt.title("Decision Tree Visualization")

    plt.show()



#A7 : Decision Boundary Visualization


def plot_decision_boundary(two_features, labels, feature_names):
   
    #Plots the decision boundary using two selected features
  

    dt_model = DecisionTreeClassifier()

    dt_model.fit(two_features, labels)

    x_min, x_max = two_features[:, 0].min() - 1, two_features[:, 0].max() + 1
    y_min, y_max = two_features[:, 1].min() - 1, two_features[:, 1].max() + 1

    grid_x, grid_y = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02)
    )

    predictions = dt_model.predict(np.c_[grid_x.ravel(), grid_y.ravel()])

    predictions = predictions.reshape(grid_x.shape)

    plt.contourf(grid_x, grid_y, predictions, alpha=0.3)

    plt.scatter(two_features[:, 0], two_features[:, 1], c=labels)

    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])

    plt.title("Decision Boundary using Decision Tree")

    plt.show()

#main


if __name__ == "__main__":

    # Load dataset
   

    dataset = pd.read_csv(
        r"D:\Users\vijiamd\Downloads\final_combined_dataset.csv"
    )

    # Remove Ward number because it is just an identifier
    dataset = dataset.drop(columns=["Ward NO"])

    # Replace missing values with 0
    dataset = dataset.fillna(0)

    target_col = "Consumption in ML"

    # Convert continuous target into 3 categories
    target = pd.qcut(dataset[target_col], q=3, labels=[0, 1, 2])

    # Feature matrix
    feature_matrix = dataset.drop(columns=[target_col]).values

    feature_names = dataset.drop(columns=[target_col]).columns

    class_names = ["Low", "Medium", "High"]

 
    #A1 : Entropy
 

    print("A1 - Entropy:", calculate_entropy(target))

    
    #A2 : Gini Index


    print("A2 - Gini Index:", calculate_gini(target))


    #A3 : Best Feature Selection
   

    best_index = find_best_feature(feature_matrix, target)

    print("A3 - Best Feature for Root:", feature_names[best_index])


    #A4 : Example of Binning
    

    binned_features = feature_matrix.copy()

    binned_features[:, 0] = equal_width_bins(feature_matrix[:, 0], number_of_bins=5)

    print("A4 - Equal width binning applied on feature 0")

    
    #A5 : Simple Decision Tree
   

    simple_tree = create_simple_tree(feature_matrix, target, feature_names)

    print("A5 - Simple Tree Root:", simple_tree)

  
    #A6 : Decision Tree Visualization
    

    plot_decision_tree(feature_matrix, target, feature_names, class_names)

   
    #A7 : Decision Boundary Visualization
   

    selected_features = feature_matrix[:, :2]

    plot_decision_boundary(selected_features, target, feature_names)