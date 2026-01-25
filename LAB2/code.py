import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cosine

#A1

def load_purchase_data(filepath):
    df = pd.read_excel(filepath, sheet_name="Purchase data") 
    X = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values #Matrix X
    y = df["Payment (Rs)"].values.reshape(-1, 1) #Matrix Y
    return X, y

def compute_rank(X):
    return np.linalg.matrix_rank(X) #Rank of Matrix X

def compute_cost_pseudoinverse(X, y):
    return np.linalg.pinv(X) @ y #cost of each product available for sale

#A2

def label_customers(y):
    return np.where(y.flatten() > 200, 1, 0) #rich

def train_classifier(X, labels):
    model = LogisticRegression()
    model.fit(X, labels)
    return model

#A3

def mean_numpy(data):
    return np.mean(data) #mean

def var_numpy(data): 
    return np.var(data) #variance

def mean_manual(data):
    return sum(data) / len(data) #my mean

def var_manual(data):
    mu = mean_manual(data)
    return sum((x - mu) ** 2 for x in data) / len(data) #my variance = sigma^2/len(data)

def avg_execution_time(func, data, runs=10):
    times = []
    for _ in range(runs):
        start = time.time()
        func(data)
        times.append(time.time() - start)
    return sum(times) / runs

#A5

def jaccard_coefficient(v1, v2): #JC = (f11/f01+f10+f11)
    f11 = np.sum((v1 == 1) & (v2 == 1))
    f01 = np.sum((v1 == 0) & (v2 == 1))
    f10 = np.sum((v1 == 1) & (v2 == 0))
    return f11 / (f01 + f10 + f11) if (f01 + f10 + f11) != 0 else 0

def simple_matching_coefficient(v1, v2): #SMC = (f11 + f00) / (f00 +f01 + f10 + f11)
    f11 = np.sum((v1 == 1) & (v2 == 1))
    f00 = np.sum((v1 == 0) & (v2 == 0))
    f01 = np.sum((v1 == 0) & (v2 == 1))
    f10 = np.sum((v1 == 1) & (v2 == 0))
    denom = f00 + f01 + f10 + f11
    return (f11 + f00) / denom if denom != 0 else 0


#A6

def cosine_similarity(v1, v2): #v1 and v2 are my two document vectors
    return 1 - cosine(v1, v2)  #we subtracted because normal cosine is not possible using scipy
                               #so we did this instead lower value means more similar and 1 is completely different

#A7

def similarity_matrices(binary_data, numeric_data): #two matrices here both only first 20 values are taken
    n = len(binary_data) #only 20 values since we passed it in the main file head(20)
    jc = np.zeros((n, n))
    smc = np.zeros((n, n))
    cos = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            jc[i, j] = jaccard_coefficient(binary_data[i], binary_data[j])
            smc[i, j] = simple_matching_coefficient(binary_data[i], binary_data[j])
            cos[i, j] = cosine_similarity(numeric_data[i], numeric_data[j])

    return jc, smc, cos

#A8 

def impute_data(df):
    for col in df.columns:
        if df[col].dtype == "object": #chooses categorical or string data
            df[col] = df[col].fillna(df[col].mode()[0]) #calculates the categorical value mode
        elif pd.api.types.is_numeric_dtype(df[col]):
            if df[col].skew() < 1:
                df[col] = df[col].fillna(df[col].mean()) #if numeric value with no outliers then find mean
            else:
                df[col] = df[col].fillna(df[col].median()) #if numeric value with outliers find median
        else:
            df[col] = df[col].fillna(df[col].mode()[0]) #fallback for other types
    return df

#A9

def normalize_data(df):
    scaler = MinMaxScaler() #this initializes the formula xnew = x - xmin / xmax - xmin
    num_cols = df.select_dtypes(include=np.number).columns #only numeric values normalized
    df[num_cols] = scaler.fit_transform(df[num_cols]) # fit() finds min and max for each coloumn and transform() rescales each value to either 0 or 1
    return df #after normalization min value = 0 , max value = 1

#MAIN

def main():
    filepath = r"D:\Documents\ML IV SEM\LAB 2\Lab Session Data.xlsx"

    # A1
    X, y = load_purchase_data(filepath)
    print("Dimensionality:", X.shape[1])
    print("Number of vectors:", X.shape[0])
    print("Rank of feature matrix:", compute_rank(X))
    print("Product costs (Candies, Mangoes, Milk):",
          compute_cost_pseudoinverse(X, y).flatten())

    # A2
    labels = label_customers(y)
    train_classifier(X, labels)
    print("Classifier trained for RICH / POOR classification")

    # A3
    stock = pd.read_excel(filepath, sheet_name="IRCTC Stock Price")

    stock["Date"] = pd.to_datetime(stock["Date"]) #converted it to datetime
    stock["Day"] = stock["Day"].astype(str).str.strip().str.lower()

    prices = stock.iloc[:, 3].dropna().values

    print("Mean (NumPy):", mean_numpy(prices)) #prints mean
    print("Variance (NumPy):", var_numpy(prices))#prints var
    print("Mean (Manual):", mean_manual(prices))#my mean
    print("Variance (Manual):", var_manual(prices))#my var

    print("Mean time NumPy:", avg_execution_time(mean_numpy, prices))
    print("Mean time Manual:", avg_execution_time(mean_manual, prices))

    wed_prices = stock[stock["Day"] == "wednesday"].iloc[:, 3] #selected just the wed prices
    apr_prices = stock[stock["Date"].dt.month == 4].iloc[:, 3] #selected only apr month prices

    print("Wednesday Sample Mean:", mean_numpy(wed_prices)) #compare with population mean
    print("April Sample Mean:", mean_numpy(apr_prices)) #compare with population mean

    loss_prob = np.mean(stock["Chg%"] < 0) #stock price below 0 loss
    print("Probability of loss:", loss_prob)

    profit_given_wed = np.mean(
    stock[stock["Day"] == "wednesday"]["Chg%"] > 0 #profit for wed
    )
    print("P(Profit | Wednesday):", profit_given_wed)

    plt.scatter(stock["Day"], stock["Chg%"])
    plt.title("Chg% vs Day of Week") #simple scatter plot for profit day of week
    plt.show()

    # A4
    thyroid = pd.read_excel(filepath, sheet_name="thyroid0387_UCI") #loaded the excel
    print(thyroid.info()) #identified the datatype
    print(thyroid.describe()) #describe keyword will give missing values,outliers in data,numerical variables
    numeric_cols = thyroid.select_dtypes(include=np.number).columns
    print("\nA4: Mean and Variance of Numeric Attributes")
    for col in numeric_cols: #only for the numerical coliumns we found out the mean and the variance
        print(f"{col} -> Mean: {thyroid[col].mean()}, Variance: {thyroid[col].var()}")



    # A5–A7
    binary_cols = [c for c in thyroid.columns  #only looking at the binary values 
                   if thyroid[c].dropna().isin([0, 1]).all()]
    
    binary_data = thyroid[binary_cols].head(20).values #takes the first 20 values for my heatmap 

    numeric_cols = thyroid.select_dtypes(include=np.number).columns #calculating cosine using numerical featured vectors only
    numeric_data = thyroid[numeric_cols].head(20).values #first 20 values for my heatmap

    # Normalize numeric data before cosine similarity
    numeric_data = MinMaxScaler().fit_transform(numeric_data)


    jc, smc, cos = similarity_matrices(binary_data, numeric_data)

    sns.heatmap(jc) #employed heatmap for jaccard
    plt.title("Jaccard Coefficient")
    plt.show()

    sns.heatmap(smc) # employed heatmap for smc
    plt.title("Simple Matching Coefficient")
    plt.show()

    sns.heatmap(cos) #employed heatmap for cosine
    plt.title("Cosine Similarity")
    plt.show()

    # A8
    thyroid = impute_data(thyroid) 

    # A9
    thyroid = normalize_data(thyroid)
    print("Data imputation and normalization completed") #just prints normalized data completed 

# ---------------- DRIVER ----------------

if __name__ == "__main__":
    main()