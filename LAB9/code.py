import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, r2_score

import lime
import lime.lime_tabular


def load_data():
    file_path = r"D:\Documents\Machine-Learning\LAB9\Features_with_target.xlsm"

    df = pd.read_excel(file_path)

    df.columns = df.columns.str.strip()  # clean names
    df = df.select_dtypes(include=['number'])  # keep numeric only
    df = df.fillna(df.mean())  # handle missing

    print(df.columns)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return train_test_split(X, y, test_size=0.2, random_state=42)


def build_stacking_model():
    base_models = [
        ("rf", RandomForestRegressor(n_estimators=100)),
        ("svr", SVR()),
        ("lr", LinearRegression())
    ]

    final_model = LinearRegression()

    model = StackingRegressor(
        estimators=base_models,
        final_estimator=final_model
    )

    return model


def build_pipeline(model):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),  # scale features
        ("model", model)
    ])
    return pipeline


def train_model(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2


def explain_with_lime(model, X_train, X_test):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=X_train.columns.tolist(),
        mode="regression"
    )

    exp = explainer.explain_instance(
        data_row=X_test.iloc[0],  # first sample
        predict_fn=model.predict
    )

    return exp


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    model = build_stacking_model()
    pipeline = build_pipeline(model)

    trained_model = train_model(pipeline, X_train, y_train)

    mse, r2 = evaluate_model(trained_model, X_test, y_test)

    print("MSE:", mse)
    print("R2 Score:", r2)

    explanation = explain_with_lime(trained_model, X_train, X_test)

    explanation.save_to_file("lime_explanation.html")  # save result
    print("LIME explanation saved")