import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Removing coloumns which more than 50% values are missing
def coloums_with_most_missing_values(train):
    missing = train.isnull().sum()
    missing = missing[missing > train.shape[0]/2]
    return missing

def preprocess_data(X):

    cat_features = X.drop(columns=['Id']).select_dtypes(include='object').columns.tolist()
    num_features = X.drop(columns=['Id']).select_dtypes(include=np.number).columns.tolist()
    all_features = cat_features + num_features

    # Pipeline for categorical features
    cat_tfms = Pipeline(steps=[
        ('cat_ordenc', ce.OrdinalEncoder(return_df=True, handle_unknown='value', handle_missing='value'))
    ])

    # Pipeline for numerical features
    num_tfms = Pipeline(steps=[
        ('num_imputer',  SimpleImputer(missing_values=np.nan, strategy='median'))
    ])

    features = ColumnTransformer(transformers=[
        ('cat_tfms', cat_tfms, cat_features),
        ('num_tfms', num_tfms, num_features)
    ], remainder='passthrough')

    X_train_tf = pd.DataFrame(features.fit_transform(X[all_features]), columns=all_features)

    return X_train_tf

# function to train and load the model during startup
def init_model():
    if not os.path.isfile("models/prices_nb.pkl"):
        clf = RandomForestRegressor(
            n_estimators=50, max_depth=None, min_samples_leaf=1, min_samples_split=2,
            max_features=.7, max_samples=None, n_jobs=-1, random_state=42)
        pickle.dump(clf, open("models/prices_nb.pkl", "wb"))


# function to train and save the model as part of the feedback loop
def train_model():
    # load the model
    clf = pickle.load(open("models/prices_nb.pkl", "rb"))

    df = pd.read_csv('./data/train.csv')
    X = df.drop('SalePrice', axis = 1)
    y = np.log(df['SalePrice'])
    missing_cols = coloums_with_most_missing_values(X).keys()
    X.drop(missing_cols, inplace=True, axis=1)
    X = preprocess_data(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3)
    clf.fit(X_train, y_train)
    rmse = mean_squared_error(y_test, clf.predict(X_test), squared=False)
    print("Model trained with RMSE: " , rmse)

    # save the model
    pickle.dump(clf, open("models/prices_nb.pkl", "wb"))
