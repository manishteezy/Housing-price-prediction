import pickle
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# define a Gaussain NB classifier
clf = RandomForestRegressor(
        n_estimators=50, max_depth=None, min_samples_leaf=1, min_samples_split=2,
        max_features=.7, max_samples=None, n_jobs=-1, random_state=42)

def preprocess_data(X):
    print(X.shape)
    cat_features = X.drop(columns=['Id']).select_dtypes(include='object').columns.tolist()
    num_features = X.drop(columns=['Id']).select_dtypes(include=np.number).columns.tolist()
    all_features = X.drop(columns=['Id']).columns.tolist()

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

# function to load the model
def load_model():
    global clf
    clf = pickle.load(open("models/seeds_nb.pkl", "rb"))


# function to predict the price using the model
def predict(data):
    x = preprocess_data(data)
    prediction = clf.predict([x])
    return prediction

