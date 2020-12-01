"""
Supervised ML predictions for wind power generation
"""

__author__ = "Maitreya Venkataswamy"

import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge


def _prepare_data(data):
    """Takes the raw data from the database and prepares it for a sklearn workflow"""
    # Get the number of turbines
    n_turb = data[0]["lat"].size

    # Split the data into the prediction and learning sets
    data_learn = [d for d in data if not np.isnan(d["power"])]
    data_new = [d for d in data if np.isnan(d["power"])]

    # Get the timestamps for each data set
    t_learn = [d["time"] for d in data_learn]
    t_new = [d["time"] for d in data_new]

    # Initialize arrays for the prepared data
    X_learn = np.zeros((len(data_learn), 2 * n_turb))
    X_new = np.zeros((len(data_new), 2 * n_turb))
    y_learn = np.zeros(len(data_learn))

    # Populate the learning data arrays
    for i, d in enumerate(data_learn):
        # Compute the wind velocity
        vel = np.hypot(d["u"], d["v"])

        # Add the turbine data
        X_learn[i, :] = np.hstack((vel, d["T"]))
        y_learn[i] = d["power"]

    # Populate the prediction data arrays
    for i, d in enumerate(data_new):
        # Compute the wind velocity
        vel = np.hypot(d["u"], d["v"])

        # Add the turbine data
        X_new[i, :] = np.hstack((vel, d["T"]))

    # Return the prepared data
    return t_learn, t_new, X_learn, y_learn, X_new


def linear_model(data):
    """Fits a linear model with polynomial features to the turbine data and predicts future power"""
    # Prepare the data for the sklearn workflow
    t_learn, t_new, X_learn, y_learn, X_new = _prepare_data(data)


    # Split the training data into a trianing set and a test set
    X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(X_learn, y_learn, t_learn,
                                                                    test_size=1/6, shuffle=False)

    # Assemble the model pipeline
    model = Pipeline([("skb", SelectKBest(f_regression, k=200)),
                      ("pf", PolynomialFeatures(degree=2)),
                      ("ss", StandardScaler()),
                      ("ridge", Ridge(alpha=100, random_state=0))])

    # Fit the model
    model.fit(X_train, y_train)

    # Return the model predictions
    return t_test, model.predict(X_test), t_new, \
           model.predict(X_new) if X_new.shape[0] > 1 else None
