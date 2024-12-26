from joblib import dump
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

x, y = make_regression(n_samples=100, n_features=1, noise=0.1)
model = LinearRegression()
model.fit(x, y)

dump(model, "model.pkl")