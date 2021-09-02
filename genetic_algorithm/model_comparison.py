from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold
import pickle
import sklearn
import pandas as pd
import numpy as np
np.random.seed(0)
def get_model(n):

    model = f"./models/model{n}/neural_model.pkl"
    with open(model, 'rb') as pkl_file:
        nn = pickle.load(pkl_file)
    return f"model{n}", nn

df = pd.read_csv('validacao.csv', index_col=False)

X = df.iloc[:, 5:]
y = df.iloc[:, 0:5]

models = [get_model(n) for n in range(1, 4)]

for name, nn in models:
    y_pred = nn.predict(X)
    mse = mean_squared_error(y, y_pred)
        
    print(name)
    print(nn)
    print(f'MSE = {mse:.5f}')