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

df = pd.read_csv('data.csv', index_col=False)

X = df.iloc[:, 5:]
y = df.iloc[:, 0:5]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1)
models = [get_model(n) for n in range(1, 4)]

for name, nn in models:
    

    mses = []
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for train_index, test_index in kf.split(X):
    
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    

        nn_clone = sklearn.base.clone(nn)
        nn_clone.fit(X_train, y_train)
        y_pred = nn_clone.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mses.append(mse)
        
    avg_mse = sum(mses) / len(mses)
    print(name)
    print(nn)
    print(avg_mse)