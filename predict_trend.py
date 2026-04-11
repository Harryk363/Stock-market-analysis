import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import ta

def predict_trend(hist):
    df = hist.copy()
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df['Open-Close'] = df['Open'] - df['Close']
    df['High-Low'] = df['High'] - df['Low']
    df['MA10'] = df['Close'].rolling(10).mean()
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df = df.dropna()

    features = ['Open-Close', 'High-Low', 'MA10', 'RSI']
    X = df[features]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred) * 100
    recall = recall_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred) * 100 

    latest_input = X.iloc[[-1]]
    trend = model.predict(latest_input)[0]

    return {
        "accuracy": round(accuracy, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1_score": round(f1, 2),
        "prediction": "UP" if trend == 1 else "DOWN"
    }


# import pandas as pd
# import numpy as np
# import xgboost as xgb
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
# import ta

# def predict_trend(hist):
#     df = hist.copy()
#     df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
#     df['Open-Close'] = df['Open'] - df['Close']
#     df['High-Low'] = df['High'] - df['Low']
#     df['MA10'] = df['Close'].rolling(10).mean()
#     df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
#     df = df.dropna()

#     features = ['Open-Close', 'High-Low', 'MA10', 'RSI']
#     X = df[features]
#     y = df['Target']

#     X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

#     base_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

#     # Hyperparameter grid for tuning
#     param_grid = {
#         'n_estimators': [50, 100, 200],
#         'max_depth': [3, 5, 7],
#         'learning_rate': [0.01, 0.05, 0.1],
#         'subsample': [0.6, 0.8, 1.0],
#         'colsample_bytree': [0.6, 0.8, 1.0]
#     }

#     # TimeSeries cross-validation
#     tscv = TimeSeriesSplit(n_splits=5)

#     grid = GridSearchCV(
#         estimator=base_model,
#         param_grid=param_grid,
#         cv=tscv,
#         scoring='f1',
#         n_jobs=1,
#         verbose=0
#     )

#     # Fit grid search on training data
#     grid.fit(X_train, y_train)

#     # Use the best estimator found
#     model = grid.best_estimator_

#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred) * 100
#     precision = precision_score(y_test, y_pred) * 100
#     recall = recall_score(y_test, y_pred) * 100
#     f1 = f1_score(y_test, y_pred) * 100 

#     latest_input = X.iloc[[-1]]
#     trend = model.predict(latest_input)[0]

#     return {
#         "accuracy": round(accuracy, 2),
#         "precision": round(precision, 2),
#         "recall": round(recall, 2),
#         "f1_score": round(f1, 2),
#         "prediction": "UP" if trend == 1 else "DOWN"
#     }


