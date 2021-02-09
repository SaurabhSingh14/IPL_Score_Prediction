import pandas as pd
from joblib import dump
from sklearn.linear_model import LinearRegression

data = pd.read_csv('ipl.csv')
# print(data.head())

# TODO: Checking null values.
# print(data.isnull().sum())

# TODO: Converting dates into datetime object.
data['date'] = pd.to_datetime(data['date'])

# TODO: Keeping only above 5 overs data.
data = data[data['overs'] >= 5.0]

# TODO: Removing non-required columns
col = ['mid', 'venue', 'batsman', 'bowler', 'striker', 'non-striker']
data.drop(labels=col, axis=1, inplace=True)

# TODO: Just keeping current teams
current_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals', 'Mumbai Indians',
             'Kings XI Punjab', 'Royal Challengers Bangalore', 'Delhi Daredevils', 'Sunrisers Hyderabad']

data = data[(data['bat_team'].isin(current_teams)) & (data['bowl_team'].isin(current_teams))]

# TODO: One hot encoding Method
data = pd.get_dummies(data=data, columns=['bat_team', 'bowl_team'])

# TODO: Splitting the time Series dataset into train and test dataset.
xtrain = data.drop(labels='total', axis=1)[data['date'].dt.year <= 2016]
xtest = data.drop(labels='total', axis=1)[data['date'].dt.year > 2016]

ytrain = data[data['date'].dt.year <= 2016]['total']
ytest = data[data['date'].dt.year > 2016]['total']

# TODO: Dropping the date column.
xtrain.drop(labels='date', axis=1, inplace=True)
xtest.drop(labels='date', axis=1, inplace=True)

# TODO: Predicting model using Random Forest Regression.--------------------------1

# TODO: Hyper-tuning the parameters.
# def hypertuning():
#     rf_regressor = RandomForestRegressor()
#     parameters = {
#         'max_depth': [3, 5, 8, 10, None],
#         'n_estimators': [10, 100, 200, 300, 400, 500],
#         'bootstrap': [True, False]
#     }
#     rf_model = RandomizedSearchCV(rf_regressor, param_distributions=parameters, n_jobs=-1, cv=5)
#     rf_model.fit(xtrain, ytrain)
#     print(rf_model.best_params_)            # {'n_estimators': 200, 'max_depth': 10, 'bootstrap': True}
#     print(rf_model.best_score_)             # 0.5629357412510111
#     return rf_model

# rf_model = hypertuning()
# ypredict = rf_model.predict(xtest)

# TODO: RMS value for Random Forest Regression.
# mse = mean_squared_error(ytest, ypredict)
# rmse = np.sqrt(mse)
# print(rmse)                                   # 17.110876839816328

# TODO: Predicting model using Linear Regression.--------------------------2

l_model = LinearRegression()
l_model.fit(xtrain, ytrain)
# ypredict = l_model.predict(xtest)

# TODO: RMS value for Linear Regression.
# mse = mean_squared_error(ytest, ypredict)
# rmse = np.sqrt(mse)
# print(rmse)                                     # 15.843229566732045

# TODO: Predicting model using Ridge Regression.--------------------------3

# TODO: Finding the best parameter values
# def bestalpha():
#     parameters = {'alpha': [1e-10, 1e-2, 0.1, 1, 10, 20, 30, 40, 50, 100]}
#     r_regressor = Ridge()
#     r_model = RandomizedSearchCV(r_regressor, parameters, cv=5)
#     r_model.fit(xtrain, ytrain)
#     print(r_model.best_params_)                       # {'alpha': 100}
#     print(r_model.best_score_)                        # 0.6144668285329716
#     return r_model

# r_model = Ridge(alpha=100)                            # After the best value of alpha is known.
# r_model.fit(xtrain, ytrain)                           # Training the data for the best value of alpha.
# r_model = bestalpha()
# ypredict = r_model.predict(xtest)

# TODO: RMS value for Ridge Regression.
# mse = mean_squared_error(ytest, ypredict)
# rmse = np.sqrt(mse)
# print(rmse)                                        # 15.845157344761251

# TODO: Predicting model using Lasso Regression.--------------------------4

# TODO: Finding the best parameter values
# def bestalpha():
#     parameters = {'alpha': [1e-10, 1e-2, 0.1, 1, 10, 20, 30, 40, 50, 100]}
#     lasso_regressor = Lasso()
#     lasso_model = RandomizedSearchCV(lasso_regressor, parameters, cv=5)
#     lasso_model.fit(xtrain, ytrain)
#     print(lasso_model.best_params_)                       # {'alpha': 1}
#     print(lasso_model.best_score_)                        # 0.6233659449284912
#     return lasso_model

# lasso_model = Lasso(alpha=1)
# lasso_model.fit(xtrain, ytrain)
# # lasso_model = bestalpha()
# ypredict = lasso_model.predict(xtest)

# TODO: RMS value for Lasso Regression.
# mse = mean_squared_error(ytest, ypredict)
# rmse = np.sqrt(mse)
# print(rmse)                                        # 16.19769683615759

# TODO: To dump the file
dump(l_model, 'IPL Score Prediction.joblib')