import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn import metrics
import mlflow
import mlflow.sklearn

df = pd.read_csv('boston_housing.csv')

X = df[['crim','zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','black','lstat']]
y = df['medv']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

alpha_values = [1, 0.1, 0.5, 0.01, 0.001, 0.0001, 0]

def eval_metrics(actual, pred):
	mae = metrics.mean_absolute_error(actual,pred)
	rmse = np.sqrt(metrics.mean_squared_error(actual,pred))
	r2 = metrics.r2_score(actual,pred)
	return {'mae':mae, 'rmse': rmse, 'r2': r2}

for alpha in alpha_values:
	with mlflow.start_run():     
		model = Ridge(alpha)
		model.fit(X_train, y_train)
		y_pred = model.predict(X_test)
		
		mlflow.log_param('alpha', alpha)
		
		# get all the metrics
		all_mretics = eval_metrics(y_test, y_pred)
		for metric_name in all_mretics.keys():
			mlflow.log_metric(metric_name, all_mretics[metric_name])
			
		mlflow.sklearn.log_model(model, 'model')
		model_path = f"ridge/ridge-alpha({alpha})"
		mlflow.sklearn.save_model(model, model_path)

