import numpy as np 
import pandas as pd
import sys
from sklearn import metrics

sys.path.append('../')
import base_paths

sys.path.append('../lib_models/support_vector_machine/')
import svm_utils

def read_features(path_features_file, columns):
	r'''Reads features from a file into NumPy arrays
	Parameters
	----------
	path_features_file : string
		Path of the .csv file containing features.
	columns : list of strings
		Each string denotes the name of a feature.
	Returns
	-------
		X : (M,N) ndarray
			Each row denotes a sample and each column denotes a feature.
		y : (M,) ndarray
			Each entry denotes a class to which sample belongs.
	'''

	for i in columns:

		df = pd.read_csv(path_features_file, header=0)
		df1 = df[columns]

		X = df1.to_numpy(dtype=np.float64)
		X = np.asarray(X)

		df2 = df['class']
		y = df2.to_numpy(dtype=np.float64)

	return X,y

def main():

	########## Reading features #######################################################################
	# Read features in a pandas data frame after doing EDA
	print('Reading features for Training and Testing')
	cols = ['mean','std','var','mean_abs_dev','median','median_abs_dev','coef_var','skewness',
	'kurtosis','maximum','minimum','mode','range','inter_quart_range','entropy']
	cols_new = []
	for i in cols:
		cols_new.append(i+'_intensity')
	X_train,y_train = read_features('features_train.csv', cols_new)
	X_test,y_test = read_features('features_test.csv', cols_new)

	################### Normalizing features ##################################################
	print(f'Normalizing')
	X = np.concatenate((X_train,X_test))
	means = np.mean(X,axis=0)
	std = np.std(X,axis=0)
	X_train = (X_train-means)/(std)
	X_test = (X_test-means)/(std)

	########## Training #######################################################################
	# Hyperparameter tuning
	import time
	a = time.time()
	print('Hyperparameter tuning')
	C=[0.1,1,10,100]
	gamma=[0.001,0.01,0.1,1,10]
	params_gs = [{'C':C,'gamma':gamma,'kernel':['rbf']}]	
	params_common = {'class_weight':None, 'decision_function_shape':'ovr'}
	model = "SVC"
	clf = svm_utils.hyper_parameter_tuning(X_train,y_train,model,params_common,params_gs,n_jobs=-1)
	r = clf.cv_results_
	best_model = clf.best_estimator_
	print(f'Current best score on training data is {clf.best_score_}')
	print(f'Current best param on training data is {clf.best_params_}')
	print(f'Time taken is {time.time()-a}')

	############# TESTING ######################################################################
	### Uncomment the following piece of code after finding the optimal parameter setting
	print(f'Prediction for Testing')

	import time
	a = time.time()
	print('Hyperparameter tuning')
	C = [100]
	gamma = [10]
	params_gs = [{'C':C,'gamma':gamma,'kernel':['rbf']}]	
	params_common = {'class_weight':None, 'decision_function_shape':'ovr'}
	model = "SVC"
	clf = svm_utils.hyper_parameter_tuning(X_train,y_train,model,params_common,params_gs,n_jobs=-1)
	best_model = clf.best_estimator_

	y_pred = best_model.predict(X_test)

	print(f'Final f1_score on testing data is {metrics.f1_score(y_test,y_pred)}')

if __name__ == '__main__':
	main()

# Versions
# 1. numpy : 1.21.2
# 2. skimage : 0.18.3
# 3. pandas : 1.3.5
# 4. sklearn : 1.0.2



