from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

def cross_validation(X,y,model,params,cv=5,scoring='f1_macro',return_model = True):
	r""" This function can be used to evaluate performance
	of SVC at parameters params.
	Parameters
	----------
	X : ndarray
		Input array of samples, size (n_samples, n_features)
	y : ndarray
		Array corresponding to class index of each sample
	params : dict
		The parameters of SVC
	model : string
		Name of the model
	scoring : string, Default='f1_macro'
		Name of the metric to be used
	return_model : bool, Default = True
		Return the model or not
	Returns
	-------
	s : float
		scores
	m : object
		if return_model is true then returns model
	Example
	-------
	params = {'kernel':"rbf",'gamma':0.1,'C':1.0}
	model = "SVC"
	s,m = svm_utils.cross_validation(X,y,model,params,cv=5,scoring='f1_macro')
	print(np.mean(np.asarray(s)))
	"""
	if model == "SVC":
		m = svm.SVC(**params)

	s = cross_val_score(m, X, y, scoring=scoring, cv=cv, n_jobs=1)

	if return_model == True:
		return s,m
	else:
		return s


def hyper_parameter_tuning(X,y,model,params_common,params_gs,scoring='f1_macro',n_jobs=None):
	r""" This function can be used to perform a grid search over parameters
	denoted by "params" of "SVC" support vector machine.
	Parameters
	----------
	X : ndarray
		Input array of samples, size (n_samples, n_features)
	y : ndarray
		Array corresponding to class index of each sample
	model : string
		Name of the model to use
	params_common : dict
		The common parameters of the model
	params_gs : dict or list of dict
		The parameters to perform grid search on apart from the common 
		parameters params_common
	scoring : string, Default='f1_macro'
		Name of the metric to be used
	n_jobs : int, Default=None
		Number of jobs to run in parallel. None means 1. -1 means use all.
	Returns
	-------
	"""	
	if model == "SVC":
		m = svm.SVC(**params_common)

	clf = GridSearchCV(m, params_gs, scoring=scoring, refit=True, cv=5, n_jobs=n_jobs, return_train_score=False, verbose=3)
	clf = clf.fit(X,y)

	return clf 

# Versions
# 1. sklearn : 1.0.2
