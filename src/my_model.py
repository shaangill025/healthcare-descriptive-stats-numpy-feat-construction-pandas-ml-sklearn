import utils
import pandas as pd
import numpy as np
import etl
import time
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn import feature_selection
from sklearn.model_selection import GridSearchCV
from sklearn .feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR
from sklearn.ensemble import GradientBoostingClassifier ,BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LarsCV, LassoCV, LogisticRegressionCV,ElasticNetCV, RidgeClassifierCV, OrthogonalMatchingPursuitCV, LassoLarsCV, MultiTaskElasticNetCV

#Note: You can reuse code that you wrote in etl.py and models.py and cross.py over here. It might help.
# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

'''
You may generate your own features over here.
Note that for the test data, all events are already filtered such that they fall in the observation window of their respective patients. Thus, if you were to generate features similar to those you constructed in code/etl.py for the test data, all you have to do is aggregate events for each patient.
IMPORTANT: Store your test data features in a file called "test_features.txt" where each line has the
patient_id followed by a space and the corresponding feature in sparse format.
Eg of a line:
60 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514
Here, 60 is the patient id and 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514 is the feature for the patient with id 60.

Save the file as "test_features.txt" and save it inside the folder deliverables

input:
output: X_train,Y_train,X_test
'''
def my_features():
	#TODO: complete this
	train_events, train_mortality, feature_map = read_from_csv('../data/train/')
	entire_feature_set = set(feature_map.idx.unique())
	test_events, _, _ = read_from_csv('../data/test/')
	train_events = process_training_data(train_events.iloc[:, :], train_mortality)
	train_features, test_features = process_features(train_events, test_events)
	patient_id_series = pd.Series(train_features.index, index=train_features.index)
	dead_ids_list = list(train_mortality.patient_id)
	train_labels = np.array([id in dead_ids_list for id in list(patient_id_series)])
	X_train = train_features
	Y_train = train_labels
	X_test = test_features.sort_index()
	test_features.index.name = 'patient_id'
	test_features_long = pd.melt(test_features.reset_index(), id_vars=['patient_id'])
	test_features_long.columns = ['patient_id', 'feature_id', 'feature_value']
	test_features_long = test_features_long.sort_values('patient_id')
	tuple_temp = test_features_long.groupby('patient_id').apply(lambda x: list(x.sort_values('feature_id').apply(lambda y:
																				(y.feature_id, y.feature_value), axis=1)))
	patient_features_dict = tuple_temp.to_dict()
	deliverable1 = open('../deliverables/test_features.txt', 'wb')
	for patient in sorted(patient_features_dict.keys()):
		deliverable1.write(bytes("{} {} \n".format(patient, utils.bag_to_svmlight(patient_features_dict[patient])),'UTF-8'))
	return X_train,Y_train,X_test


'''
You can use any model you wish.

input: X_train, Y_train, X_test
output: Y_pred
'''
def my_classifier_predictions(X_train,Y_train,X_test):
	#TODO: complete this
	model = train_model(X_train, Y_train)
	model_train_pred = model.predict_proba(X_train)
	model_test_pred = model.predict_proba(X_test)
	utils.generate_submission("../deliverables/test_features.txt", model.predict_proba(X_test)[:, 1])
	return model.predict(X_test).astype(int)

def read_from_csv(filepath):
	try:	
		events = pd.read_csv(filepath + 'events.csv', parse_dates=['timestamp'])
		events = events.sort_values('timestamp')
	except IOError:
		events = None
	
	try:	
		mortality = pd.read_csv(filepath + 'mortality_events.csv', parse_dates=['timestamp'])
		mortality = mortality.sort_values('timestamp')
	except IOError:
		mortality = None

	try:	
		feature_map = pd.read_csv(filepath + 'event_feature_map.csv')
	except IOError:
		events = None
	return events, mortality, feature_map

def process_training_data(train_events, train_mortality):
	indx_date = etl.calculate_index_date(train_events, train_mortality, '/tmp/')
	return etl.filter_events(train_events, indx_date, '/tmp/')

def process_features(train_df, test_df):
	train_desc = train_df.groupby('patient_id').event_description.apply(lambda x: x.str.cat(sep = ' '))
	test_desc = test_df.groupby('patient_id').event_description.apply(lambda x: x.str.cat(sep =' ')) 
	count_vectorizer = CountVectorizer(ngram_range=(1,2), min_df=0.2, max_df=0.75)
	train_events = count_vectorizer.fit_transform(train_desc)
	test_events = count_vectorizer.transform(test_desc) 
	return (pd.DataFrame(train_events.toarray(), index=train_desc.index), pd.DataFrame(test_events.toarray(), index=test_desc.index))

def train_model(X_train, Y_train):
	model_temp = Pipeline(steps=[('red',preprocessing.MinMaxScaler()),
							   ('model_temp', RandomForestClassifier())])
	paramters = dict(model_temp__n_estimators=np.arange(20, 181, 20), model_temp__min_samples_split=np.arange(5, 101, 40),
						model_temp__min_samples_leaf=np.arange(1,11,3))
	model_temp = GridSearchCV(model_temp, paramters, n_jobs=30, scoring='roc_auc', verbose=0, cv=5)
	model_temp.fit(X_train,Y_train)
	best_model_temp = model_temp.best_estimator_
	return best_model_temp

def main():
	X_train, Y_train, X_test = my_features()
	Y_pred = my_classifier_predictions(X_train,Y_train,X_test)
	utils.generate_submission("../deliverables/test_features.txt",Y_pred)
	#The above function will generate a csv file of (patient_id,predicted label) and will be saved as "my_predictions.csv" in the deliverables folder.

if __name__ == "__main__":
    main()

	