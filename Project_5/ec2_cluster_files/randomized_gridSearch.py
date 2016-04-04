#!/usr/bin/python
#Author: Zach Farmer
#Purpose: Parallelize algorithm and feature selection with randomized grid search.
#Use numpy memarrays and Ipython Parallel, run on AWS EC2 using StarCluster.   
"""
    The following code blocks borrower heavily from Olivier Grisel
    "Advanced Machine Learning with scikit-learn" tutorial given at PyCon
    2013 in Santa Clara. The tutorial provided ipython notebooks with
    examples, I have used the code examples in those notebooks as a guidline
    for implementing my gridsearch optimization in parallel. For those 
    interested you can find the notebooks and link to the tutorial at the 
    following github: https://github.com/ogrisel/parallel_ml_tutorial/
"""
import sys
import numpy as np
import os
import time
import pickle
import re
import scipy.stats as sp

from pprint import pprint
from create_my_dataset import newFeatures, dropFeatures
from feature_format import featureFormat, targetFeatureSplit  


from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA 

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline, FeatureUnion

#from sklearn.externals import joblib
#from IPython.parallel import Client
from sklearn.grid_search import ParameterGrid
from sklearn.grid_search import RandomizedSearchCV

from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegressionCV


def PickleBestClassifers(best_classifiers, file_Name):
    """
        Input:
            A python dictionary containing the names of classifers as keys and the 
            pipeline object containing the optimized paramters for the feature selection
            and learning algorithms. The name that the pickled file will be saved
            under as a python string.
            
        Output:
            (None) pickled object saved to the local directory.
    """     
    # Pickle the results, as the gridSearch takes forever
    fileObject = open(file_Name,'wb') 
    
    pickle.dump(best_classifiers, fileObject)   
    
    fileObject.close()
    
    return None
  
if __name__ == "__main__":
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)
    
    np.random.seed(42)
    
    ## Create new features 
    my_dataset = newFeatures(data_dict) 
    
    ## Find unique features in my_dataset
    features = [value for value in my_dataset.itervalues() for value in value.keys()]
    unique_features = list(set(features))
    
    ## Remove non-numeric features (email_address)
    reduced_features = dropFeatures(unique_features, ['email_address'])

    ## Method for moving an item in a list to a new position found at:
    ## http://stackoverflow.com/questions/3173154/move-an-item-inside-a-list
    ## posted by nngeek
    ## ensure that 'poi' is the first value in the feature list
    try:
        reduced_features.remove('poi')
        reduced_features.insert(0, 'poi')
    except ValueError:
        pass

    ## Extract features and labels from dataset
    data = featureFormat(my_dataset, reduced_features, sort_keys=True)
    ## Return as numpy arrays    
    labels, numpy_features = targetFeatureSplit(data)
    
    ## Create training and test splits on all of the features, feature 
    ## selection to be performed in the pipeline 
    X_train, X_test, y_train, y_test = train_test_split(numpy_features,\
                                                        labels,\
                                                        test_size=0.2,\
                                                        random_state=42)
    
    if "Best_Classifiers.pkl" not in os.listdir('.'):    
        ## List of classifiers to explore and compare
        classifiers = {
                        "GNB": GaussianNB(), 
                        "SVC": svm.SVC(),
                        "RDF": RandomForestClassifier(),
                        "ADB": AdaBoostClassifier(DecisionTreeClassifier(class_weight='balanced')),
                        "LRC": LogisticRegressionCV()
                        }
       
        ## dictionary of parameters for the GridSearchCV 
        param_grid = dict(
                    features__pca__n_components = sp.randint(1,len(X_train[0])),
                    features__univ_select__percentile = range(1,100,10),
                    SVC__C = sp.expon(scale = 100),
                    SVC__gamma = sp.expon(scale=.1),
                    SVC__kernel = ['rbf', 'linear','sigmoid'],
                    SVC__class_weight = ['balanced'],
                    RDF__n_estimators = range(1,500,1),
                    RDF__criterion = ['gini','entropy'],
                    RDF__max_depth = range(1,len(X_train[0]),1),
                    RDF__class_weight = ['balanced'],
                    ADB__n_estimators = range(1,500,1),
                    ADB__learning_rate = sp.expon(scale= 300),
                    LRC__Cs = range(0,10,1),
                    LRC__class_weight = ['balanced']
                        )    

        best_classifiers = {}

        for classifier in classifiers:
            ## Method for supplying just the parameter grid entries related to the classifier
            ## in the current interation while excluding the other classifer paramters.
            # dict comprehension method courtesy of BernBarn at:
            # http://stackoverflow.com/questions/14507591/python-dictionary-comprehension
            param_for_class = {key: value for key,value in param_grid.iteritems() if
                                 re.search(key.split("_")[0],'features ' + classifier)}
            
            ## Feature selection method, same for all classifiers
            pca = PCA()
            selection = SelectPercentile()
            
            feature_select = FeatureUnion([("pca",pca),("univ_select",selection)])
            
            ## Active the classifier for the current iteration
            clf = classifiers[classifier]
        
            ## Pipeline the feature selection, feature scaling and classifier for optimization
            ## procedure
            pipeline = Pipeline([
                                ("features", feature_select),
                                ("scaler", MinMaxScaler()),
                                (classifier,clf)
                                ])
            #print param_for_class
            search = RandomizedSearchCV(estimator = pipeline,
                                        param_distributions = param_for_class,
                                        scoring = 'f1_weighted',
                                        n_jobs=-1,
                                        cv = 10,
                                        n_iter = 500, 
                                        error_score = 0)
            
            results = search.fit(X_train,y_train)
            best_classifiers[classifier] = results.best_estimator_

            ## Save the best classifier pipeline objects to local directory using pickle
        PickleBestClassifers(best_classifiers,"Best_Classifiers.pkl")
    else:
        ## After initial run of grid search, reference the pickled outcomes for the 
        ## rest of the analysis. Actual searching process takes a while
        ## on my system setup, so I want to run it as few times as possible. 
        savedResults = open("Best_Classifiers.pkl",'r')  
        best_classifiers = pickle.load(savedResults)  
    
    for key,value in best_classifiers.iteritems():
        print "Parameters for {0}\nFEATURE SELECTION:\n[{1}]\nSCALER:\n[{2}]\nCLASSIFIER:\n[{3}]\n\n".\
        format(key,value.steps[0][1].get_params(),
               value.steps[1][1],
               value.steps[2][1])
        
        ## Method of accessing pipeline objects and performing transformation found at 
        ## Zac Stewarts blog:
        ## http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html
        
        ## transform and predict on the X_Test split of the data
        X_test_data = value.steps[0][1].transform(X_test)
        X_test_data_scl = value.steps[1][1].transform(X_test_data)
        pred = value.steps[2][1].predict(X_test_data_scl) 
    
        ## return classification report of prediction results compared to truth values
        print key + " Score:" + "\n" + (classification_report(y_test, 
                                                              pred, 
                                                              target_names=['non-poi','poi']
                                                             ))

 