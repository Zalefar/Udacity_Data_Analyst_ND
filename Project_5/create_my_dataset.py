#!/usr/bin/python
#Author: Zach Farmer
#Purpose: Create my own dataset from provided data_dict 
"""
    Functions for creating my own dataset from the provided data_dict.
    
    Function for dropping features by hand, chiefly for the 
    removal of the non-numeric feature 'email_address'. One could 
    certainly manually remove other variables to perform feature selection
    It is recommended that a more analytical approach be taken using sklearn's
    feature selection methods. 

    Function for computing fraction between two features provided. For the purpose
    of creating new features based on ratios. 
    
    Function for generating new features provided. This function implements hard 
    coded features. It is not abstractable for the creation of any other features 
    as written.  
    
    Function for removing outliers found in the dataset, it can except a list of 
    datapoints to remove.
    
    Function for correcting financial data for one of the persons in the data set.
    This function is hard-coded and cannot be abstracted for use on other persons,
    and features.
    
"""   

def dropFeatures(features, remove_list):
    """
        Parameters: 
            features = Python list of unique features in the data_dict. 
            remove_list = Python list of features to be removed.(drop 
            non-numeric features such as the email address) 
    
        Output: 
            Python list of unique features sans the features in the remove
            list. 
    """   
    ## Method courtesy of user: Donut at: 
    ## http://stackoverflow.com/questions/4211209/remove-all-the-elements-that-occur-in-one-list-from-another
    features_remove = remove_list
    learning_features = [feature for feature in features if feature not in features_remove]
    
    return learning_features

## Following code adapted from Udacity's Intro to Machine learning lesson 11 Visualizing
## Your New Feature Quiz
def computeFraction(feature_1, feature_2 ):
    """ 
        Parameters: 
            Two numeric feature vectors for which we want to compute a ratio
            between
     
        Output: 
            Return fraction or ratio of feature_1 divided by feature_2
    """
    
    fraction = 0.
    
    if feature_1 == "NaN":  
        fraction = 0.0
    elif feature_2 == "NaN":
        fraction = 0.0
    else: 
        fraction = int(feature_1) / float(feature_2)

    return fraction
     
def newFeatures(data_dict):
    """
        Parameters: 
            data_dict provided by Udacity instructors   
    
        Output: 
            data_dict with new features (hard-coded)
    """
    ## following is not extensible to making any other features 
    ## then what is hard coded below.

    ## Note: Some of the following features are susceptible to data leakage.
    ## The features which inlcude the links to poi's through emails mean that 
    ## the test data potentially inlcudes ground truth information of actual poi's.
    ## potentially inlcudes ground truth information of actual poi's. 

    for name in data_dict:

        from_poi_to_this_person = data_dict[name]["from_poi_to_this_person"]
        to_messages = data_dict[name]["to_messages"]
        fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
        
        data_dict[name]["fraction_from_poi"] = fraction_from_poi


        from_this_person_to_poi = data_dict[name]["from_this_person_to_poi"]
        from_messages = data_dict[name]["from_messages"]
        fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    
        data_dict[name]["fraction_to_poi"] = fraction_to_poi
        
        salary = data_dict[name]['salary']
        total_payments = data_dict[name]['total_payments'] 
        salary_to_totalPayment_ratio = computeFraction(salary, total_payments)
        
        data_dict[name]['salary_to_totalPayment_ratio'] = salary_to_totalPayment_ratio
        
        salary = data_dict[name]['salary']
        total_stock = data_dict[name]['total_stock_value'] 
        salary_to_stockValue_ratio = computeFraction(salary, total_stock)
        
        data_dict[name]['salary_to_stockValue_ratio'] = salary_to_stockValue_ratio
        
    return data_dict   

def removeOutliers(data_dictionary, list_data_points):
    """
        remove the data points associated with any discovered outliers
        
        Parameters:
            data_dictionay = The Udacity provided data set data_dict or my_dataset.
            data_point_name = The key name for the datapoint containing
            outliers. (e.g. 'Total'). 
        Output:
            data_dictionary with the provided data point removed from the 
            dictionary.
    """
    for elem in list_data_points:
        try:
            data_dictionary.pop(elem,0)
            return data_dictionary
        except ValueError:
            print "data_point not found in data_dict."
            pass 
    
    return None

def fixFinancialData(data_dictionary):
    """
        Fix the financial data mistake for 'BHATNAGAR SANJAY'
        Parameters:
            data_dictionary: Python dictionary containing the data set 
        Output:
            Pythin dictionary containing the data set with 'BHATNAGAR SANJAY'
        financial data corrected
    """
    data_dictionary['BHATNAGAR SANJAY']['total_payments'] = 137864
    data_dictionary['BHATNAGAR SANJAY']['restricted_stock'] = 2604490
    data_dictionary['BHATNAGAR SANJAY']['restricted_stock_deferred'] = -2604490
    data_dictionary['BHATNAGAR SANJAY']['expenses'] = 137864
    data_dictionary['BHATNAGAR SANJAY']['other'] = 'NaN'
    data_dictionary['BHATNAGAR SANJAY']['exercised_stock_options'] = 15456290
    data_dictionary['BHATNAGAR SANJAY']['total_stock_value'] = 15456290
    
    return data_dictionary