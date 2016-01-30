#!/python/bin/python
#Author: Zach Farmer
#Purpose: Feature Selection
"""
    Following function is designed to find features that contain the greatest
    explanation power in regards to the classification goal of identifying 
    poi's. Function implements sklearn's SelectPercentile method and PCA methods.
    Parameters for these two methods should be discovered using the gridsearch 
    optimization in a later script. 
"""   

def featureSelection(reduced_features,labels,clnd_features,percentile,n_components,results=False):
    """
        Parameters: 
            reduced_features = Unique feature names in python list after dropping non-numeric
            feaures. 
            labels = ground truth labels for the data points.
            clnd_features = data point features in numpy array format corresponding
            to the labels.
            percentile= the parameter for the SelectPercentile method;
            between 0.0-1.0.
            n_components = the n_components for the pca. 
            results = False returns python list of selected features. If True
            returns the metrics of the feature selectors (F-statistic, and p-values from
            f_classif) and the top 'n' pca component variance measurements. 
    
        Output: 
           Resulting list of feature from the SelectPercentile function and the 
           number of principle components used. If p_results = True then the 
           statistics of the SelectPercentile method using f_classif will be printed.
           In addition the explained variance of the top 'x' principle components will
           also be printed.
    """
    from sklearn.feature_selection import SelectPercentile, f_classif
    from sklearn.decomposition import PCA 
    from itertools import compress
    
    selector = SelectPercentile(f_classif, percentile=percentile)
    selector.fit_transform(clnd_features, labels)
    
    pca = PCA(n_components = n_components)
    pca.fit_transform(clnd_features, labels)
    
    if results == True:
    
        f_stat = sorted(zip(reduced_features[1:],f_classif(clnd_features,labels)[0]),\
                         key = lambda x: x[1], reverse=True)
        
        p_vals = sorted(zip(reduced_features[1:],f_classif(clnd_features,labels)[1]),\
                        key = lambda x: x[1])
  
        expl_var = pca.explained_variance_ratio_
        
        return f_stat,p_vals,expl_var
    else:
        ## return a boolean index of the retained features 
        retained_features = selector.get_support()
        
        ## index the original features by the boolean index of top x% features 
        ## return a python list of the features to be used for training 
        features_list = list(compress(reduced_features[1:],retained_features))
    
        ## add back in the 'poi' to the first position in the final features list
        features_list.insert(0,'poi')
        
        return features_list