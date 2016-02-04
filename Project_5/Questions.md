##Questions     

***1. Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?***     
***
**Answer:**         
*Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question.*          
* The fundamental goal of this project is to determine whether or not an Enron employee is a person of interest (___POI___<font color="red" size = 3>*</font>) in the massive fraud perpetrated by the corporation. We will use a machine learning classifier taking as input a series of features and outputting a prediction as to whether a person is a POI or not. The series of input features will be made up of the massive court mandated release of Enron's data, from financial data to email messages.  
> *"This dataset was collected and prepared by the CALO Project (A Cognitive Assistant that Learns and Organizes). It contains data from about 150 users, mostly senior management of Enron, organized into folders. The corpus contains a total of about 0.5M messages. This data was originally made public, and posted to the web, by the Federal Energy Regulatory Commission during its investigation."* [Enron Email Dataset](https://www.cs.cmu.edu/~./enron/ "https://www.cs.cmu.edu/~./enron/")       

* The financial data was collected for the employees in the email corpus and mapped to the relevant people by Udacity Instructors.

* By treating each of those features gathered from the above resources as vectors containing underlying information regarding possible fraud we can mathematically work towards constructing a model to predict behavior that is possibly fraudulent. By these means if an effective model can be found we should be able to simply plug in the inputs (features) of an employee and be told with hopefully high accuracy whether that employee was likely engaged in fraudulent behavior. Remember that in our case we are simply deciding whether or not an employee should be given extra scrutiny (i.e. person of interest).   

> <font color="red", size = 2>*What is a person of interest: Indicted, Settled without admitting guilt, Testified in exchange for immunity*</font>     

* Summary of the data set found in the data_dict. This summary covers the original data set before prunning, additions and transformations.
```
Number of data points in the data_dict: 146
Number of Features: 21
Number of Persons of Interest in data_dict: 18
```   

* Number and Percent of missing values ('NaN's') for each feature for the full data set,
a subset containing just the poi's and a subset of the dataset sans poi's:

Features | Number_Full | Percent_Full | Number_POI | Percent_POI | Number_sansPOI | Percent_sansPOI
--------|-------------|--------------|------------|-------------|----------------|----------------
 bonus | 64 | 43.84 | 2 | 11.11 | 62 | 48.44
deferral_payments | 107 | 73.29 | 13 | 72.22 | 94 | 73.44
deferred_income | 97 | 66.44 | 7 | 38.89 | 90 | 70.31
director_fees | 129 | 88.36 | 18 | 100.00 | 111 | 86.72   
email_address | 35 | 23.97 | 0 | 0.00 | 35 | 27.34
exercised_stock_options | 44 | 30.14 | 6 | 33.33 | 38 29.69
expenses | 51 | 34.93 | 0 | 0.00 | 51 | 39.84
from_messages | 60 | 41.10 | 4 | 22.22 | 56 | 43.75
from_poi_to_this_person | 60 | 41.10 | 4 | 22.22 | 56 | 43.75
from_this_person_to_poi | 60 | 41.10 | 4 | 22.22 | 56 | 43.75
loan_advances | 142 | 97.26 | 17 | 94.44 | 125 | 97.66
long_term_incentive | 80 | 54.79 | 6 | 33.33 | 74 | 57.81
other | 53 | 36.30 | 0 | 0.00 | 53 | 41.41
poi | 0 | 0.00 | 0 | 0.00 | 0 | 0.00
restricted_stock | 36 | 24.66 | 1 | 5.56 | 35 | 27.34
restricted_stock_deferred | 128 | 87.67 | 18 | 100.00 | 110 | 85.94
salary | 51 | 34.93 | 1 | 5.56 | 50 | 39.06
shared_receipt_with_poi | 60 | 41.10 | 4 | 22.22 | 56 | 43.75
to_messages | 60 | 41.10 | 4 | 22.22 | 56 | 43.75
total_payments | 21 | 14.38 | 0 | 0.00 | 21 | 16.41
total_stock_value | 20 | 13.70 | 0 | 0.00 | 20 | 15.6

> A quick glance at the data suggests that a.) there is just not that much data with only 146 data points and that b.) for a many of those data points large portions of their features are missing values. The key problem with this data set is the nominal number of data points, most machine learning algorithms performance is closely related to the quantity of data on which to train[*"Scaling to Very Very Large Corpora for Natural Language Disambiguation"*](http://ucrel.lancs.ac.uk/acl/P/P01/P01-1005.pdf) (in addition to the quality, however in our case our concern is mostly with the quantity and lack thereof). This means that the missing values are a big deal, Udacity's `featureFormat` function chooses to replace all the 'NaN' values with 0.0 and then remove any data points where all of the features are 0.0 (an additional option allows you to remove a data point that contains any 0.0 valued features--to aggressive in my opinion). There are problems that can arise by simply assuming that a 'NaN' can be replaced with a 0.0 without consequence. However other imputation methods are especially difficult in this case as we have so little other data from which to draw inferences in order to weigh imputation methods. So keeping in mind that choosing to replace the missing values with 0's is inserting information into our model and is not innocuous I will continue with the `featureFormat` function.

*Were there any outliers in the data when you got it, and how did you handle those?*       
* Yes there was at least one major outlier that perpetuated itself throughout the financial data. This outlier was a data point labeled "TOTAL" that upon examination of the financial data pdf clearly represented the 'Totals' of the financial features. Obviously this data point is not a person and the information it contains already exists in each of the features by simply summing over all of their values. I chose to drop the data point. 'THE TRAVEL AGENCY IN THE PARK' is not really a person and although may have been used to commit fraud it is a coporate entity and not an individual, I also dropped this data point. Reviewing some more of the extreme values in the features (values that are greater then or less then 1.5 times the I.Q.R. (inter-quartile-range) in either direction) I found at least one data point with incorrect information. According to the `enron61702insiderpay.pdf` document Bhatnagar Sanjay's financial data was all switched around, I fixed these errors and not finding any more red flags in the data moved on to creating my own features. 

***2. What features did you end up using in your poi identifier, and what selection process did you use to pick them? did you have to do any scaling? why or why not? Explain what feature you tried to make, and the rationale behind it. If you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like selectkbest, please report the feature scores and reasons for your choice of parameter values.***   
***
**Answer:**   
*What features did you end up using in your POI identifier?*   
* Features used in the POI identifier model from Sklearn's SelectPercentile:
```python
['expenses',      
 'deferred_income',    
 'from_poi_to_this_person',      
 'exercised_stock_options',       
 'shared_receipt_with_poi',      
 'loan_advances',       
 'other',     
 'bonus',      
 'total_stock_value',      
 'long_term_incentive',     
 'restricted_stock',     
 'salary',      
 'total_payments',     
 'fraction_to_poi',      
]
```
* In addition I implemented a PCA on all of the features and returned the top 22 components. These top 22 components were combined with the top 61% of features using sklearn's feature union method       

*What selection process did you use to pick them?*     
* I used sklearn's Select-Percentile method with percentile = 61% and sklearn's f\_classif (ANOVA) providing the metrics returning F-statistics and P-values. For the top 22 principle components I used sklearn's PCA fit_transform.      

*Did you have to do any scaling? Why or why not?*     
* Yes I implemented a MinMaxScaler. My features contain vastly different scales; from the to/from email percentages to total stock values in the tens of millions. While certain algorithms may handle such vast differences with greater ease (decision trees), In order to be as flexible as possible with my algorithm choices I scaled all of the features for my dataset.     

*Engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it.*   
* I felt that Katie's (Udacity Intro to Machine Learning course co-instructor) idea of generating features that measured the frequency of communications both to and from known POIs was an excellent idea. This method of utilizing the emails allows us to gather some of the information available in the email corpus without trying to engineer methods of including the actual content of the emails in the same dataset as the financial data, this spared me from the trouble of combining two different datasets, which did not contain all of the same data points. Furthermore Katie's idea of a shared receipt also resonated with me as an excellent method of capturing second degree associations with POIs. After implementing the fraction_to and fraction_from this person to POI and the shared receipt feature I engineered two other features. Salary to total payment ratio and salary to total stock value ratio. These features were engineered on the hypothesis that an individual is most likely to commit fraud because they are receiving some sort of benefit from doing so. In other words it is a risk and reward trade-off and my theory was that individuals committing fraud would show a lower ratio of salary to total compensation metrics. In other words two people in similar professional standings with fairly similar base salaries should show differences in total compensation metrics. In the case of an individual committing fraud they would be more likely to receive financial gains as a result of their fraudulently attained success in the form of some type 'bonus' compensation and therefore they should present higher total payments and total stock values and consequently a lower ratio of salary to those metrics. My engineered features did not make it past the feature selection stage, unless you count them as possibly being included in the top principle components. I believe that because many of the key individuals involved in the fraud were at the very top of the company and those individuals naturally receive total compensation far in excess of their base salaries the information I was hoping to discover in the data wasn't very distinctive.   

*In your feature selection step, if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.*    
* I used Sklearn's SelectPercentile method to select my features. The parameter choice of 61% for the Select-Percentile method was arrived at after running a randomized grid search over my feature selection and algorithm parameters. The combination with the best outcome was a combination of the top 61% of features and the top 22 principle components from the PCA. Technically the top 22 components are transformations of the features therefore I used 14 of the original features but all the features were used to generate the 22 principle components.    

* F-statistics of sklearn's f-classif (ANOVA) for features:
```python
[('total_stock_value', 22.783481328003685),
 ('exercised_stock_options', 22.610389609556254),
 ('bonus', 21.060001707536571),
 ('salary', 18.575703268041785),
 ('fraction_to_poi', 16.641707070468989),
 ('deferred_income', 11.595547659730601),
 ('long_term_incentive', 10.072454529369441),
 ('total_payments', 9.3782232542572856),
 ('restricted_stock', 8.9617839833519675),
 ('shared_receipt_with_poi', 8.7464855321290802),
 ('loan_advances', 7.2427303965360181),
 ('expenses', 5.5575259528039913),
 ('from_poi_to_this_person', 5.3449415231473374),
 ('other', 4.2198879086807812),
 ('fraction_from_poi', 3.2107619169667441),
 ('salary_to_totalPayment_ratio', 2.7730011744152532),
 ('from_this_person_to_poi', 2.4265081272428781),
 ('director_fees', 2.1076559432760908),
 ('to_messages', 1.6988243485808501),
 ('restricted_stock_deferred', 0.74349338833843037),
 ('deferral_payments', 0.2170589303395084),
 ('from_messages', 0.16416449823428736),
 ('salary_to_stockValue_ratio', 0.022229695372865222)]
```
* P-values of sklearn's f-classif (ANOVA) for features:
```python
[('total_stock_value', 4.4581962180195415e-06),
 ('exercised_stock_options', 4.8180739513255001e-06),
 ('bonus', 9.7024743412322453e-06),
 ('salary', 3.0337961075304736e-05),
 ('fraction_to_poi', 7.4941540250267645e-05),
 ('deferred_income', 0.00085980314391990965),
 ('long_term_incentive', 0.0018454351466116368),
 ('total_payments', 0.0026272646435664833),
 ('restricted_stock', 0.0032527805526498307),
 ('shared_receipt_with_poi', 0.0036344020243633686),
 ('loan_advances', 0.0079738162605691599),
 ('expenses', 0.019766170956403498),
 ('from_poi_to_this_person', 0.022220727960811395),
 ('other', 0.041786766547797359),
 ('fraction_from_poi', 0.075284900599151758),
 ('salary_to_totalPayment_ratio', 0.098071056290785164),
 ('from_this_person_to_poi', 0.12152433983710857),
 ('director_fees', 0.14876949527311398),
 ('to_messages', 0.19455111487450777),
 ('restricted_stock_deferred', 0.3899979566984314),
 ('deferral_payments', 0.64200389403828306),
 ('from_messages', 0.68596070789958996),
 ('salary_to_stockValue_ratio', 0.88168921319693783)]
```
* The variance explained by the top 22 components of the PCA:
```python
[(1, 0.80812036659075415),
 (2, 0.15586745915579647),
 (3, 0.015109657195381328),
 (4, 0.011686812385154257),
 (5, 0.0044491635021789676),
 (6, 0.0021341618555299909),
 (7, 0.0013267525381302767),
 (8, 0.00074184001233473297),
 (9, 0.00045975620232343613),
 (10, 8.933871000077515e-05),
 (11, 9.8349310287513656e-06),
 (12, 4.58776419547978e-06),
 (13, 2.3210865596604806e-07),
 (14, 2.2198143780232896e-08),
 (15, 7.3222820998496433e-09),
 (16, 6.8783812819255299e-09),
 (17, 6.2553998581518125e-10),
 (18, 1.3945484619683135e-11),
 (19, 1.0227424614083703e-11),
 (20, 1.5071393030939921e-14),
 (21, 1.3600895757998964e-16),
 (22, 8.2573694413750787e-17)]
```
***3. what algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?***    
****
**Answer:**     
```
Parameters for GNB
FEATURE SELECTION:
[{'n_jobs': 1, 'univ_select': SelectPercentile(percentile=21,
         score_func=<function f_classif at 0x10ca48c08>), 'pca__copy': True, 'transformer_list': [('pca', PCA(copy=True, n_components=7, whiten=False)), ('univ_select', SelectPercentile(percentile=21,
         score_func=<function f_classif at 0x10ca48c08>))], 'pca__n_components': 7, 'pca__whiten': False, 'pca': PCA(copy=True, n_components=7, whiten=False), 'transformer_weights': None, 'univ_select__score_func': <function f_classif at 0x10ca48c08>, 'univ_select__percentile': 21}]
SCALER:
[MinMaxScaler(copy=True, feature_range=(0, 1))]
CLASSIFIER:
[GaussianNB()]


GNB Score:
             precision    recall  f1-score   support

    non-poi       1.00      0.93      0.96        14
        poi       0.50      1.00      0.67         1

avg / total       0.97      0.93      0.94        15

Parameters for LRC
FEATURE SELECTION:
[{'n_jobs': 1, 'univ_select': SelectPercentile(percentile=61,
         score_func=<function f_classif at 0x10ca48c08>), 'pca__copy': True, 'transformer_list': [('pca', PCA(copy=True, n_components=22, whiten=False)), ('univ_select', SelectPercentile(percentile=61,
         score_func=<function f_classif at 0x10ca48c08>))], 'pca__n_components': 22, 'pca__whiten': False, 'pca': PCA(copy=True, n_components=22, whiten=False), 'transformer_weights': None, 'univ_select__score_func': <function f_classif at 0x10ca48c08>, 'univ_select__percentile': 61}]
SCALER:
[MinMaxScaler(copy=True, feature_range=(0, 1))]
CLASSIFIER:
[LogisticRegressionCV(Cs=2, class_weight='balanced', cv=None, dual=False,
           fit_intercept=True, intercept_scaling=1.0, max_iter=100,
           multi_class='ovr', n_jobs=1, penalty='l2', random_state=42,
           refit=True, scoring=None, solver='lbfgs', tol=0.0001, verbose=0)]


LRC Score:
             precision    recall  f1-score   support

    non-poi       1.00      0.86      0.92        14
        poi       0.33      1.00      0.50         1

avg / total       0.96      0.87      0.89        15

Parameters for RDF
FEATURE SELECTION:
[{'n_jobs': 1, 'univ_select': SelectPercentile(percentile=91,
         score_func=<function f_classif at 0x10ca48c08>), 'pca__copy': True, 'transformer_list': [('pca', PCA(copy=True, n_components=2, whiten=False)), ('univ_select', SelectPercentile(percentile=91,
         score_func=<function f_classif at 0x10ca48c08>))], 'pca__n_components': 2, 'pca__whiten': False, 'pca': PCA(copy=True, n_components=2, whiten=False), 'transformer_weights': None, 'univ_select__score_func': <function f_classif at 0x10ca48c08>, 'univ_select__percentile': 91}]
SCALER:
[MinMaxScaler(copy=True, feature_range=(0, 1))]
CLASSIFIER:
[RandomForestClassifier(bootstrap=True, class_weight='balanced',
            criterion='entropy', max_depth=21, max_features='auto',
            max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=161, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)]


RDF Score:
             precision    recall  f1-score   support

    non-poi       0.93      1.00      0.97        14
        poi       0.00      0.00      0.00         1

avg / total       0.87      0.93      0.90        15

Parameters for ADB
FEATURE SELECTION:
[{'n_jobs': 1, 'univ_select': SelectPercentile(percentile=71,
         score_func=<function f_classif at 0x10ca48c08>), 'pca__copy': True, 'transformer_list': [('pca', PCA(copy=True, n_components=12, whiten=False)), ('univ_select', SelectPercentile(percentile=71,
         score_func=<function f_classif at 0x10ca48c08>))], 'pca__n_components': 12, 'pca__whiten': False, 'pca': PCA(copy=True, n_components=12, whiten=False), 'transformer_weights': None, 'univ_select__score_func': <function f_classif at 0x10ca48c08>, 'univ_select__percentile': 71}]
SCALER:
[MinMaxScaler(copy=True, feature_range=(0, 1))]
CLASSIFIER:
[AdaBoostClassifier(algorithm='SAMME.R',
          base_estimator=DecisionTreeClassifier(class_weight='balanced', criterion='gini',
            max_depth=None, max_features=None, max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best'),
          learning_rate=87.8748897177, n_estimators=360, random_state=None)]


ADB Score:
             precision    recall  f1-score   support

    non-poi       0.92      0.86      0.89        14
        poi       0.00      0.00      0.00         1

avg / total       0.86      0.80      0.83        15

Parameters for SVC
FEATURE SELECTION:
[{'n_jobs': 1, 'univ_select': SelectPercentile(percentile=91,
         score_func=<function f_classif at 0x10ca48c08>), 'pca__copy': True, 'transformer_list': [('pca', PCA(copy=True, n_components=9, whiten=False)), ('univ_select', SelectPercentile(percentile=91,
         score_func=<function f_classif at 0x10ca48c08>))], 'pca__n_components': 9, 'pca__whiten': False, 'pca': PCA(copy=True, n_components=9, whiten=False), 'transformer_weights': None, 'univ_select__score_func': <function f_classif at 0x10ca48c08>, 'univ_select__percentile': 91}]
SCALER:
[MinMaxScaler(copy=True, feature_range=(0, 1))]
CLASSIFIER:
[SVC(C=230.677442765, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape=None, degree=3, gamma=0.100266991948,
  kernel='rbf', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False)]


SVC Score:
             precision    recall  f1-score   support

    non-poi       0.92      0.86      0.89        14
        poi       0.00      0.00      0.00         1

avg / total       0.86      0.80      0.83        15
```      
* I use a cross-validated logistic regression classifier for my algorithm. I tried out 5 different classifiers in order to determine which ones provides the best performance. Using Sklearn's classification report to analyze the results of the 5 classifiers I found that a Gaussian Naive Bayes algorithm had the best precision and recall for identifying both poi's and non-poi's (specifically a high recall which I deemed as the more important of the two metrics in this case). However testing with Udacity's `tester.py` script indicated that the Logistice Regression CV provided the best outcome. The adaBoost, randomforest, gaussian naive bayes and support vector classifier algorithms all had great success at identifying poi's and non-poi's with high precision and recall. However none of these algorithms had any success with precision and recall when it came to identifying poi's. Given the choice between being especially aggressive in predicting poi's and generating lots of false positives vs. being to conservative and failing to identify known poi's I will lean towards the algorithm that successfully manages to predict most of the know poi's as poi's and generates a few more false positives as a result.  

***4. What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm?*** 
****
**Answer:**        
* Tuning of parameters is done to adjust those components of the algorithm that are associated with how the algorithm is fitting your data. To tune the parameters we have to decide on a metric of success and compare the results of that metric between different parameter values, selecting the 'tuned' parameters that offer us the best scores. These parameters have great influence over bias and variance which means that getting this step wrong can introduce high bias or high variance into our models. I chose to address the problem of tuning my parameters using a randomized cross-validated grid search over a range of values for many of my tunable parameters. This method automates the process to a degree, I still have to provide the ranges of parameters to be tested. The model I used implemented a tuned feature selection algorithm, a tuned pca and a tuned logistic regression classifier. Essentially I perturbed the number of features to included, varied the number of principle components to included and finally adjusted the strength of the regularization term in the classifier. By performing all these steps I was able to tune my entire model to achieve relatively descent success in identifying poi's and non-poi's    

***5. What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?***  
****
**Answer:**   
* Validation is the step taken to assess the accuracy of our trained model. It's necessary to ensure that our model will generalize well, which is after all the primary purpose of a machine learning algorithm. A classic mistake with validation is to not appropriately separate testing and training data. If we use any of the data points in the training of an algorithm, whether that is to train the model to fit the data or to tune the hyper-parameters of a model, in the testing of our trained model then we have deprived ourselves of that data's anonymity. We should always test our model using data that was not associated with the fitting of our model or the training of it's hyper-parameters. This is extremely important because if we use data that we trained with to test the performance of a model we are failing to address the primary directive of any forecasting model which is to provide accurate assessments on data not seen before. We would be allowing the information contained in training phases to leak into our generalizable model, meaning that we would have no way of accurately predicting how well our model will perform on completely unknown data because it will not have seen completely unknown data. By not properly validating our model we will not have any idea how effective our algorithm really is, not a good thing especially if money is on the line. I validated my data first by splitting the original data set using a stratified split to separate my data into a training set of the features and corresponding labels and testing set of features of the same format. After the separation I passed the training sets into a grid search which used a stratified shuffle split to again split my training data so that I could perform not just a model fitting to my data but could also fit the 'best' hyper-parameters. By using this cross-validated method I was able to find the best hyper-parameters without validating on the same data used to fit the model. This meant that after finding the best parameters I could then test the final model and parameters on my first held out test set to validate the overall model score without having to utilize any of the data points used in the fitting of the model and in the training of the hyper-parameters to measure the success of my model.  

***6. Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance.***  
****
**Answer:**  
```
Logistic Regression Classifier Report:
             precision    recall  f1-score   support

    non-poi       1.00      0.86      0.92        14
        poi       0.33      1.00      0.50         1

avg / total       0.96      0.87      0.89        15
```    
> NOTE: The following answer is based on my results from testing on my holdout test set. The results from the `tester.py` are different, however the essence of the results are the same. The test set uses more 'support' data points both non-poi and poi so the extremes in my results, the 100%, are mellowed out to much lower then 100%. That being said the result are proportionally very similar between my test data and the `tester.py` outcome.    

***Answer:*** The classification report above tells us a few important things about the model's performance. The precision metric is indicating the percentage of relevant (that is positive and correct) predictions made by the model over all the positive predictions made by the model (correct or incorrect). In other words how many of the positive predictions made by the model are actually relevant (positive and correct) when compared to ground truth reality. Recall on the other hand compares the number of relevant (positive and correct) predictions made by the model vs. the total number of relevant (positive and correct) values that actually exist in reality. In other words how many of the real relevant (positive and correct) ground truth values does our model identify compared to the total number that actually exists. The classification report above further breaks down the precision and recall for each of the possible labels non-poi and poi. Given this information we can say that according to this classification report our model does an excellent job of correctly identifying all non-poi's labels (perfect 100% of predictions are in reality correct) and less success with poi's (only 33% of are predicted positive labels are in reality actually real pois). Our model however has more success when it comes to correctly identifying all the actual poi (positive and correct) labels that exist in the test set data (100% perfect capture of all the possible positive poi labels that can be found in the test data set). We can see that it identifies most of the non-pois that can be found in the test data set. Although this model is more likely to cause a greater workload for those tasked with looking into the individuals, as a result of predicting more poi's than actually exist, the trade-off is in my mind worth it. I am more biased towards an algorithm that will correctly identify all of the poi's that do exist in reality even if it means flagging more innocent individuals for further scrutiny. 
