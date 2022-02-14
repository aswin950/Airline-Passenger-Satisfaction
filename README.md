# Airline Passenger Satisfaction
Machine learning Models for Airline Passenger Satisfaction using Python.

The dataset was retrieved from www.kaggle.com/teejmahal20/airline-pDassenger-satisfaction

Dataset download link: https://github.com/aswin950/Airline-Passenger-Satisfaction/blob/main/test.csv

Python Jupyter notebook link: https://github.com/aswin950/Airline-Passenger-Satisfaction/blob/main/Airline%20Passenger%20Satisfication.ipynb

Models:

1.) Decision Tree Classifier,

2.) Gradient Boosting Classifier,

3.) Logistic Regression Classifier, and 

4.) Random Forest Classifier

# I. INTRODUCTION

The dataset was retrieved from www.kaggle.com/teejmahal20/airline-pDassenger-satisfaction, where there are two files test.csv with 20% of full dataset. We will be using the test file for our final project. This also has training data set with about 80% of full data. The data set is about Airline Passenger Satisfaction survey and this data set is posted by the user TJ Klein where he has mentioned that he modified the dataset from the user John D. The data set has 25,975 data points and 25 columns.

# II. EXPLORATORY DATA ANALYSIS:

Exploratory data analysis is carried out in this section in the context of data exploration. The libraries used in this assignment are Pandas for data analysis, KNNImputer is used for imputing the missing values, Matplotlib and seaborn library data visualization.
Head of Data Frame is used to display the first five rows from the data set.

<img width="468" alt="image" src="https://user-images.githubusercontent.com/61600236/153953462-dbe8c8b6-1841-4aff-9736-30b7faa5985b.png">

Figure 1.1 Head of Data Frame

Next, the data information is displayed to show the data type. The data set consists of 25 columns and 25,976 rows. Numerical variables are Age, Flight Distance, Inflight Wi-Fi service, Ease of Online booking, Gate location, Seat comfort, Cleanliness. And categorical variables are Gender, Customer Type, Type of Travel, satisfaction.

<img width="468" alt="image" src="https://user-images.githubusercontent.com/61600236/153953730-fed2f93c-1d81-4fec-9c77-01947a5593bc.png">

Figure 1.2 Information of data

Data cleanup and preprocessing has to be done to remove the unwanted columns and the head of data is displayed.

<img width="468" alt="image" src="https://user-images.githubusercontent.com/61600236/153953804-0659bbd8-13d3-499a-ac5c-8e0f0a44297b.png">

Figure 1.3 Remove Unwanted Column

We can observe that columns are too lengthy. We will rename these columns. 

<img width="468" alt="image" src="https://user-images.githubusercontent.com/61600236/153953850-3538d333-a184-4adb-9fdf-69d987587238.png">

Figure 1.4 Renaming the Columns

Missing values needs to be checked that are present in the data set. missing values are present in the Arrival Delay in Minute's variable. Below is a screenshot showing number missing values in each column.

<img width="161" alt="image" src="https://user-images.githubusercontent.com/61600236/153953879-4e68b964-675f-4272-85f5-d8c1c4ee35af.png">

Figure 1.5 Missing Values

The percentage of missing values in Arrival Delay time is 0.31%. We can either ignore or we can impute values by using mean, median, and KNN imputation method. Now, we can fill the missing values with 0 and check for any duplicates in the dataset.

<img width="333" alt="image" src="https://user-images.githubusercontent.com/61600236/153953922-eba4463a-6e2d-4638-b0cb-4e98e7574f52.png">

Figure 1.6 Filling Missing Values & Duplicates

Outliers / Anomalies are checked for Age, Flight Distance. No outliers are present in this Age variable. Outliers are present in the Flight distance variable. But, for this kind of dataset we can ignore outlier analysis because it is not important.

<img width="468" alt="image" src="https://user-images.githubusercontent.com/61600236/153953994-ebd80679-43d7-4ea4-b40b-540f230a766f.png">

Figure 1.7 Boxplot of Age

<img width="468" alt="image" src="https://user-images.githubusercontent.com/61600236/153954005-455712fd-dfd9-48fd-9ade-d58dac37939c.png">

Figure 1.8 Boxplot of Flight Distance

Since the data is cleaned and processed now, we can start the Exploratory Data Analysis. We can see summary statistics like mean, median, mode, count etc.

<img width="468" alt="image" src="https://user-images.githubusercontent.com/61600236/153954050-46e30df5-3bcc-4a4a-b16a-b89e398fb1d3.png">

Figure 1.9 Glimpse of Dataset

Correlation Matrix is displayed to find the correlation between the variables. It can be observed that there is less correlation between the variables.

<img width="468" alt="image" src="https://user-images.githubusercontent.com/61600236/153954083-9e5d8411-b0cb-4f7a-9b55-75087cb3363e.png">

Figure 1.10 Correlation Matrix

Univariate Analysis is used to know the count of the Gender and is displayed below in a bar chart. From the graph we know that number of Female passengers are more than that of Male passengers.

<img width="468" alt="image" src="https://user-images.githubusercontent.com/61600236/153954130-8fd8e3f6-e821-44ec-a2e8-09da55259720.png">

Figure 1.11 Bar Graph to know the count of Gender

We can also count the number of loyal customers and disloyal customers using the bar chart. From the graph shown below we can find that loyal customers are more than disloyal customers.

<img width="431" alt="image" src="https://user-images.githubusercontent.com/61600236/153954172-e156950b-c4a6-4b5d-962d-e994ba3b1a94.png">

Figure 1.12 Bar Graph to know the count of Customer Type

Let us find use travel type to know the count business travel and personal travel. From the figure 1.13 we see that business travel passengers are more than personal travel passengers.

<img width="468" alt="image" src="https://user-images.githubusercontent.com/61600236/153954205-e323ba38-bc89-4591-a4aa-ab432f131f24.png">

Figure 1.13 Bar Graph of Travel Type

Now, we can find the class which maximum and lowest number of passenger’s travel. There are 3 different classes they are Economy, Business class, and Economy Plus. We can find that the count of Business class passengers is the highest and Economy Plus passengers are the lowest.

<img width="414" alt="image" src="https://user-images.githubusercontent.com/61600236/153954248-5bb9ce7d-aaea-4450-8cef-261c82896091.png">

Figure 1.14 Bar Graph of Class

Figure 1.15 shows the customer satisfaction among the airline passengers. From this we can conclude that neutral or dissatisfied customers are higher than the satisfied customers.

<img width="434" alt="image" src="https://user-images.githubusercontent.com/61600236/153954307-19ef75f7-66e0-4ee3-aa41-838129eb268b.png">

Figure 1.15 Bar Graph of Customer Satisfaction

Bivariate Analysis is used to find the average of customer satisfaction, average of gender, average of customer type, average of travel type, and average of class. The below chart represents the average age of customer satisfaction. The average age of satisfied passengers is around 42 and the average age of neutral or dissatisfied passengers is around 37.

<img width="363" alt="image" src="https://user-images.githubusercontent.com/61600236/153954342-dbcfa413-1614-43f1-8a9b-a8d715277d43.png">

Figure 1.16 Average age of Customer Satisfaction

Next, we can find the average age of Male and Female passengers using bar chart. The average age of Male and Female passengers is around 39.

<img width="468" alt="image" src="https://user-images.githubusercontent.com/61600236/153954380-2060bf05-f716-4d3a-9b21-7b670cd6a435.png">

Figure 1.17 Average Age of Gender

Average age of loyal customer is around 42 and average of disloyal customer is around 29.

<img width="447" alt="image" src="https://user-images.githubusercontent.com/61600236/153954422-97739974-3783-4c05-ad90-800b3a73cd7e.png">

Figure 1.18 Average Age of Loyal & Disloyal Customer

Now, we will find the average age of business travel and personal travel. The average age of business travel passengers is around 39 and the personal travel passengers is around 38.

<img width="468" alt="image" src="https://user-images.githubusercontent.com/61600236/153954469-4f33ec43-2083-411b-b608-0e55cdb3ff70.png">

Figure 1.19 Average Age of Business Travel

The average age of all class passengers is visualized. From the figure 1.19 we can say that the average age of Eco class passengers is around 37, the average age of business class passengers is around 43 and the average age of economy plus is around 38.

<img width="403" alt="image" src="https://user-images.githubusercontent.com/61600236/153954498-be5e3a7f-f14c-4cec-ba89-897c45c85c08.png">

Figure 1.20 Average Age of Class

# III. BUILDING MODELS FOR CUSTOMER SATISFACTION

In this section, we will be building various models to predict the customer satisfaction and their ROC and accuracy for performance. Let’s start by encoding the variables, since we are going to use this dataset to predict customer satisfaction we will be encoding the ‘satisfied’ customer as 1 and ‘neutral or dissatisfied’  customer as 0 as shown in fig 2.1 (a) and we have encoded the other variables too that are Gender, Customer_Type, Travel_type, and Class. If we need to build models and predict based on these variables the encoded variables can be used and it is shown in fig 2.1 (b).

<img width="260" alt="image" src="https://user-images.githubusercontent.com/61600236/153954559-1f8972ca-1d26-4bed-9504-0c622421b8ef.png">

Figure 2.1 (a) Encoding Customer Satisfaction

<img width="468" alt="image" src="https://user-images.githubusercontent.com/61600236/153954612-6b46fa6e-092b-4ab6-9952-b49650a982ba.png">

Figure 2.1 (b) Encoding Other Variables

Split the dataset into train and test, we have split the model as 80% as training data and 20% as test data to use in our models.

<img width="468" alt="image" src="https://user-images.githubusercontent.com/61600236/153956769-1d1440f5-2789-420e-bc48-65d8fb90195e.png">

Figure 2.2 Split the Dataset

Decision Tree Classifier: 
     Decision Tree Classifier is a simple machine learning model for classification problems. This is a type of supervised machine learning where we build a model and feed data with correct outputs and then we let the model learn from these patterns. Then we feed our model new data that it hasn’t observed before to see how it performs. As we know, decision tree has three nodes which are root node, decision node and terminal node. Hence, predicting a value means asking question from top node to the terminal node where we get a decision.
     
 <img width="423" alt="image" src="https://user-images.githubusercontent.com/61600236/153956938-3c0621d7-ae6e-4c84-aa5f-be64e08ad3af.png">

Figure 2.3 Decision Tree Classifier

<img width="312" alt="image" src="https://user-images.githubusercontent.com/61600236/153956967-dabbdc42-6115-40e4-a573-5ff1e6bbcb6f.png">

Figure 2.4 Decision Tree Classifier – Confusion Matrix & Classification report

<img width="396" alt="image" src="https://user-images.githubusercontent.com/61600236/153956992-a1d3487e-7f1a-43fd-bc5f-34da28a40423.png">

Figure 2.5 Decision Tree Classifier – ROC Curve

<img width="287" alt="image" src="https://user-images.githubusercontent.com/61600236/153957018-36eddd1a-0b05-42e5-9a1b-5b2825788260.png">

Figure 2.6 Importance Features

In fig 2.3, we have built a decision tree classifier model where it shows the CPU time, system time, wall time and total time in milliseconds and model performance is executed. Satisfied and neutral or dissatisfied passengers are represented and the accuracy score of decision tree classifier is found out to be 0.9351424172440339

In figure 2.4, we have created a confusion matrix and performance report. To better understand about confusion matrix, take a look at the figure 2.7 shown below.

<img width="304" alt="image" src="https://user-images.githubusercontent.com/61600236/153957093-563e9729-8318-48e9-b621-ae4af72340fc.png">

Figure 2.7 Confusion Matrix

Let’s us discuss about the terms mentioned in the confusion matrix,
True Positives: Actual positive values and predicted positive values are same.
True Negatives: Actual Negative values and predicted negative values are same.
False Positives: Where we have actual negative values and predicted positive values.
False Negatives: Where we have actual positive values and predicted negative values.
So, in our case, we have 
True Positives: Number of correctly predicted neutral or dissatisfied passengers is 2,123
True Negatives: Number of correctly predicted satisfied passengers is  2,7362
False Positives: Number of incorrectly predicted dissatisfied or neutral passengers is 175
False Negatives: Number of incorrectly predicted satisfied passengers is 162
From fig 2.5, we can say that f1 score is 0.94 and area under curve is 0.93  meaning the model is performing well. The AUC values represent the model performance and their range is mentioned in the below table.

AUC VALUES	        TEST QUALITY
0.9 – 1.0	   Excellent
0.8 - 0.9	   Very Good
0.7 – 0.8	   Good
0.6 – 0.7	   Satisfactory
0.5 – 0.6	   Unsatisfactory

And Figure 2.6 represents the important features for decision tree model and the bar plot.

Gradient Boosting Classifier: 
     As the name implies gradient boosting classifiers is a type of algorithm that are used for classification tasks. Gradient boosting classifier is a type of supervised machine learning algorithm that can combine many poorly performing models together to create a strong predictive model. Figure 2.8 – Figure 2.11 represents the gradient boosting classifier confusion matrix, performance report, ROC curve, and importance features

<img width="408" alt="image" src="https://user-images.githubusercontent.com/61600236/153957348-883bbb67-0db4-4d2e-8a59-a957887426f7.png">

Figure 2.8 Gradient Boosting Classifier

<img width="284" alt="image" src="https://user-images.githubusercontent.com/61600236/153957389-d0d7f4ab-2dbb-4a35-898f-fdf60a3b31dc.png">

Figure 2.9 Gradient Boosting Classifier - Classifier – Confusion Matrix & Classification report

<img width="385" alt="image" src="https://user-images.githubusercontent.com/61600236/153957409-7c9a2c50-6621-41ba-bfc2-ca6c57c2ef62.png">

Figure 2.10 Gradient Boosting Classifier – ROC Curve

<img width="305" alt="image" src="https://user-images.githubusercontent.com/61600236/153957433-a60aaebe-f6fe-4be6-bb33-99be2e7878f2.png">

Figure 2.11 Importance Features

Figure 2.8 represents the gradient boosting classifier model and the model accuracy score is found to be 0.9463048498845266 and from fig 2.9 we know the f1-score obtained from performance report as 0.95 with area under curve value as 0.99. From fig 2.10, the AUC value of 0.99 means that our model is performing excellently. And from fig 2.9 confusion matrix we can say that,

True Positives: Number of correctly predicted neutral or dissatisfied passengers is 2,795
True Negatives: Number of correctly predicted satisfied passengers is 2,795
False Positives: Number of incorrectly predicted dissatisfied or neutral passengers is 176
False Negatives: Number of incorrectly predicted satisfied passengers is 103
Figure 2.11 represents the importance feature and the scores and a bar plot of importance features.

Logistic Regression Classifier:
     Logistic Regression is a machine learning algorithm to predict the probability of a categorical dependent variable. In logistic regression, dependent variable is a binary value where it contains data as either 1 or 0. Logistic regression is a special type of linear regression in which the target variable is categorical. Logistic regression can predict the probability of an occurrence of a binary event. Figure 2.12 – Figure 2.15 represents the logistic regression classifier confusion matrix, performance report, ROC curve, and importance features.

<img width="324" alt="image" src="https://user-images.githubusercontent.com/61600236/153957504-abe8080d-33c4-46b3-90d4-5c56bbe27800.png">

Figure 2.12 Logistic Regression

<img width="344" alt="image" src="https://user-images.githubusercontent.com/61600236/153957530-49fb9c89-310a-4c98-a869-7fb2f57dbbb5.png">

Figure 2.13 Logistic Regression – Accuracy and Confusion matrix

<img width="273" alt="image" src="https://user-images.githubusercontent.com/61600236/153957561-f896c4c4-678a-43b0-9ad2-5b3aca3e4ce5.png">

Figure 2.14 Logistic Regression – Classification Report & ROC Curve

Figure 2.13 represents the logistic regression model and the model accuracy is found to be 0.6774441878367975 and from fig 2.14 we know the f1-score obtained from performance report as 0.68 with area under curve value as 0.75. The AUC value of 0.75 means that the model is performing good but it is not excellent. And from fig 2.9 confusion matrix we can say that,
True Positives: Number of correctly predicted neutral or dissatisfied passengers is 1,675
True Negatives: Number of correctly predicted satisfied passengers is 1,845
False Positives: Number of incorrectly predicted dissatisfied or neutral passengers is 623
False Negatives: Number of incorrectly predicted satisfied passengers is 1053
Figure 2.15 below represents the importance feature and the scores and a bar plot of importance features.

<img width="271" alt="image" src="https://user-images.githubusercontent.com/61600236/153957590-eaf5b85d-044a-4077-b35b-447fbc193d5d.png">

Figure 2.15 Importance Features

Random Forest Classifier:
     Random Forest is a supervised learning algorithm that can be used for both regression and classification. Random Forests creates decision trees on randomly selected data samples then gets prediction from each tree to select the best solution by voting. Prediction result that we get with the most votes is the final prediction. Random Forest is regarded as a highly accurate and strong method because of the number of decision trees involved in the process. Fig 2.16 represents random forest classifier model where it shows the CPU time, system time, wall time and total time in milliseconds. And Satisfied and neutral or dissatisfied passengers are represented and the accuracy score of random forest classifier is found out to be 0.9586220169361047

<img width="448" alt="image" src="https://user-images.githubusercontent.com/61600236/153957621-e24998c5-c14a-4619-9165-c6484f866dcf.png">

Figure 2.16 Random Forest Classifier

<img width="325" alt="image" src="https://user-images.githubusercontent.com/61600236/153957643-7639d61c-37d2-4d38-aa67-f3b7b24733d5.png">

Figure 2.17 Random Forest Classifier – Confusion Matrix, Classification Report & ROC Curve

<img width="233" alt="image" src="https://user-images.githubusercontent.com/61600236/153957665-4c63ceb4-b32c-4c35-923a-30cdf04ada7c.png">

Figure 2.18 Random Forest Classifier – Importance Feature

Fig 2.18 represents the importance feature for random forest model and its score with the bar plot of importance feature. The table below concludes all the finding from the models.

Model	                            Accuracy  F1-SCORE	  AUC
Decision Tree Classifier	        0.93	     0.94	      0.93
Gradient Boosting Classifier	    0.94	     0.95	      0.99
Logistic Regression Classifier	  0.67	     0.68	      0.75
Random Forest Classifier	        0.95	     0.96	      0.99

# IV.   CONCLUSION AND NEXT STEPS:
   The selected data set was checked for duplicates, renaming the column, checked for duplicates before starting the Exploratory Data Analysis. Anomalies and Outliers was found on flight distance. Various different machine learning models were used to predict the customer satisfaction in which we can find the best performing model for prediction. The Accuracy, f1- score, AUC values and importance features were found for each model. 
   
   Random Forest Classifier model has the highest accuracy among all the models with highest value for both f1- score and Area Under Curve value.
   
# REFERENCES
Bronshtein, A. (2019, October 29). A Quick Introduction to the "Pandas" Python Library. Medium. Retrieved from: https://towardsdatascience.com/a-quick-introduction-to-the-pandas-python-library-f1b678f34673

Htoon, K. S. (2020, July 3). A Guide to KNN Imputation. Medium. Retrieved from: https://medium.com/@kyawsawhtoon/a-guide-to-knn-imputation-95e2dc496e

Kaushik. (2020, July 20). KNNImputer: Way To Impute Missing Values. Analytics Vidhya. Retrieved from: https://www.analyticsvidhya.com/blog/2020/07/knnimputer-a-robust-way-to-impute-missing-values-using-scikit-learn/


10 minutes to pandas. 10 minutes to pandas - pandas 1.3.0 documentation. (n.d.). Retrieved from: https://pandas.pydata.org/docs/user_guide/10min.html#missing-data

Matplotlib Pyplot. (n.d.). Retrieved from: https://www.w3schools.com/python/matplotlib_pyplot.asp

Borcan, M. (2020, March 21). Decision tree classifiers explained. Medium. Retrieved from: https://medium.com/@borcandumitrumarius/decision-tree-classifiers-explained-e47a5b68477a

SalRite. (2018, December 18). Demystifying 'confusion matrix' confusion. Medium. Retrieved from:  https://towardsdatascience.com/demystifying-confusion-matrix-confusion-9e82201592fd

Sluijmers, M. (2020, July 9). Python (SCIKIT-LEARN): Logistic Regression classification. Medium. Retrieved from: https://towardsdatascience.com/python-scikit-learn-logistic-regression-classification-eb9c8de8938d

Nelson, D. (n.d.). Gradient boosting classifiers in Python with scikit-learn. Stack Abuse. Retrieved from: https://stackabuse.com/gradient-boosting-classifiers-in-python-with-scikit-learn

Donges, N. (n.d.). A complete guide to the random forest algorithm. Built In. Retrieved from: https://builtin.com/data-science/random-forest-algorithm


