####################################### Naive Bayes Assignment Submission #################################
"""

Please ensure you update all the details:
    
Name:        CH Pavan kumar

Batch Id:    15 Sep 2023
_
Topic:      Naive Bayes Assignment 

"""
'''CRISP-ML(Q)

a. Business & Data Understanding
   In this scenario, we're tasked with building a classification model to predict salary based on a given dataset. 
   The dataset likely contains various features such as education level, years of experience, job title, etc.,
   which will be used to predict whether an individual's salary falls into a particular category or range.
   The goal of this project is to create a model that can accurately predict salary levels based on these features. 
   This model can be useful for a variety of stakeholders such as:
      1. Human Resources Departments
      2. Job Seekers
      3. Companies
      4. Recruitment Agencies
       
      i. Business Objective - Predict salary levels accurately using the Naive Bayes algorithm.
     ii. Business Constraint - Minimize misclassification errors while maintaining model simplicity and interpretability.
     
     Success Criteria:
         1.Business Success Criteria - Increase the accuracy of salary predictions by 10% compared to previous methods,
                                       leading to better budget planning and resource allocation.
                                       
         2.ML Success Criteria -       Achieve an accuracy of over 85% in predicting salary levels using the Naive Bayes algorithm,
                                       with a precision and recall of at least 80%, ensuring reliable predictions for stakeholders.
                                       
         3.Economic Success Criteria - By accurately predicting salary levels and aligning them with industry standards, 
                                       achieve a cost savings of $100,000 to $120,000 annually by optimizing salary negotiations and reducing turnover costs.
                                       
     Data Collection : 
         
         
     Metadata Description:

     age: Age of the individual.
     
     workclass: The type of employment (e.g., private, government, self-employed).
     
     education: The highest level of education achieved by the individual.
     
     educationno: The numerical representation of education level.
     
     maritalstatus: The marital status of the individual (e.g., married, single, divorced).
     
     occupation: The type of job or occupation held by the individual.
     
     relationship: The relationship status of the individual (e.g., husband, wife, own-child).
     
     race: The race of the individual.
     
     sex: The gender of the individual.
     
     capitalgain: The amount of capital gains earned by the individual.
     
     capitalloss: The amount of capital losses incurred by the individual.
     
     hoursperweek: The number of hours worked per week.
     
     native: The country of origin or nationality of the individual.
     
     Salary: The target variable indicating the salary level of the individual (e.g., <=50K, >50K).

'''
# Code Modularity should be maintained

# Import all required libraries and modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB , BernoulliNB
from sklearn.metrics import accuracy_score, classification_report
import sklearn.metrics as skmit
# combine both Load both test   and train data
test = pd.read_csv(r"/Users/pavankumar/Documents/My Learning /Data Sciences/Naive Bayes Assignment sub/SalaryData_Train.csv")
train = pd.read_csv(r"/Users/pavankumar/Documents/My Learning /Data Sciences/Naive Bayes Assignment sub/SalaryData_Test.csv")

# Combined both data rows togethier
data = pd.concat([train, test], axis = 0)
data.to_csv("data.csv")

# Retrive the info of both datasets
data.info()
categ = data.select_dtypes(include = ['object']).columns
categ.size


# remove unwanted columns
##############################################
data['workclass'].unique().size            # 1 noise
data['education'].unique().size            # 2 
data['maritalstatus'].value_counts().size  # 3 noise
data['occupation'].unique().size           # 4
data['relationship'].unique().size         # 5
data['race'].value_counts().size           # 6 little noise
data['sex'].value_counts().size            # 7
data['native'].unique().size               # 8 noise
data['Salary'].unique()                    # 9


""" as we see this both columns represent same vales in different such as education for categorial and 
 educationno for in ordinal, so we can remove the categorical no to reduce noice"""
 
# removing noise data
data.drop(['maritalstatus','native','workclass','race'], axis = 1, inplace = True)
#########################################################
# check imbalance on the salary column
data['Salary'].value_counts()
data['Salary'].value_counts() / len(data['Salary']) # In Percentage

# Mapping the Salary to numeric values 1 and 0. 
# This step is required for metric calculations in model evaluation phase.
data['Salary_numerical'] = data['Salary'].map({' <=50K': 0, ' >50K': 1})

# Check the converted values
print(data['Salary_numerical'].value_counts())
###############################################################################
# My Sql
from sqlalchemy import create_engine
  
# Creating engine which connect to Mysql

engine = ("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user = "root", # user
                               pw = "98486816", # password
                               db = "Clustering")) # database

db = create_engine(engine)

data.to_sql('Salary_Navie_Bayes', con = db, if_exists = 'replace', index = False)


###############################################################################
# Select query
sql = 'SELECT * from Clustering.Salary_Navie_Bayes'
salary_data = pd.read_sql_query(sql, db)
# preprocessing : Convert categorical variables into numericalusing one_hot_encoding
data = pd.get_dummies(data, columns = ['education','sex','relationship','occupation'])


# Split the data into train and test sets
# from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.2, stratify = data['Salary_numerical'], random_state=42)

# Split the data into features and Target variables
x = train.drop(['Salary','Salary_numerical'], axis = 1)
y = train['Salary_numerical']

y.value_counts() /  len(y)# In Percentage imbalanced

x1 = test.drop(['Salary','Salary_numerical'], axis = 1)
#y1 = test['Salary_numerical']
#y1.value_counts() /  len(y1) In Percentage imbalanced

# Smote
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state = 0)

# Transform the dataset
x_train, y_train = smote.fit_resample(x , y)

y_train.unique()
y_train.values.sum() # Number of "1"s
y_train.size - y_train.values.sum() # Number of "0" s
# The data is now balanced
y_train.value_counts() / len(y_train) # In Percentage balanced 50 - 50 
# Model Building

# Initializer the Navie Bayes Classifer
# from sklearn.naive_bayes import MultinomialNB, GaussianNB
mb_classifier = MultinomialNB(alpha = 5) # we see the accuracy is very low
nb_classifier = GaussianNB()
# Train the classifier 

nb_classifier.fit(x_train, y_train) # trained on train datasets

# Prediction
y_pred_train_nb = nb_classifier.predict(x_train)
y_pred_test_nb = nb_classifier.predict(x1)    # predicted on test data sets

pd.crosstab(y_pred_train_nb, y_train)
pd.crosstab(y_pred_test_nb, test['Salary_numerical'])    # Cross checked


# Model Accuracy
# from sklearn.metrics import accuracy_score, classification_report
train_accuracy = accuracy_score(y_pred_train_nb, y_train)
train_accuracy

test_accuracy = accuracy_score(y_pred_test_nb, test['Salary_numerical'])   # Accuracy for the test data
test_accuracy

# alternative 
test_accuracy1 = np.mean(test['Salary_numerical'] == y_pred_test_nb)

print("\n\nTraining Accuracy:", train_accuracy)
print("\n\nTest Accuracy:", test_accuracy)

# Classification report for test set
print("\n\nClassification Report for Test Set:")
print(classification_report(test['Salary_numerical'], y_pred_test_nb))

# metrics
print("accuracy : %.2f,  \nsensitivity : %.2f, \nspecificity %.2f, \nPrecision : %.2f" %
       (skmit.accuracy_score(test['Salary_numerical'],y_pred_test_nb.ravel()),
       skmit.recall_score(test['Salary_numerical'],y_pred_test_nb.ravel()),
       skmit.recall_score(test['Salary_numerical'],y_pred_test_nb.ravel(), pos_label = 0),
       skmit.precision_score(test['Salary_numerical'],y_pred_test_nb)))

# Confusion Matrix - heat - map
cm = skmit.confusion_matrix(test['Salary_numerical'],y_pred_test_nb)
cmplot = skmit.ConfusionMatrixDisplay(cm, display_labels = ['less than <=50K', 'more than >50K'])
cmplot.plot()
cmplot.ax_.set(title = "Salary detection of Confusion Matrix",
               xlabel = "Predicted values", ylabel = "Actual values")


# Saving the Best model using piple line
from imblearn.pipeline import make_pipeline
# Building the Pipeline
# Preparing a naive bayes model on training data set 

nb = GaussianNB()

# Defining Pipeline
pipe1 = make_pipeline(smote, nb)

# fit to train data
processed = pipe1.fit(x, y)

# Save the model for prediction
import joblib
joblib.dump(processed, "processed1")

# lodel the saved model for prediction
model = joblib.load("processed1")
model

# Predictions
pres_test = model.predict(x1)

# Evalution on test data with metrics
pd.crosstab(pres_test, test['Salary_numerical'])

# Accuracy
accuracy_score(pres_test, test['Salary_numerical'])

# # metrics
print("accuracy : %.2f,  \nsensitivity : %.2f, \nspecificity %.2f, \nPrecision : %.2f" %
       (skmit.accuracy_score(pres_test, test['Salary_numerical'].ravel()),
       skmit.recall_score(pres_test, test['Salary_numerical'].ravel()),
       skmit.recall_score(pres_test, test['Salary_numerical'].ravel(), pos_label = 0),
       skmit.precision_score(pres_test, test['Salary_numerical'])))

# Confusion Matrix - heat - map
cm = skmit.confusion_matrix(pres_test, test['Salary_numerical'])
cmplot = skmit.ConfusionMatrixDisplay(cm, display_labels = ['less than <=50K', 'more than >50K'])
cmplot.plot()
cmplot.ax_.set(title = "Salary detection of Confusion Matrix",
               xlabel = "Predicted values", ylabel = "Actual values")