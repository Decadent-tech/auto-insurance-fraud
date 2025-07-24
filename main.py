import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.model_selection import train_test_split
#import msno
import missingno as msno

import warnings
warnings.filterwarnings('ignore')
sns.set_style('dark')

plt.style.use('ggplot')
pd.set_option('display.max_columns', None)

#Import the dataset and first look of the dataset

data = pd.read_csv('dataset/insurance_claims (1).csv')
print(data.head())

#checking the data shape , and information and column details and describe its features 
print(data.shape)
print(data.info())
print(data.describe(include = 'all'))

#Checking the null values in the dataset
msno.matrix(data)
#plt.show()
#plt.savefig('EDA/missing_values_matrix.png', dpi=300, bbox_inches='tight')


#authorities_contacted has missing values 
print("Percentage of missing values in 'authorities_contacted':")
print((data['authorities_contacted'].isnull().sum()/ len(data))*100)


##Checking null values

print((data.isnull().sum()/data.shape[0])*100)

# dropping the '_c39' column as it has no information
data.drop(columns=['_c39'], inplace=True)

# Replacing values with '?' with nan values
data.replace('?', np.nan, inplace = True)
print(data.head())

# Checking the rows with  values in 
for col in data.columns:
    if data[col].dtype == 'object':
        print(f"Unique values in '{col}': {data[col].value_counts()}")

# checking the percentage of fraud reported   
print("Percentage of fraud reported:")
print(plt.pie(data['fraud_reported'].value_counts(), labels=data['fraud_reported'].unique(), autopct='%1.1f%%'))
plt.title('Fraud Reported Distribution')    
#plt.savefig('EDA/fraud_reported_distribution.png', dpi=300, bbox_inches='tight')
#plt.show()    

#Checking fraud accross gender distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='fraud_reported', hue='insured_sex', data=data, palette='Set1')
plt.title('Fraud Report distribution based on insured gender')
#plt.savefig('EDA/fraud Report distribution based on insured gender.png', dpi=300, bbox_inches='tight')
#plt.show()


#missing values
cols_with_na = [cols for cols in data.columns if data[cols].isnull().sum()>1]
for col in cols_with_na:
    print(col, np.round(data[col].isnull().mean(), 4), '% of missing values')
#authorities_contacted 0.091 % of missing values
#property_damage 0.36 % of missing values
#police_report_available 0.343 % of missing values

#see if missing value columns have any relationship with dependent variable
for col in cols_with_na:
    data1 = data.copy()
    
    data1[col]=np.where(data[col].isnull(),1,0)
    data1.groupby(col)['total_claim_amount'].median().plot.bar()
    plt.title(col)
    #plt.show()
    #plt.savefig(f'EDA/{col}_total_claim_amount_median.png', dpi=300, bbox_inches='tight')
    plt.close()

# imputing null values using mode
data['collision_type'] = data['collision_type'].fillna(data['collision_type'].mode()[0])
data['authorities_contacted'] = data['authorities_contacted'].fillna(data['authorities_contacted'].mode()[0])
data['property_damage'] = data['property_damage'].fillna(data['property_damage'].mode()[0])
data['police_report_available'] = data['police_report_available'].fillna(data['police_report_available'].mode()[0])

#final check on mulls 
print(data.isna().sum())

#| Feature                | Likely Reason to Drop                         |
#  ---------------------- | --------------------------------------------- |
#  `policy_number`        | ID column, no predictive power                |
#| `policy_bind_date`     | Date, unless doing time-based analysis        |
#| `policy_state`         | Might have low variance / irrelevant          |
#| `insured_zip`          | High cardinality, hard to generalize          |
#| `incident_location`    | High cardinality                              |
#| `incident_date`        | Might leak label info                         |
#| `incident_state/city`  | Sparse or high cardinality                    |
#| `insured_hobbies`      | Possibly noisy/unstructured                   |
#| `auto_make/model/year` | High cardinality, complex unless modeled well |
#| `policy_csl`           | Might be redundant with policy coverage vars  |

# Dropping columns that are not useful for analysis

to_drop = ['policy_number','policy_bind_date','policy_state','insured_zip','incident_location','incident_date',
           'incident_state','incident_city','insured_hobbies','auto_make','auto_model','auto_year','policy_csl']

data.drop(to_drop, inplace = True, axis = 1)

data.drop(columns = ['age', 'total_claim_amount'], inplace = True, axis = 1)


data["total_claim"] = data["injury_claim"] + data["property_claim"] + data["vehicle_claim"]
data.drop(["injury_claim", "property_claim", "vehicle_claim"], axis=1, inplace=True)


# extracting categorical columns
cat_df = data.select_dtypes(include = ['object'])


#using get_dummies for categorical variables
cat_df = pd.get_dummies(cat_df, drop_first = True)
cat_df = cat_df.astype(int)

# extracting the numerical columns
num_df = data.select_dtypes(include = ['int64'])

# checking for multicollinearity
plt.figure(figsize = (18, 12))
corr = num_df.corr()
mask = np.triu(np.ones_like(corr, dtype = bool))
sns.heatmap(data = corr, mask = mask, annot = True, fmt = '.2g', linewidth = 1)
#plt.show()
#plt.savefig('EDA/correlation_heatmap.png', dpi=300, bbox_inches='tight')


# combining the Numerical and Categorical dataframes to get the final dataset
combined_df = pd.concat([num_df, cat_df], axis = 1)
print(combined_df.columns)
# Model building

X = combined_df.drop(columns=['fraud_reported_Y'], axis=1)
y = combined_df['fraud_reported_Y']
# Plotting the distribution of each feature
plt.figure(figsize = (25, 20))
plotnumber = 1
for col in X.columns:
    if plotnumber <= 24:
        ax = plt.subplot(5, 5, plotnumber)
        sns.distplot(X[col])
        plt.xlabel(col, fontsize = 15)
        
    plotnumber += 1
    
plt.tight_layout()
#plt.show()
#plt.savefig('EDA/distribution_plots.png', dpi=300, bbox_inches='tight')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
# Scaling the numeric values in the dataset

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(X_train)
X_train = pd.DataFrame(scaled_data, columns=X_train.columns)
print(X_train.head())
print("----------------------------------------------------")
print(y_train.head())

#Model classifier SVM 

from sklearn.svm import SVC

svc = SVC()
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
svc_test_acc = svc.score(X_test, y_test)
print(f"Test accuracy of SVM is : {svc_test_acc}")
# Visualizing the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix for SVM Classifier')
plt.xlabel('Predicted')
plt.ylabel('Actual')
#plt.savefig('EDA/confusion_matrix_svm.png', dpi=300, bbox_inches='tight')
#plt.show()


# using DT since class is imbalanced Decision Tree classifier model
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
cm = classification_report(y_test, y_pred)
print(cm)
# Visualizing the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix for DT Classifier')
plt.xlabel('Predicted')
plt.ylabel('Actual')
#plt.savefig('EDA/confusion_matrix_dtc.png', dpi=300, bbox_inches='tight')
#plt.show()


# using Hyperparameter tuning for Decision Tree Classifier
# hyper parameter tuning with parameters

from sklearn.model_selection import GridSearchCV

grid_params = {
    'criterion' : ['gini', 'entropy'],
    'max_depth' : [3, 5, 7, 10],
    'min_samples_split' : range(2, 10, 1),
    'min_samples_leaf' : range(2, 10, 1)
}

grid_search = GridSearchCV(dtc, grid_params, cv = 5, n_jobs = -1, verbose = 1)
grid_search.fit(X_train, y_train)

# best parameters and best score

print(grid_search.best_params_)#Fitting 5 folds for each of 512 candidates, totalling 2560 fits
#{'criterion': 'gini', 'max_depth': 3, 'min_samples_leaf': 4, 'min_samples_split': 3}
print(grid_search.best_score_)#0.8

# best estimator 

dtc = grid_search.best_estimator_
y_pred = dtc.predict(X_test)
from sklearn.metrics import accuracy_score
dtc_train_acc = accuracy_score(y_train, dtc.predict(X_train))
print(f"Training accuracy of Decision Tree is : {dtc_train_acc}")
dtc_test_acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy of Decision Tree is : {dtc_test_acc}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
# Visualizing the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix for DT Classifier after Hyperparameter Tuning')
plt.xlabel('Predicted')
plt.ylabel('Actual')
#plt.savefig('EDA/confusion_matrix_dtc_tuned.png', dpi=300, bbox_inches='tight')
#plt.show()

#Random Forest classifier
from sklearn.ensemble import RandomForestClassifier

rand_clf = RandomForestClassifier(criterion= 'entropy', max_depth= 10, max_features= 'sqrt', min_samples_leaf= 1, min_samples_split= 3, n_estimators= 140)
rand_clf.fit(X_train, y_train)

y_pred = rand_clf.predict(X_test)
rand_clf_train_acc = accuracy_score(y_train, rand_clf.predict(X_train))
rand_clf_test_acc = accuracy_score(y_test, y_pred)

print(f"Training accuracy of Random Forest is : {rand_clf_train_acc}")
print(f"Test accuracy of Random Forest is : {rand_clf_test_acc}")

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# Visualizing the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix for RF Classifier ')
plt.xlabel('Predicted')
plt.ylabel('Actual')
#plt.savefig('EDA/confusion_matrix_rf_tuned.png', dpi=300, bbox_inches='tight')
#plt.show()

#Adaboost classifier
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier()

parameters = {
    'n_estimators' : [50, 70, 90, 120, 180, 200],
    'learning_rate' : [0.001, 0.01, 0.1, 1, 10],
    'algorithm' : ['SAMME', 'SAMME.R']
}

grid_search = GridSearchCV(ada, parameters, n_jobs = -1, cv = 5, verbose = 1)
grid_search.fit(X_train, y_train)
# best parameter and best score

print(grid_search.best_params_)
print(grid_search.best_score_)
# best estimator 

ada = grid_search.best_estimator_

y_pred = ada.predict(X_test)
ada_train_acc = accuracy_score(y_train, ada.predict(X_train))
ada_test_acc = accuracy_score(y_test, y_pred)

print(f"Training accuracy of Ada Boost is : {ada_train_acc}")
print(f"Test accuracy of Ada Boost is : {ada_test_acc}")

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# Visualizing the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix for ADABOOST Classifier ')
plt.xlabel('Predicted')
plt.ylabel('Actual')
#plt.savefig('EDA/confusion_matrix_ada_boost.png', dpi=300, bbox_inches='tight')
#plt.show()

#Model: Gradient boosting
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of gradient boosting classifier

gb_acc = accuracy_score(y_test, gb.predict(X_test))

print(f"Training Accuracy of Gradient Boosting Classifier is {accuracy_score(y_train, gb.predict(X_train))}")
print(f"Test Accuracy of Gradient Boosting Classifier is {gb_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, gb.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, gb.predict(X_test))}")
# Visualizing the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix for GBDT Classifier ')
plt.xlabel('Predicted')
plt.ylabel('Actual')
#plt.savefig('EDA/confusion_matrix_gbdt.png', dpi=300, bbox_inches='tight')
#plt.show()

#Stochastic Gradient Boosting (SGB)
sgb = GradientBoostingClassifier(subsample = 0.90, max_features = 0.70)
sgb.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of stochastic gradient boosting classifier
sgb_acc = accuracy_score(y_test, sgb.predict(X_test))
print(f"Training Accuracy of Stochastic Gradient Boosting is {accuracy_score(y_train, sgb.predict(X_train))}")
print(f"Test Accuracy of Stochastic Gradient Boosting is {sgb_acc} \n")
print(f"Confusion Matrix :- \n{confusion_matrix(y_test, sgb.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, sgb.predict(X_test))}")

# Visualizing the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix for SGBT Classifier ')
plt.xlabel('Predicted')
plt.ylabel('Actual')
#plt.savefig('EDA/confusion_matrix_sgbt.png', dpi=300, bbox_inches='tight')
#plt.show()

#Extra Trees Classifier
from sklearn.ensemble import ExtraTreesClassifier

etc = ExtraTreesClassifier()
etc.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of extra trees classifier

etc_acc = accuracy_score(y_test, etc.predict(X_test))

print(f"Training Accuracy of Extra Trees Classifier is {accuracy_score(y_train, etc.predict(X_train))}")
print(f"Test Accuracy of Extra Trees Classifier is {etc_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, etc.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, etc.predict(X_test))}")

# Visualizing the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix for etc Classifier ')
plt.xlabel('Predicted')
plt.ylabel('Actual')
#plt.savefig('EDA/confusion_matrix_etc.png', dpi=300, bbox_inches='tight')
#plt.show()

#Voting Classifier
from sklearn.ensemble import VotingClassifier

classifiers = [('Support Vector Classifier', svc),  ('Decision Tree', dtc), ('Random Forest', rand_clf),
               ('Ada Boost', ada), ('Gradient Boosting Classifier', gb), ('SGB', sgb),
              ('Extra Trees Classifier', etc)]

vc = VotingClassifier(estimators = classifiers)
vc.fit(X_train, y_train)

y_pred = vc.predict(X_test)
# accuracy_score, confusion_matrix and classification_report

vc_train_acc = accuracy_score(y_train, vc.predict(X_train))
vc_test_acc = accuracy_score(y_test, y_pred)
print(f"Training accuracy of Voting Classifier is : {vc_train_acc}")
print(f"Test accuracy of Voting Classifier is : {vc_test_acc}")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
#creating dataframe for models and obtained scores
models = pd.DataFrame({
    'Model' : ['SVC', 'Decision Tree', 'Random Forest','Ada Boost', 'Gradient Boost', 'SGB', 'Extra Trees', 'Voting Classifier'],
    'Score' : [svc_test_acc, dtc_test_acc, rand_clf_test_acc, ada_test_acc, gb_acc, sgb_acc, etc_acc, vc_test_acc]
})
print(models.sort_values(by = 'Score', ascending = False))
#Using bar plot, comparing the obtained test accuracies of ML models used
fig = px.bar(data_frame = models, x = 'Score', y = 'Model', color = 'Score', template = 'plotly_dark', 
       title = 'Comparison of Models')
# Save the plot
fig.write_image('EDA/model_comparison.png', width=800, height=600, scale=2)

import joblib
joblib.dump(etc, 'model/best_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')


