import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from EDA import EDA

# import the dataset
df = pd.read_csv('appdata10.csv')

# EDA
df = EDA.eda(df)

# =============================================================================
# Data Pre-Processing
# =============================================================================

# Splitting Independent and Response Variables
y = df["enrolled"]
X = df.drop(columns="enrolled")

# Splitting the df into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Removing Id('user')
train_id = X_train['user']
X_train = X_train.drop(columns = ['user'])
test_id = X_test['user']
X_test = X_test.drop(columns = ['user'])

# Feature Scaling
sc = StandardScaler()

X_train2 = pd.DataFrame(sc.fit_transform(X_train))
X_train2.columns = X_train.columns.values
X_train2.index = X_train.index.values
X_train = X_train2

X_test2 = pd.DataFrame(sc.transform(X_test))
X_test2.columns = X_test.columns.values
X_test2.index = X_test.index.values
X_test = X_test2


# =============================================================================
# Model Building
# =============================================================================

# Fitting Model to the Training Set
# screens can be correlated to each other
# L1 (Lasso) is a penalty for any particular field that is correlated to y (df["enrolled"])
model = LogisticRegression(penalty = 'l1', random_state = 0, solver = 'saga')
model.fit(X_train, y_train)
# Predicting Test Set
y_pred = model.predict(X_test)

# Evaluating Results
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred) # tp / (tp + fp)
recall_score(y_test, y_pred) # tp / (tp + fn)
f1_score(y_test, y_pred)

df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, fmt='g')
print("Test Set Accuracy: %0.4f" % accuracy_score(y_test, y_pred))


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10)
print("Accuracy with CV: %0.4f (+/- %0.4f)" % (accuracies.mean(), accuracies.std()*2))


# =============================================================================
# Model Tuning
# =============================================================================

## Grid Search (Round 1)

# Select Regularization Method
penalty = ['l1', 'l2']

# Create regularization hyperparameter space
C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

# Combine Parameters
parameters = {'C': C, 'penalty': penalty}

grid_search = GridSearchCV(estimator = model, param_grid = parameters, 
                           scoring = "accuracy", cv = 10, n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)


best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

## Grid Search (Round 2)

# Select Regularization Method
penalty = ['l1', 'l2']

# Create regularization hyperparameter space
C = [0.1, 0.5, 0.9, 1, 2, 5]

# Combine Parameters
parameters = {'C': C, 'penalty': penalty}

grid_search = GridSearchCV(estimator = model, param_grid = parameters, 
                           scoring = "accuracy", cv = 10, n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: %0.4f,\nBest Parameters: C = %0.1f,  Penalty = %s" 
      %(best_accuracy, best_parameters['C'], best_parameters['penalty']))


# =============================================================================
# Results
# =============================================================================

# Formatting Final Results
final_results = pd.concat([y_test, test_id], axis = 1).dropna()
final_results['predicted_reach'] = y_pred
final_results = final_results[['user', 'enrolled', 'predicted_reach']].reset_index(drop=True)
final_results.to_csv('final_result.csv', index = False)


