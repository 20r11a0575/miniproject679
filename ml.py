from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, roc_curve, auc, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_curve, auc
import time
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

file_path = 'projects/dataset-derilium.csv'
#  file.save(file_path)


def generate_kde_plot(feature, xlim_min, xlim_max, df):
  data_group1 = df[df['Group'] == 1][feature].dropna()
  data_group0 = df[df['Group'] == 0][feature].dropna()

  # Create KDE estimations for each group
  kde_group1 = gaussian_kde(data_group1)
  kde_group0 = gaussian_kde(data_group0)

  # Define x values for the plot
  x = np.linspace(xlim_min, xlim_max, 1000)

  # Plot KDE for Group 1
  plt.plot(x, kde_group1(x), label='Group 1', color='blue')

  # Plot KDE for Group 0
  plt.plot(x, kde_group0(x), label='Group 0', color='red')

  # Set labels, legend, and title
  plt.xlabel(feature)
  plt.ylabel('Density')
  plt.legend()
  plt.title(f'KDE Plot for {feature}')

  # Save the plot as an image
  img_name = f'{feature}_plot.png'
  plt.savefig(f'static/{img_name}')

  return img_name


# Load the CSV file into a Pandas DataFrame
df = pd.read_csv(file_path)
df = df.loc[df['Visit'] == 1]
df = df.reset_index(drop=True)
df['M/F'] = df['M/F'].replace(['F', 'M'], [0, 1])
df['Group'] = df['Group'].replace(['Converted'], ['Demented'])
df['Group'] = df['Group'].replace(['Demented', 'Nondemented'], [1, 0])
df = df.drop(['MRI ID', 'Visit', 'Hand'], axis=1)
Demented = df[df['Group'] == 1]['M/F'].value_counts()
Nondemented = df[df['Group'] == 0]['M/F'].value_counts()
df_bar = pd.DataFrame([Demented, Nondemented])
df_bar.index = ['Demented', 'Nondemented']
ax = df_bar.plot(kind='bar', stacked=True, figsize=(8, 5))
plt.xlabel('Group')
plt.ylabel('Number of patients')
plt.legend()
plt.title('Gender and Demented rate')
# Save the chart as an image
img = BytesIO()
plt.savefig(img, format='png')
img.seek(0)
chart_url = base64.b64encode(img.getvalue()).decode()
img.close()

# Your data processing code
img_name0 = generate_kde_plot('MMSE', 15, 30, df)
img_name1 = generate_kde_plot('ASF', 0.5, 2, df)
img_name2 = generate_kde_plot('eTIV', 900, 2100, df)
img_name3 = generate_kde_plot('nWBV', 0.6, 0.9, df)
img_name4 = generate_kde_plot('Age', 50, 100, df)
img_name5 = generate_kde_plot('EDUC', df['EDUC'].min(), df['EDUC'].max(), df)
pd.isnull(df).sum()
df_dropna = df.dropna(axis=0, how='any')
pd.isnull(df_dropna).sum()
df_dropna['Group'].value_counts()
x = df['EDUC']
y = df['SES']

# Remove rows with missing SES values
ses_not_null_index = y[~y.isnull()].index
x = x[ses_not_null_index]
y = y[ses_not_null_index]

# Fit a trend line
z = np.polyfit(x, y, 1)
p = np.poly1d(z)

# Create the scatter plot with the trend line
plt.plot(x, y, 'go', x, p(x), "r--")
plt.xlabel('Education Level (EDUC)')
plt.ylabel('Social Economic Status (SES)')
plt.savefig('scatter_plot.png')
df.groupby(['EDUC'])['SES'].median()
df["SES"].fillna(df.groupby("EDUC")["SES"].transform("median"), inplace=True)
pd.isnull(df['SES']).value_counts()
# Dataset with imputation
Y = df['Group'].values  # Target for the model
X = df[['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV',
        'ASF']]  # Features we use

# splitting into three sets
X_trainval, X_test, Y_trainval, Y_test = train_test_split(X,
                                                          Y,
                                                          test_size=0.25,
                                                          random_state=0)

# Feature scaling
scaler = MinMaxScaler().fit(X_trainval)
X_trainval_scaled = scaler.transform(X_trainval)
X_test_scaled = scaler.transform(X_test)
# Dataset after dropping missing value rows
Y = df_dropna['Group'].values  # Target for the model
X = df_dropna[['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV',
               'ASF']]  # Features we use

# splitting into three sets
X_trainval_dna, X_test_dna, Y_trainval_dna, Y_test_dna = train_test_split(
    X, Y, test_size=0.25, random_state=0)

# Feature scaling
scaler = MinMaxScaler().fit(X_trainval_dna)
X_trainval_scaled_dna = scaler.transform(X_trainval_dna)
X_test_scaled_dna = scaler.transform(X_test_dna)
acc = []  # list to store all performance metric
start_time = time.time()
df_eval = df_dropna.copy()
X_eval_scaled = scaler.transform(
    df_eval[['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']])

# Dataset after dropping missing value rows
best_score = 0
kfolds = 5  # set the number of folds
best_parameters = 0
for c in [0.001, 0.1, 1, 10, 100]:
  logRegModel = LogisticRegression(C=c)
  # perform cross-validation
  scores = cross_val_score(logRegModel,
                           X_trainval_scaled_dna,
                           Y_trainval_dna,
                           cv=kfolds,
                           scoring='accuracy')

  # compute mean cross-validation accuracy
  score = np.mean(scores)

  # Find the best parameters and score
  if score > best_score:
    best_score = score
    best_parameters = c

  # rebuild a model on the combined training and validation set
SelectedLogRegModel = LogisticRegression(C=best_parameters).fit(
    X_trainval_scaled_dna, Y_trainval_dna)

# predict probabilities on the test set
probs = SelectedLogRegModel.predict_proba(X_test_scaled_dna)[:, 1]

# calculate mean squared error (MSE)
mse = mean_squared_error(Y_test_dna, probs)

# calculate root mean squared error (RMSE)
rmse = np.sqrt(mse)

# calculate recall, fpr, tpr, and auc
PredictedOutput = SelectedLogRegModel.predict(X_test_scaled_dna)
test_recall = recall_score(Y_test_dna, PredictedOutput, pos_label=1)
fpr, tpr, thresholds = roc_curve(Y_test_dna, probs, pos_label=1)
test_auc = auc(fpr, tpr)
df_eval['Predicted_Group'] = SelectedLogRegModel.predict(X_eval_scaled)
end_time = time.time()
cpu_time = end_time - start_time
error_rate = 1 - SelectedLogRegModel.score(X_test_scaled_dna, Y_test_dna)
#display(df)
#print(df_eval)#.to_string(index=False))
print('Logistic Regression Model:')
print("Best accuracy on validation set is:", best_score)
print("Best parameter for regularization (C) is: ", best_parameters)
print("Test accuracy with best C parameter is",
      SelectedLogRegModel.score(X_test_scaled_dna, Y_test_dna))
print("Test recall with the best C parameter is", test_recall)
print("Test AUC with the best C parameter is", test_auc)
print("RMSE with the best C parameter is", rmse)
print("Error Rate is ", error_rate)
print("CPU time is", cpu_time)

m = 'Logistic Regression (w/ dropna)'
acc.append([
    m,
    SelectedLogRegModel.score(X_test_scaled_dna, Y_test_dna), test_recall,
    test_recall, fpr, tpr, thresholds
])
start_time = time.time()
best_score = 0  # Initialize best_score to 0 or any other suitable default value
best_parameter_c = 0  # Initialize best_parameter_c
best_parameter_gamma = ''  # Initialize best_parameter_gamma
best_parameter_k = ''  # Initialize best_parameter_k

for c_paramter in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
  for gamma_paramter in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    for k_parameter in ['rbf', 'linear', 'poly', 'sigmoid']:
      svmModel = SVC(kernel=k_parameter, C=c_paramter, gamma=gamma_paramter)
      scores = cross_val_score(svmModel,
                               X_trainval_scaled,
                               Y_trainval,
                               cv=kfolds,
                               scoring='accuracy')
      score = np.mean(scores)
      if score > best_score:
        best_score = score
        best_parameter_c = c_paramter
        best_parameter_gamma = gamma_paramter
        best_parameter_k = k_parameter

SelectedSVMmodel = SVC(C=best_parameter_c,
                       gamma=best_parameter_gamma,
                       kernel=best_parameter_k).fit(X_trainval_scaled,
                                                    Y_trainval)

test_score = SelectedSVMmodel.score(X_test_scaled, Y_test)
PredictedOutput = SelectedSVMmodel.predict(X_test_scaled)
test_recall = recall_score(Y_test, PredictedOutput, pos_label=1)
fpr, tpr, thresholds = roc_curve(Y_test, PredictedOutput, pos_label=1)
test_auc = auc(fpr, tpr)

end_time = time.time()
cpu_time = end_time - start_time
error_rate = 1 - test_score
print('SVM..:')
print("Best accuracy on cross-validation set is:", best_score)
print("Best parameter for C is:", best_parameter_c)
print("Best parameter for gamma is:", best_parameter_gamma)
print("Best parameter for kernel is:", best_parameter_k)
print("Test accuracy with the best parameters is", test_score)
print("Test recall with the best parameters is", test_recall)
print("Test AUC with the best parameters is:", test_auc)
print("Error Rate is", error_rate)
print("CPU time is", cpu_time)
m = 'SVM'
acc.append([m, test_score, test_recall, test_auc, fpr, tpr, thresholds])
start_time = time.time()
best_score = 0
best_M = 0
best_d = 0
best_m = 0
M = 0
d = 0
for M in range(2, 15, 2):  # combines M trees
  for d in range(1, 9):  # maximum number of features considered at each split
    for m in range(1, 9):  # maximum depth of the tree
      # train the model
      # n_jobs(4) is the number of parallel computing
      forestModel = RandomForestClassifier(n_estimators=M,
                                           max_features=d,
                                           n_jobs=4,
                                           max_depth=m,
                                           random_state=0)

      # perform cross-validation
      scores = cross_val_score(forestModel,
                               X_trainval_scaled,
                               Y_trainval,
                               cv=kfolds,
                               scoring='accuracy')

      # compute mean cross-validation accuracy
      score = np.mean(scores)

      # if we got a better score, store the score and parameters
      if score > best_score:
        best_score = score
        best_M = M
        best_d = d
        best_m = m

# Rebuild a model on the combined training and validation set
SelectedRFModel = RandomForestClassifier(n_estimators=M,
                                         max_features=d,
                                         max_depth=m,
                                         random_state=0).fit(
                                             X_trainval_scaled, Y_trainval)

PredictedOutput = SelectedRFModel.predict(X_test_scaled)
test_score = SelectedRFModel.score(X_test_scaled, Y_test)
test_recall = recall_score(Y_test, PredictedOutput, pos_label=1)
fpr, tpr, thresholds = roc_curve(Y_test, PredictedOutput, pos_label=1)
test_auc = auc(fpr, tpr)

end_time = time.time()
cpu_time = end_time - start_time
error_rate = 1 - test_score
print('Random Forest Classifier...:')
print("Best accuracy on validation set is:", best_score)
print("Best parameters of M, d, m are: ", best_M, best_d, best_m)
print("Test accuracy with the best parameters is", test_score)
print("Test recall with the best parameters is:", test_recall)
print("Test AUC with the best parameters is:", test_auc)
print("Error Rate is", error_rate)
print("CPU time is", cpu_time)

m = 'Random Forest'
acc.append([m, test_score, test_recall, test_auc, fpr, tpr, thresholds])

# Prepare input data

# ..x`1
