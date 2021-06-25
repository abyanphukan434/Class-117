import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt  

df = pd.read_csv('heart.csv')

print(df.head())

age = df['age']

heart_attack = df['target']

age_train, age_test, heart_attack_train, heart_attack_test = train_test_split(age, heart_attack, test_size = 0.25, random_state = 0)

X = np.reshape(age_train.ravel(), (len(age_train), 1))

Y = np.reshape(heart_attack_train.ravel(), (len(heart_attack_train), 1))

classifier = LogisticRegression(random_state = 0)

classifier.fit(X, Y)

X_test = np.reshape(age_test.ravel(), (len(age_test), 1))

Y_test = np.reshape(heart_attack_test.ravel(), (len(heart_attack_test), 1))

heart_attack_prediction = classifier.predict(X_test)

predicted_values = []

for i in heart_attack_prediction:
  if i == 0:
    predicted_values.append('No')
  else:
    predicted_values.append('Yes')

actual_values = []

for i in Y_test.ravel():
  if i == 0:
    actual_values.append('No')
  else:
    actual_values.append('Yes')

labels = ["Yes", "No"]

cm = confusion_matrix(actual_values, predicted_values, labels)

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax)

ax.set_xlabel('Predicted')
ax.set_ylabel('Actual') 
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels)

accuracy = 36 + 16 / 36 + 16 + 17 + 7

accuracy = 52/76

print(accuracy)

factors = df[['age', 'sex', 'cp', 'chol', 'thalach']]

heart_attack = df['target']

factors_train, factors_test, heart_attack_train, heart_attack_test = train_test_split(factors, heart_attack, test_size = 0.25, random_state = 0)

sc_x = StandardScaler()

factors_train = sc_x.fit_transform(factors_train)

factors_test = sc_x.fit_transform(factors_test)

classifier2 = LogisticRegression(random_state = 0)

classifier2.fit(factors_train, heart_attack_train)

heart_attack_prediction_1 = classifier2.predict(factors_test)

predicted_values_1 = []

for i in heart_attack_prediction_1:
  if i == 0:
    predicted_values_1.append('No')
  else:
    predicted_values_1.append('Yes')

actual_values_1 = []

for i in Y_test.ravel():
  if i == 0:
    actual_values_1.append('No')
  else:
    actual_values_1.append('Yes')

labels = ["Yes", "No"]

cm = confusion_matrix(actual_values_1, predicted_values_1, labels)

ax= plt.subplot()

sns.heatmap(cm, annot=True, ax = ax)

ax.set_xlabel('Predicted')

ax.set_ylabel('Actual') 
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels)

accuracy = 33 + 23 / 33 + 23 + 10 + 10

accuracy = 56/76

print(accuracy)