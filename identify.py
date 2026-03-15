import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# read data into dataframe 
df = pd.read_csv("./GTZAN/features_30_sec.csv", usecols=range(1, 60)) # skip the first column (filename)
# data frame consists of all 59 features, including the label (genre)

# col_names = df.columns[range(21, 57)]
# df = df.drop(columns=col_names) 
# print(df)

feature_names = df.columns.tolist()
feature_names = feature_names[:-1] # remove the last column (label) from feature names
# print(feature_names)

# split data into features & target variable (genres/labels)
X = df.iloc[:, :-1].values # target
y = df.iloc[:, -1].values # features

# split the data into test group and train group 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# feature scale the data so they are on same scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) 

# build the classifier
randomforest = RandomForestClassifier(random_state=42)

# train the classifier
randomforest.fit(X_train, y_train)

# test the classifier
prediction = randomforest.predict(X_test)

# calculate the accuracy of the forest, and print
accuracy = accuracy_score(y_test, prediction)
print(accuracy)

feature_importances = randomforest.feature_importances_
plt.barh(feature_names, feature_importances)
plt.xlabel("Feature Importance")
plt.show()