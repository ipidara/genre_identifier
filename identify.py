import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier

def random_forest(df):
    feature_names = df.columns.tolist()
    feature_names = feature_names[:-1] # remove the last column (label) from feature names
    # print(feature_names)

    # split data into features & target variable (genres/labels)
    X = df.iloc[:, :-1].values # features
    y = df.iloc[:, -1].values # target

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
    print("Accuracy:", accuracy)

    # create confusion matrix and display
    confusion = confusion_matrix(y_test, prediction)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=randomforest.classes_)
    disp.plot()
    plt.xticks(rotation=90)
    plt.show()

    # create feature importances graph, and 
    feature_importances = randomforest.feature_importances_

    # sort features by importance
    indices = np.argsort(feature_importances)

    sorted_features_names = np.array(feature_names)[indices]
    sorted_importances = feature_importances[indices]

    plt.figure(figsize=(10, 16))
    plt.barh(sorted_features_names, sorted_importances)
    plt.xlabel("Feature Importance", fontsize=14)
    plt.ylabel("Feature", fontsize=14)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    plt.show()

# read data into dataframe 
df = pd.read_csv("./GTZAN/features_30_sec.csv")
# data frame consists of the name of the file and all 59 features, including the label (genre)

# make a copy of datafrom to remove repeated values
df_removals = df.copy()

# df_removals = df_removals[df_removals["filename"] != "disco.00051.wav"]
removal_files = ["disco.00051.wav", "disco.00070.wav", "disco.00060.wav", "disco.00089.wav", "disco.00074.wav", "disco.00099.wav", 
                 "hiphop.00045.wav", "hiphop.00078.wav", 
                 "jazz.00051.wav", "jazz.00053.wav", "jazz.00055.wav", "jazz.00058.wav", "jazz.00060.wav", "jazz.00062.wav", "jazz.00065.wav", "jazz.00067.wav", "jazz.00068.wav", "jazz.00069.wav", "jazz.00070.wav", "jazz.00071.wav", "jazz.00072.wav", 
                 "metal.00013.wav", "metal.00094.wav", "metal.00061.wav", "metal.00062.wav", "metal.00063.wav", "metal.00064.wav", "metal.00065.wav", "metal.00066.wav", "metal.00058.wav",
                 "pop.00022.wav", "pop.00031.wav", "pop.00046.wav", "pop.00080.wav", "pop.00057.wav", "pop.00060.wav", "pop.00059.wav", "pop.00071.wav", "pop.00090.wav", 
                 "reggae.00054.wav", "reggae.00056.wav", "reggae.00057.wav", "reggae.00060.wav", "reggae.00058.wav", "reggae.00069.wav", "reggae.00074.wav", "reggae.00081.wav", "reggae.00082.wav", "reggae.00091.wav", "reggae.00092.wav", 
                 "rock.00016.wav"]

# remove all the repeated audio files
df_removals = df_removals[~df_removals["filename"].isin(removal_files)]

# remove the name of the file from the dataframes
df = df.drop(columns=["filename"])
df_removals = df_removals.drop(columns=["filename"])

# col_names = df.columns[range(21, 57)]
# df = df.drop(columns=col_names) 
# print(df)

# run the random forest model and get outputs on each dataframe
random_forest(df)
random_forest(df_removals)