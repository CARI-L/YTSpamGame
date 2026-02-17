from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd


# fetch dataset 
youtube_spam_collection = fetch_ucirepo(id=380) 
  
# data (as pandas dataframes) 
X = youtube_spam_collection.data.features 
y = youtube_spam_collection.data.targets.values.ravel()

# dropping date. Lots of preprocessing for a likely less important feature
X = X.drop(columns="DATE")

# split data (80/20), stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# our training models will not be able to handle
# raw text input in the data.
# vectorization is done to convert it into number
# based data that the model can learn off of.
preprocess = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(stop_words='english'), 'CONTENT'),
        ('author', OneHotEncoder(handle_unknown='ignore'), ['AUTHOR'])
    ])
pred = {}

# Extensive Method Testing
# 5 different classification methods are tested
# Some methods have differentiations that are tested
# The best version is selected for final comparisons

# Naive Bayes
mNB = Pipeline([
    ('prep', preprocess),
    ('clf', MultinomialNB())
])
mNB.fit(X_train, y_train)
pred["mNB"] = mNB.predict(X_test)

cNB = Pipeline([
    ('prep', preprocess),
    ('clf', ComplementNB())
])
cNB.fit(X_train, y_train)
pred["cNB"] = cNB.predict(X_test)
print("===========\nNaive Bayes\n===========")
print("Type:      Multinomial  Complement\n----------------------------------")
print("Accuracy:  %.3f        %.3f" %(accuracy_score(y_test, pred["mNB"]), accuracy_score(y_test, pred["cNB"])))
print("Precision: %.3f        %.3f" %(precision_score(y_test, pred["mNB"]), precision_score(y_test, pred["cNB"])))
print("Recall:    %.3f        %.3f" %(recall_score(y_test, pred["mNB"]), recall_score(y_test, pred["cNB"])))
print("F1:        %.3f        %.3f\n\n" %(f1_score(y_test, pred["mNB"]), f1_score(y_test, pred["cNB"])))

# SVM
dSVM = Pipeline([
    ('prep', preprocess),
    ('clf', SVC(random_state=42))
])
dSVM.fit(X_train, y_train)
pred["dSVM"] = dSVM.predict(X_test)

lSVM = Pipeline([
    ('prep', preprocess),
    ('clf', SVC(kernel='linear', random_state=42,))
])
lSVM.fit(X_train, y_train)
pred["lSVM"] = lSVM.predict(X_test)

print("=======================\nSupport Vector Machines\n=======================")
print("Kernel:    Linear  RBF\n------------------------")
print("Accuracy:  %.3f   %.3f" %(accuracy_score(y_test, pred["lSVM"]), accuracy_score(y_test, pred["dSVM"])))
print("Precision: %.3f   %.3f" %(precision_score(y_test, pred["lSVM"]), precision_score(y_test, pred["dSVM"])))
print("Recall:    %.3f   %.3f" %(recall_score(y_test, pred["lSVM"]), recall_score(y_test, pred["dSVM"])))
print("F1:        %.3f   %.3f\n\n" %(f1_score(y_test, pred["lSVM"]), f1_score(y_test, pred["dSVM"])))


# KNearestNeighbours
print("====================\nK-Nearest Neighbours\n====================")
print("K:         ", end="")

k_range = range(3, 8)
for k in k_range:
    print(k, "     ", end="")
    neigh = Pipeline([
        ('prep', preprocess),
        ('clf', KNeighborsClassifier(n_neighbors=k))
    ])
    neigh.fit(X_train, y_train)
    pred[f"neigh{k}"] = neigh.predict(X_test)
print("\n--------------------------------------------")
print("Accuracy:  %.3f  %.3f  %.3f  %.3f  %.3f" %(accuracy_score(y_test, pred["neigh3"]), accuracy_score(y_test, pred["neigh4"]), accuracy_score(y_test, pred["neigh5"]), accuracy_score(y_test, pred["neigh6"]), accuracy_score(y_test, pred["neigh7"])))
print("Precision: %.3f  %.3f  %.3f  %.3f  %.3f" %(precision_score(y_test, pred["neigh3"]), precision_score(y_test, pred["neigh4"]), precision_score(y_test, pred["neigh5"]), precision_score(y_test, pred["neigh6"]), precision_score(y_test, pred["neigh7"])))
print("Recall:    %.3f  %.3f  %.3f  %.3f  %.3f" %(recall_score(y_test, pred["neigh3"]), recall_score(y_test, pred["neigh4"]), recall_score(y_test, pred["neigh5"]), recall_score(y_test, pred["neigh6"]), recall_score(y_test, pred["neigh7"])))
print("F1:        %.3f  %.3f  %.3f  %.3f  %.3f\n\n" %(f1_score(y_test, pred["neigh3"]), f1_score(y_test, pred["neigh4"]), f1_score(y_test, pred["neigh5"]), f1_score(y_test, pred["neigh6"]), f1_score(y_test, pred["neigh7"])))


# Artificial Neural Network
mlp1 = Pipeline([
    ('prep', preprocess),
    ('clf', MLPClassifier(max_iter=500, random_state=42, early_stopping=True))
])
mlp1.fit(X_train, y_train)
pred["mlp1"] = mlp1.predict(X_test)

mlp2 = Pipeline([
    ('prep', preprocess),
    ('clf', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, early_stopping=True))
])
mlp2.fit(X_train, y_train)
pred["mlp2"] = mlp2.predict(X_test)

print("==========================\nArtificial Neural Network\n==========================")
print("Layers:    (100,)  (100, 50)\n----------------------------")
print("Accuracy:  %.3f   %.3f" %(accuracy_score(y_test, pred["mlp1"]), accuracy_score(y_test, pred["mlp2"])))
print("Precision: %.3f   %.3f" %(precision_score(y_test, pred["mlp1"]), precision_score(y_test, pred["mlp2"])))
print("Recall:    %.3f   %.3f" %(recall_score(y_test, pred["mlp1"]), recall_score(y_test, pred["mlp2"])))
print("F1:        %.3f   %.3f\n\n" %(f1_score(y_test, pred["mlp1"]), f1_score(y_test, pred["mlp2"])))

# Random Forest (Test N)
print("==============\nRandom Forests\n==============")
print("Estimators: 100                       200                       500")
print("Max Depth:  ", end="")
test_params = {
    "n_estimators": [100, 200, 500],
    "max_depth": [20, 40, None]
}
for n in test_params["n_estimators"]:
    for d in test_params["max_depth"]:  
        forest = Pipeline([
            ('prep', preprocess),
            ('clf', RandomForestClassifier(max_depth=d, n_estimators=n, random_state=42))
        ])
        forest.fit(X_train, y_train)
        pred[f"forest{n}-{d}"] = forest.predict(X_test)
        print(d,"     ", end="")
print("\n-------------------------------------------------------------------------------------")
print("Accuracy:   %.3f   %.3f   %.3f     %.3f   %.3f   %.3f     %.3f   %.3f   %.3f" %(accuracy_score(y_test, pred["forest100-20"]), accuracy_score(y_test, pred["forest100-40"]), accuracy_score(y_test, pred["forest100-None"]), accuracy_score(y_test, pred["forest200-20"]), accuracy_score(y_test, pred["forest200-40"]), accuracy_score(y_test, pred["forest200-None"]), accuracy_score(y_test, pred["forest500-20"]), accuracy_score(y_test, pred["forest500-40"]), accuracy_score(y_test, pred["forest500-None"])))
print("Precision:  %.3f   %.3f   %.3f     %.3f   %.3f   %.3f     %.3f   %.3f   %.3f" %(precision_score(y_test, pred["forest100-20"]), precision_score(y_test, pred["forest100-40"]), precision_score(y_test, pred["forest100-None"]), precision_score(y_test, pred["forest200-20"]), precision_score(y_test, pred["forest200-40"]), precision_score(y_test, pred["forest200-None"]), precision_score(y_test, pred["forest500-20"]), precision_score(y_test, pred["forest500-40"]), precision_score(y_test, pred["forest500-None"])))
print("Recall:     %.3f   %.3f   %.3f     %.3f   %.3f   %.3f     %.3f   %.3f   %.3f" %(recall_score(y_test, pred["forest100-20"]), recall_score(y_test, pred["forest100-40"]), recall_score(y_test, pred["forest100-None"]), recall_score(y_test, pred["forest200-20"]), recall_score(y_test, pred["forest200-40"]), recall_score(y_test, pred["forest200-None"]), recall_score(y_test, pred["forest500-20"]), recall_score(y_test, pred["forest500-40"]), recall_score(y_test, pred["forest500-None"])))
print("F1:         %.3f   %.3f   %.3f     %.3f   %.3f   %.3f     %.3f   %.3f   %.3f\n\n" %(f1_score(y_test, pred["forest100-20"]), f1_score(y_test, pred["forest100-40"]), f1_score(y_test, pred["forest100-None"]), f1_score(y_test, pred["forest200-20"]), f1_score(y_test, pred["forest200-40"]), f1_score(y_test, pred["forest200-None"]), f1_score(y_test, pred["forest500-20"]), f1_score(y_test, pred["forest500-40"]), f1_score(y_test, pred["forest500-None"])))

print("===============\nThe Best Models\n===============")
print("Model Type: Naive Bayes  SVM    Nearest Neighbours  Neural Network  Random Forest\n---------------------------------------------------------------------------------")
print("Accuracy:   %.3f        %.3f  %.3f               %.3f           %.3f" %(accuracy_score(y_test, pred["cNB"]), accuracy_score(y_test, pred["lSVM"]), accuracy_score(y_test, pred["neigh3"]), accuracy_score(y_test, pred["mlp1"]), accuracy_score(y_test, pred["forest100-20"])))
print("Precision:  %.3f        %.3f  %.3f               %.3f           %.3f" %(precision_score(y_test, pred["cNB"]), precision_score(y_test, pred["lSVM"]), precision_score(y_test, pred["neigh3"]), precision_score(y_test, pred["mlp1"]), precision_score(y_test, pred["forest100-20"])))
print("Recall:     %.3f        %.3f  %.3f               %.3f           %.3f" %(recall_score(y_test, pred["cNB"]), recall_score(y_test, pred["lSVM"]), recall_score(y_test, pred["neigh3"]), recall_score(y_test, pred["mlp1"]), recall_score(y_test, pred["forest100-20"])))
print("F1:         %.3f        %.3f  %.3f               %.3f           %.3f\n\n" %(f1_score(y_test, pred["cNB"]), f1_score(y_test, pred["lSVM"]), f1_score(y_test, pred["neigh3"]), f1_score(y_test, pred["mlp1"]), f1_score(y_test, pred["forest100-20"])))

# print confusion matrices for best models

# test tool
neigh = Pipeline([
    ('prep', preprocess),
    ('clf', KNeighborsClassifier(n_neighbors=3))
])
neigh.fit(X_train, y_train)

forest = Pipeline([
    ('prep', preprocess),
    ('clf', RandomForestClassifier(n_estimators=200, random_state=42))
])
forest.fit(X_train, y_train)

username = ""
while True:
    name = input("User Name (leave blank to reuse): ")
    if username == "": username = name
    while username == "":
        print("User Name not set, please set username")
        username = input("User Name: ")
    content = ""
    content = input("Write your comment:\n")
    while content == "":
        print("Content of comment cannot be empty")
        content = input("Write your comment:\n")
    print()
    
    # create dataframe
    data = {"AUTHOR": [username], "CONTENT": [content]}
    df = pd.DataFrame.from_dict(data)
    print("Naive Bayes says: THIS IS %sSPAM" %("NOT " if cNB.predict(df)[0] == 0 else ""))
    print("SVM says: THIS IS %sSPAM" %("NOT " if lSVM.predict(df)[0] == 0 else ""))
    print("Nearest Neighbours says: THIS IS %sSPAM" %("NOT " if neigh.predict(df)[0] == 0 else ""))
    print("Artificial Neural Network says: THIS IS %sSPAM" %("NOT " if mlp1.predict(df)[0] == 0 else ""))
    print("Random Forest says: THIS IS %sSPAM" %("NOT " if forest.predict(df)[0] == 0 else ""))
    q = input("\nQuit? (y/n): ").lower()
    if q == "y":
        break
    print()
