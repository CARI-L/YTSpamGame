from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

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

# KNearestNeighbours
k_range = range(3, 8)
for k in k_range:
    neigh = Pipeline([
        ('prep', preprocess),
        ('clf', KNeighborsClassifier(n_neighbors=k))
    ])
    neigh.fit(X_train, y_train)
    pred[f"neigh{k}"] = neigh.predict(X_test)

# Artificial Neural Network
mlp = Pipeline([
    ('prep', preprocess),
    ('clf', MLPClassifier(max_iter=500, random_state=42, early_stopping=True))
])
mlp.fit(X_train, y_train)
pred["mlp"] = mlp.predict(X_test)

# Random Forest (Test N)
test_params = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 20, 40]
}
for n in test_params["n_estimators"]:
    for d in test_params["max_depth"]:  
        forest = Pipeline([
            ('prep', preprocess),
            ('clf', RandomForestClassifier())
        ])
        forest.fit(X_train, y_train)
        pred[f"forest{n}-{d}"] = forest.predict(X_test)
