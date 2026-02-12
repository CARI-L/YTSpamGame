from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# fetch dataset 
youtube_spam_collection = fetch_ucirepo(id=380) 
  
# data (as pandas dataframes) 
X = youtube_spam_collection.data.features 
y = youtube_spam_collection.data.targets 

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
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

pred = {}

# Extensive Method Testing
# 5 different classification methods are tested
# Some methods have differentiations that are tested
# The best version is selected for final comparisons

# Naive Bayes (Multinomial vs Complement)
# SVM (linear vs rbf?)
# KNearestNeighbours (test k)
# Artificial Neural Network
# Random Forest (Test N)