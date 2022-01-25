import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import tree
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pickle
from pymongo import MongoClient
from sklearn import preprocessing
client = MongoClient("mongodb://localhost:27017/")
db = client["ppDB"]
df = db["predict"].find()
df = pd.DataFrame.from_dict(df)
df = df.drop(columns="_id")
prepare_df = df.apply(preprocessing.LabelEncoder().fit_transform)

target = prepare_df["price"]
prepare_df = df.drop(columns="price")
X_train, X_test, y_train, y_test = train_test_split(prepare_df, target, test_size=0.75, random_state=42)
#Training Data and Predictor Variable
# Use all data for training (tarin-test-split not used)
clf = tree.DecisionTreeClassifier()
model = clf.fit(X_train.values, y_train.values)
pred = clf.predict(X_test)
#Fitting model with trainig data
print(pred)
# Saving model to current directory
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
pickle.dump(model, open('model.pkl','wb'))
model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[100,10,100,0,1]]))
