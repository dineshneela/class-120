import pandas as pd 
import plotly.express as px 
import numpy
import statistics
import random 
import csv
df = pd.read_csv("income.csv")
print(df.head())
print(df.describe)
from sklearn.model_selection import train_test_split
X = df[["age", "hours-per-week", "education-num", "capital-gain", "capital-loss"]]
y = df["income"]
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X, y, test_size = 0.25, random_state = 42)



from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler 

sc = StandardScaler()

X_train_1 = sc.fit_transform(X_train_1) 
X_test_1 = sc.fit_transform(X_test_1) 

model_1 = GaussianNB()
model_1.fit(X_train_1, y_train_1)

y_pred_1 = model_1.predict(X_test_1)

accuracy = accuracy_score(y_test_1, y_pred_1)
print(accuracy)
