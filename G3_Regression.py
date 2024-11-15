# Step 0: Import the necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn import linear_model
import numpy as np


# Step 1: Import Data
try:
    file = pd.read_csv("student-por.csv")
except FileNotFoundError:
    print("File Not Found")

print(file.columns)
file.head(5)


# Step 2: Clean the Data
n_file = file.drop(columns=['school', 'reason', 'traveltime', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences'] , axis=1)
n_file.head()

for col in ['sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'guardian']:
    n_file[col] = pd.factorize(n_file[col])[0]
    
n_file.head()


# Step 3: Split the Data into Training/testing
X = np.asanyarray(n_file[['sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob',
       'Fjob', 'guardian', 'studytime', 'failures', 'G1', 'G2']])
y = np.asanyarray(n_file[["G3"]])
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=38)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)



# Step 4: Create a Model and Step 5: Train the Model
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
# The coefficients
print ('Coefficients: ', regr.coef_)




# Step 6: Make Predictions
y_hat= regr.predict(X_test)



# Step 7: Evaluation and Improve
print("Residual sum of squares: %.2f"
      % np.mean((y_hat - y_test) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_test, y_test))