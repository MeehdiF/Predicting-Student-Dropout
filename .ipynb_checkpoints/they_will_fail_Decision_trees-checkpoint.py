# Step 0: Import the necessary libraries
import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics


# Step 1: Import Data
try:
    file = pd.read_csv("student-por.csv")
except FileNotFoundError:
    print("File Not Found")

print(file.columns)
file.head(5)



"""
### Colmns:
- 1. **school**
- 2. **sex**
- 3. **age**
- 4. **address**
- 5. **famsize**
- 6. **Pstatus**
- 7. **Medu**
- 8. **Fedu**
- 9. **Mjob**
- 10. **Fjob**
- 11. **reason**
- 12. **guardian**
- 13. **traveltime**
- 14. **studytime**
- 15. **failures**
- 16. **schoolsup**
- 17. **famsup**
- 18. **paid**
- 19. **activities**
- 20. **nursery**
- 21. **higher**
- 22. **internet**
- 23. **romantic**
- 24. **famrel**
- 25. **freetime**
- 26. **goout**
- 27. **Dalc**
- 28. **Walc**
- 29. **health**
- 30. **absences**
- 31. **G1**
- 32. **G2**
- 33. **G3**
 
### Important:


- 'sex'
- 'age'
- 'address'
- 'famsize'
- 'Pstatus'
- 'Medu'
- 'Fedu'
- 'Mjob'
- 'Fjob'
- 'guardian'
- 'studytime'
- 'failures'
- 'G1', 'G2'

## Target:
- 'G3'
"""



# Step 2: Clean the Data
n_file = file.drop(columns=['school', 'reason', 'traveltime', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences'] , axis=1)
n_file.head()


succeeded = []
for g3 in n_file['G3'].values:
    if g3 < 10:
        succeeded.append(0)
    else:
        succeeded.append(1)

# Using 'succeeded' as the column name and equating it to the list
n_file = n_file.assign(succeeded=succeeded)
n_file.head()


for col in ['sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'guardian']:
    n_file[col] = pd.factorize(n_file[col])[0]
    
n_file.head()


# Step 3: Split the Data into Training/testing
X = np.asanyarray(n_file[['sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob',
       'Fjob', 'guardian', 'studytime', 'failures', 'G1', 'G2']])
y = np.asanyarray(n_file[["succeeded"]])
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]
X_trainset, X_testset, y_trainset, y_testset = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_trainset.shape,  y_trainset.shape)
print ('Test set:', X_testset.shape,  y_testset.shape)


# Step 4: Create a Model
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree # it shows the default parameters


# Step 5: Train the Model
drugTree.fit(X_trainset,y_trainset)


# Step 6: Make Predictions
predTree = drugTree.predict(X_testset)


# Step 7: Evaluation and Improve
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))