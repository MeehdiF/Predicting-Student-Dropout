# Step 0: Import the necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt



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

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# Step 4: Create a Model
LR = LogisticRegression(C=0.01, solver='liblinear')
LR



# Step 5: Train the Model
LR.fit(X_train,y_train)
LR



# Step 6: Make Predictions
yhat = LR.predict(X_test)
yhat
yhat_prob = LR.predict_proba(X_test)
yhat_prob



# Step 7: Evaluation and Improve
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)

print (classification_report(y_test, yhat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['succeeded','faill'],normalize= False,  title='Confusion matrix')

print(f"f1_score: {f1_score(y_test, yhat, average='weighted')}")
print(f"jaccard_score: {jaccard_score(y_test, yhat, pos_label=1)}")