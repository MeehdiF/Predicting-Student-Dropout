# Step 0: Import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
import itertools
from sklearn.svm import SVC
from sklearn.metrics import f1_score, jaccard_score
from sklearn.preprocessing import StandardScaler


## Function for Find the best kernel, C and gamma
def finder_for_svm(train_x, train_y):
    param_grid = {
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "C": [0.1, 1, 10],
        "gamma": ["scale", 1, 0.01, 0.1]
    }
    grid_search = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=1)
    try:
        grid_search.fit(train_x, train_y)
    except Exception as e:
        print(f"Error during model fitting: {e}")
        return None
    ##Showing the result
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best kernel found: {grid_search.best_params_['kernel']}")

    print("And...")

    best_model = grid_search.best_estimator_
    accuracy = best_model.score(train_x, train_y)
    print(f"Test set accuracy: {accuracy:2f}")
    return best_model


##Function for plotting the Data
def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1) [:, np.newaxis]
        print("Normalize Confusion matrix")
    else:
        print("Confusion matrix, without normalization")
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)


    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center", color="white" if cm[i, j]> thresh else "black")
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Prediction label")
    plt.show()



# Step 1: Import Data
try:
    file = pd.read_csv("student-por.csv")
except FileNotFoundError:
    print("File Not Found")

print(file.columns)
file.head(5)


"""
## Colmns:
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
n_file['G3'].value_counts()
succeeded = []
for g3 in n_file['G3'].values:
    if g3 < 10:
        succeeded.append(0)
    else:
        succeeded.append(1)

# Using 'succeeded' as the column name and equating it to the list
n_file = n_file.assign(succeeded=succeeded)
n_file.head()
n_file['succeeded'].value_counts()


for col in ['sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'guardian']:
    n_file[col] = pd.factorize(n_file[col])[0]
    
n_file.head()
n_file.dtypes




# Step 3: Split the Data into Training/testing
n_file.columns
X = np.asanyarray(n_file[['sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob',
       'Fjob', 'guardian', 'studytime', 'failures', 'G1', 'G2']])
y = np.asanyarray(n_file[["succeeded"]])
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)



# Step 4: Create a Model and Step 5: Train the Model
model = finder_for_svm(X_train, y_train)
print(model)


## Check if model is successfully created
if model is None:
    print("Model creation failed. Exiting.")
    exit()


# Step 6: Make Predictions
predicted_y = model.predict(X_test)

## Check if predictions are made correctly
if len(predicted_y) != len(y_test):
    print("Mismatch in length between predictions and actual labels.")
    exit()


##Compute Confusion matrix
cnf_matrix = confusion_matrix(y_test, predicted_y, labels=[0, 1])
np.set_printoptions(precision=2)

print(classification_report(y_test, predicted_y))

##plot Confusion matrix(non-normalized)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["fail", "succeeded"], normalize=False, title="Confusion matrix")


print(f"f1_score: {f1_score(y_test, predicted_y, average='weighted')}")
print(f"jaccard_score: {jaccard_score(y_test, predicted_y, pos_label=1)}")




# Step 7: Evaluation and Improve
print(f"f1_score: {f1_score(y_test, predicted_y, average='weighted')}")
print(f"jaccard_score: {jaccard_score(y_test, predicted_y, pos_label=1)}")