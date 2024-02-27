'''

                          NAIVE BAYES
                          ===========
'''

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the training data
SalaryData_Train = pd.read_csv("C:/DS2/1.1_Naive_Bayes/SalaryData_Train.csv")

# Identify features (X) and target variable (y) in the training data
X_train = SalaryData_Train.drop('Salary', axis=1)
y_train = SalaryData_Train['Salary']

# Convert categorical variables to numerical using one-hot encoding
X_train = pd.get_dummies(X_train, drop_first=True)

# Train Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Load the test data
SalaryData_Test = pd.read_csv("C:/DS2/1.1_Naive_Bayes/SalaryData_Test.csv")

# Identify features (X) for the test data
X_test = SalaryData_Test.drop('Salary', axis=1)

# Convert categorical variables to numerical using one-hot encoding
X_test = pd.get_dummies(X_test, drop_first=True)

# Make predictions using the trained model
y_pred_test = model.predict(X_test)

# Assuming 'Salary' is the actual target column in the test data
y_true_test = SalaryData_Test['Salary']

# Evaluate the model on the test data
accuracy_test = accuracy_score(y_true_test, y_pred_test)
print(f'Accuracy on test data: {accuracy_test * 100:.2f}%')