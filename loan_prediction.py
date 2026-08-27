import pandas as pd

# Load the dataset
df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")  # use your exact filename

# First look
print(df.head())        # first 5 rows
print(df.shape)          # (rows, columns)
print(df.info())         # column names, data types, missing values
print(df.describe())     # stats for numeric columns
#Categorical columns - filled with most common value
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])

# Numeric columns - fill with median/mode
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

# Check - ab koi missing value nahi honi chahiye
print(df.isnull().sum())
# Convert Yes/No, Male/Female type columns into numbers
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
df['Property_Area'] = df['Property_Area'].map({'Urban': 2, 'Semiurban': 1, 'Rural': 0})
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
df['Dependents'] = df['Dependents'].replace('3+', 3).astype(float)
df['Dependents'] = df['Dependents'].astype(int)

#property area has 3 categories (Urban, Rural, Semiurban)
df = pd.get_dummies(df, columns=['Property_Area'], drop_first=True)
#check the result
print(df.head())
print(df.info())

#Features (Input) - sab columns excpet Loan_ID aur Loan_Status
X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)

#Target(output) -  jo humen predict krna hey
y = df['Loan_Status']

#Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training data size:", X_train.shape)
print("Testing data size:", X_test.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#training model
from sklearn.linear_model import LogisticRegression

#model banao
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Model trained ")

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#predict on test data
y_pred = model.predict(X_test)

#evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#detailed report
print(classification_report(y_test, y_pred))

#confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier
#naya model bnao
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
#predictions lo 
y_pred_rf = rf_model.predict(X_test)
 #Accuracy or report
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf)) 