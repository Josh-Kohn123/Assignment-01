print("#####Data Exploration#######")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport

## Open file
df = pd.read_csv("customers_annual_spending_dataset.csv", header=0)
#print(df.info())
#print(df.columns)
#use ydata_profiling to in depth analysis of report before cleaning the data
profile = ProfileReport(df, title="Profiling Report")
#profile.to_file("your_report.html")

#notice missing data points in many columns. Apply appropriate preprocessing.

#LastName: fillna with random string. Each cell should get a inque fillna
df["LastName"] = df['LastName'].fillna("missing")
#print(df["LastName"].values)

# Set HasComplaint to 1 where ComplaintSatisfaction is greater than 0
df.loc[df["ComplaintSatisfaction"] > 0, "HasComplaint"] = 1
#print(df["HasComplaint"].unique)

#Set empty ComplaintSatisfaction to be the average ComplaintSatisfaction
mean_satisfaction = df["ComplaintSatisfaction"].mean()
df.loc[df["ComplaintSatisfaction"].isna(), "ComplaintSatisfaction"] = mean_satisfaction

#Cardtype
#print(df["CardType"].unique())
#print(df["CardType"].value_counts())
#For the purpose of this assignment, the type of card is irrelevant. We will split up the empty cells equally.
cardtypes = ["American Express","Visa","MasterCard","Discover"]
df["CardType"]= df["CardType"].apply(lambda x: np.random.choice(cardtypes) if pd.isna(x) else x)
#print(df.info())

#The recordnumber,customerID and Lastname are not variables tat effect outcomes. They are nominal figures so we can remove them
df=df.drop(['RecordNumber','CustomerId', 'LastName'], axis=1)

#The rest of the info is crucial, so any column with missing cells should be removed. 
df_cleaned = df.dropna()
#print(df_cleaned.info())

#We are left with 818 complete datapoints.
df_cleaned.info()
#print(df_cleaned)

print("#####Data Preprocessing#######")
#Before running the linear regression, we need to change the columns to intergers/float.
#print(df_cleaned.info()) 
#Let's deal with Location first. How many unique locations are there?
num_locations = df_cleaned['Location'].groupby(df_cleaned['Location']).count()
unique_locations = df_cleaned["Location"].unique()
#print(unique_locations)
#print(num_locations)
#since these values are nominal, we should create a different row with dummy variables (1=here, 0= not here)
for uni in unique_locations:
    df_cleaned.insert(len(df_cleaned.columns),uni,0)
    df_cleaned.loc[df_cleaned['Location']==uni,uni]=1
#     print(f"unique values for {uni} column: {df_cleaned[uni].unique()}")
# print(df_cleaned.columns)

#Let's now work on Gender:
unique_gender = df_cleaned['Gender'].unique()
for uni in unique_gender:
    df_cleaned.insert(len(df_cleaned.columns),uni,0)
    df_cleaned.loc[df_cleaned["Gender"]==uni,uni]=1

#And now CardType:
unique_cards = df_cleaned['CardType'].unique()
for uni in unique_cards:
    df_cleaned.insert(len(df_cleaned.columns),uni,0)
    df_cleaned.loc[df_cleaned["CardType"]==uni,uni]=1
    #print(f"unique values for {uni} column: {df_cleaned[uni].unique()}")
    #print(f"count of {uni}:{df_cleaned[uni].sum()}")

#Let's make sure all of our columns are int/float:
df_cleaned=df_cleaned.drop(columns=["Location","Gender","CardType"])
#print(df_cleaned.info())

#Now let's print out the updated report:
cleaned_profile = ProfileReport(df_cleaned, title="Profiling Report2")
#cleaned_profile.to_file("your_report2.html")

# I get an error that some of my column names are dtype 'str_' which is creating problems for me. Let's look to see which columns are problematic:
#print(df_cleaned.dtypes)

print("######BUILDING REGRESSION MODEL######")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# #our dependant variable (y) for the linear regression is 'AnnualSpending'.
#print(df_cleaned['AnnualSpending'].mean())
y = df_cleaned['AnnualSpending']
X= df_cleaned.drop(columns=['AnnualSpending'])
X.columns = X.columns.astype(str)

# # Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Print the coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

compare_table = {}
compare_table["Y_Test"] = y_test
compare_table["Y_Predict"]= y_pred
compare_table = pd.DataFrame(compare_table)
compare_table["Difference"]=compare_table["Y_Test"]-compare_table["Y_Predict"]
plt.hist(compare_table["Difference"])
plt.show()

print("#######Convert the model to binary classification problem:######")

print(y.mean())
df_cleaned.insert(len(df_cleaned.columns),"BelowAverage",0)
df_cleaned.loc[df_cleaned["AnnualSpending"]<y.mean(),"BelowAverage"]=1
df_cleaned.drop(columns="AnnualSpending")
#print(df_cleaned.columns)
#print(df_cleaned.BelowAverage)
#print(df_cleaned.AnnualSpending.values)
#print(len(df_cleaned.AnnualSpending.values))

print("#######Classification Model Training and Evaluation#######")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Expand X so also not include 'BelowAverage' column 
df_cleaned['BelowAverage'] = df_cleaned['BelowAverage']
X = df_cleaned.drop(columns=['BelowAverage'])
X.columns = X.columns.astype(str)
y = df_cleaned["BelowAverage"]
#X = X.drop(columns=['BelowAverage'])

# Split the data into training and testing sets (70% train, 30% test)
#X.columns = X.columns.astype(str)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# An appropriate evaluation metric is a confusion table and subsequently an F1 score. This is because 
# we want to see how many correct vs incorrect predictions the model made, and likewise the reliability (based on קליטה ודיוק) of the model

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

