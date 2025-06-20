import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

#####################################################################
#####################################################################
#  1 Data Exploration and Preprocessing

# 1.a  Load the ”ds salaries” dataset
df = pd.read_csv('ds_salaries.csv')

# Forcing float64 data type to maximize accuracy
for col in df.select_dtypes(include=['number']).columns:
    df[col] = df[col].astype('float64')

# 1.b Handle any missing values or outliers, if present
# Check for duplicate rows
duplicate_rows = df[df.duplicated()]
print("There a are " + str(len(duplicate_rows)) + " duplicate rows")
# Duplicated rows are nearly one third of the dataset, we will drop them since they 
# could skew the analysis.
df = df.drop_duplicates()

# Check for missing values and remove rows with any NaN entries
nan_entries = df[df.isna().any(axis=1)]
print("There a are " + str(len(nan_entries)) + " NaN entries")
if not nan_entries.empty:
    df = df.drop(nan_entries.index)
# Actually there are no NaN entries in the dataset, so we can proceed with the analysis.

# Visualize the distribution of salary in USD to check for outliers
plt.hist(df['salary_in_usd'], bins=30, edgecolor='black')
plt.xlabel('Salary in USD')
plt.ylabel('Frequency')
plt.title('Distribution of Salary in USD')
plt.show()

# The distribution appears normal, 
# but there are some outliers on the right side of the distribution.
print("DataFrame shape:", df.shape)
# Remove the outliers by setting a threshold for salary in USD
# Remove the outliers
df = df[df['salary_in_usd'] <= 400000]
print("DataFrame shape after removing USD outliers:", df.shape)

# 1.b Visualize the distribution of the ”job title” and ”employee residence” variables 
fig, axs = plt.subplots(2, 1, figsize=(16, 6))

df['job_title'].value_counts().head(10).plot(
    kind='bar', ax=axs[0], color='skyblue', edgecolor='black'
)
axs[0].set_xlabel('Job Title')
axs[0].set_ylabel('Count')
axs[0].set_title('Top 10 Job Titles')

df['employee_residence'].value_counts().head(10).plot(
    kind='bar', ax=axs[1], color='salmon', edgecolor='black'
)
axs[1].set_xlabel('Employee Residence')
axs[1].set_ylabel('Count')
axs[1].set_title('Top 10 Employee Residences')

plt.tight_layout()
plt.show()

# There are a lot of different job titles, so we keep only the most frequent ones.
# Keep only entries whose job title's last word is among the 10 most frequent
top10_last_words = df['job_title'].apply(lambda x: x.split()[-1]).value_counts().nlargest(10).index
df = df[df['job_title'].apply(lambda x: x.split()[-1] in top10_last_words)]
print("DataFrame shape after removing job title outliers:", df.shape)

# US residents are overrepresented in the dataset, so we will keep only one every nine US residents.
# Keep only one every nine US residents in the dataset
us_residents = df[df['employee_residence'] == 'US']
keep_indices = us_residents.sample(frac=1/9, random_state=42).index
non_us_residents = df[df['employee_residence'] != 'US']
df = pd.concat([non_us_residents, df.loc[keep_indices]], ignore_index=True)
print("DataFrame shape after keeping one every nine US residents:", df.shape)

# Plot 10 most frequent job title last words
fig, axs = plt.subplots(2, 1, figsize=(16, 6))

last_words = df['job_title'].apply(lambda x: x.split()[-1])
last_words.value_counts().head(10).plot(
    kind='bar', ax=axs[0], color='lightgreen', edgecolor='black'
)
axs[0].set_xlabel('Job Title Last Word')
axs[0].set_ylabel('Count')
axs[0].set_title('Top 10 Job Title Last Words')

# Plot 10 most frequent employee residences
df['employee_residence'].value_counts().head(10).plot(
    kind='bar', ax=axs[1], color='orange', edgecolor='black'
)
axs[1].set_xlabel('Employee Residence')
axs[1].set_ylabel('Count')
axs[1].set_title('Top 10 Employee Residences')

plt.tight_layout()
plt.show()

#####################################################################
#####################################################################
### 2 Feature Selection and Engineering

# 2.a Identify the relevant features from the dataset that can potentially 
# influence salary prediction.
# The work year would be relevant, but entries are from 2020-2023, 
# so it may not add much value.
# The experience level is a crucial factor as it typically correlates with salary.
# The employment type would be relevant but most of the entries are 'full-time', 
# so it may not add much value.
# The job title is also important as it directly relates to the role and responsibilities,
# which influence salary.
# The employee residence and company location can affect salary due to cost of living 
# differences.
# The company size can also influence salary, as larger companies may have more 
# resources to pay higher salaries.
# The remote ratio does not seem to have an impact on salary based on the dataset.
# In fact considering the average salary of different remote ratios, it appears that 
# remote work does not significantly affect salary.

# plot average salary by remote ratio
avg_salary_by_remote = df.groupby('remote_ratio')['salary_in_usd'].mean()
plt.figure(figsize=(8,5))
avg_salary_by_remote.plot(kind='bar', color='skyblue', edgecolor='black')
plt.xlabel('Remote Ratio')
plt.ylabel('Average Salary in USD')
plt.title('Average Salary in USD by Remote Ratio')
plt.tight_layout()
plt.show()

# The relevant features that can potentially influence salary prediction are:
relevant_features = [
    'experience_level', 'job_title', 'employee_residence',
    'company_location', 'company_size'
]

# 2.b Perform feature engineering, scaling features.
# Normalize the salary_in_usd column to a scale between 0 and 1
df['salary_in_usd_normalized'] = df['salary_in_usd'] / df['salary_in_usd'].max()

# Encode categorical features to numerical values for model training
# Encode experience level and company size to numerical values
experience_map = { 'EX':4,'SE':3, 'MI':2, 'EN':1}
size_map = {'S': 1, 'M': 2, 'L': 3}
df['experience_level'] = df['experience_level'].map(experience_map)
df['company_size'] = df['company_size'].map(size_map)

# Encode job_title, employee_residence, and company_location as category codes
for col in ['job_title', 'employee_residence', 'company_location']:
    df[col] = pd.Categorical(df[col]).codes

#####################################################################
#####################################################################
# 3 Model Development and Evaluation
# 3.a Split the dataset into training and testing subsets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

X_train = train_df[relevant_features].copy()
X_test = test_df[relevant_features].copy()

# Align columns of train and test sets
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

y_train = train_df['salary_in_usd_normalized']
y_test = test_df['salary_in_usd_normalized']

# 3.b-c Choose and train a regression model, evaluate its performance 
#@# Test different hyerparameters for the Decision Tree Regressor
param_grid = {
    'criterion': ['squared_error', 'absolute_error'],
    'n_estimators': [300, 400, 500, 600],
    'max_depth': [20, 30, 40, 50, None],
    'min_samples_split': [2, 5, 8, 12],
    'min_samples_leaf': [1, 2, 3, 5],
    'max_features': ['sqrt', 'log2', 0.7, 0.8]
}
# Search the best Random Forest Regressor
search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions=param_grid,
    n_iter=40,
    scoring='r2',
    cv=3,
    n_jobs=-1,
    verbose=1,
    random_state=42
)
# Train different hyperparameters for the Decision Tree Regressor
search.fit(X_train, y_train)
best_model = search.best_estimator_
print("Best hyperparameters found:", search.best_params_)

# Test the best model on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Decision Tree Regressor MSE: {mse:.2f}")
print(f"Decision Tree Regressor R^2: {r2:.2f}")

# 3.d Fine-tune the model, by adjusting hyperparameters to improve performance
### Best Random Forest Regressor hyperparameters found:
best_model = RandomForestRegressor(
    n_estimators= 200,
    min_samples_split= 5,
    min_samples_leaf= 1,
    max_features= 'sqrt',
    max_depth= None,
    criterion= 'squared_error',
    random_state=42)
best_model.fit(X_train, y_train)

# Visualize feature importance
importances = best_model.feature_importances_
features = X_train.columns
feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
feat_imp.plot(kind='bar', figsize=(12,6), title="Feature Importance")
plt.tight_layout()
plt.show()

#####################################################################
#####################################################################
# 4 Prediction and Interpretation
# 4.a Use the trained model to make predictions on the testing data
y_pred = best_model.predict(X_test)

# 4.b Assess the model’s performance by comparing the predicted salaries 
# with the actual salaries
# Visualize the predicted vs actual salary distribution
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].hist(y_pred, bins=30, edgecolor='black')
axs[0].set_xlabel('Salary in USD (Normalized)')
axs[0].set_ylabel('Frequency')
axs[0].set_title('Predicted Salary Distribution of test set')

axs[1].hist(y_test, bins=30, edgecolor='black')
axs[1].set_xlabel('Salary in USD (Normalized)')
axs[1].set_ylabel('Frequency')
axs[1].set_title('Actual Salary Distribution of test set')

plt.tight_layout()
plt.show()

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Decision Tree Regressor MSE: {mse:.2f}")
print(f"Decision Tree Regressor R^2: {r2:.2f}")



y_pred_train = best_model.predict(X_train)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].hist(y_pred_train, bins=30, edgecolor='black')
axs[0].set_xlabel('Salary in USD (Normalized)')
axs[0].set_ylabel('Frequency')
axs[0].set_title('Predicted Salary Distribution over training set')

axs[1].hist(y_train, bins=30, edgecolor='black')
axs[1].set_xlabel('Salary in USD (Normalized)')
axs[1].set_ylabel('Frequency')
axs[1].set_title('Actual Salary Distribution of training set')

plt.tight_layout()
plt.show()

# Visualize the residuals to check for patterns
err = y_train - y_pred_train
plt.figure(figsize=(10, 5))
plt.hist(err, bins=100, edgecolor='black', color='lightcoral')
plt.title('Residuals Distribution of training set')
plt.show()
#############################################
#######################################################
#############################################
# # Other models to try

# # # Train a linear regressor
# best_model = LinearRegression()
# best_model.fit(X_train, y_train)

# # Test the best model on the test set
# y_pred = best_model.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print(f"Linear Regressor MSE: {mse:.2f}")
# print(f"Linear Regressor R^2: {r2:.2f}")

# #@# Test different hyerparameters for the HistGradientBoostingRegressor
# # Define hyperparameter grid for HistGradientBoostingRegressor
# param_grid = {
#     'learning_rate': [0.01, 0.05, 0.1, 0.2],
#     'max_iter': [100, 200, 300],
#     'max_depth': [None, 10, 20, 40],
#     'min_samples_leaf': [20, 30, 50],
#     'l2_regularization': [0, 0.1, 1.0],
#     'max_bins': [150, 250]
# }
# # Perform randomized search CV for HistGradientBoostingRegressor
# search = RandomizedSearchCV(
#     HistGradientBoostingRegressor(random_state=42),
#     param_distributions=param_grid,
#     n_iter=20,
#     scoring='r2',
#     cv=3,
#     n_jobs=-1,
#     verbose=1,
#     random_state=42
# )
# search.fit(X_train, y_train)
# best_model = search.best_estimator_
# # print("Best hyperparameters found (HistGradientBoostingRegressor):", search.best_params_)

# # Test the best model on the test set
# y_pred = best_model.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print(f"HistGradientBoosting Regressor MSE: {mse:.2f}")
# print(f"HistGradientBoosting Regressor R^2: {r2:.2f}")