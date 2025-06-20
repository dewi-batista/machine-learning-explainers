import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# patient dataset
dataset = {
    'Age (years)': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
    'BMI (kg/m²)': [22.5, 24.0, 26.5, 28.0, 29.5, 30.0, 31.5, 32.0, 33.5, 34.0],
    'Cholesterol Level (mg/dL)': [180, 190, 210, 220, 240, 250, 260, 270, 280, 290]
}

train_data = {
    'Age (years)': [25, 35, 45, 55, 65],
    'BMI (kg/m²)': [24.0, 26.5, 29.5, 31.5, 33.5],
    'Cholesterol Level (mg/dL)': [180, 210, 240, 260, 280]
}

test_data = {
    'Age (years)': [30, 40, 50, 60, 70],
    'BMI (kg/m²)': [24.0, 28.0, 30.0, 32.0, 34.0],
    'Cholesterol Level (mg/dL)': [190, 220, 250, 270, 290]
}

# store data in a DataFrame
df = pd.DataFrame(dataset)
df_test = pd.DataFrame(test_data)

##############################################################################
# Plotting dataset
##############################################################################

# plot Age vs BMI vs Cholesterol
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['Age (years)'], df['BMI (kg/m²)'], df['Cholesterol Level (mg/dL)'], c='r', marker='o')
ax.set_xlabel('Age (years)')
ax.set_ylabel('BMI (kg/m²)')
ax.set_zlabel('Cholesterol Level (mg/dL)')
plt.savefig('../figures/lin_reg_example/age_vs_bmi_vs_cholesterol.pdf')
plt.close(fig)

# combined plot for Age vs Cholesterol and BMI vs Cholesterol
fig, ax = plt.subplots()
ax.scatter(df['Age (years)'], df['Cholesterol Level (mg/dL)'], c='b', marker='o', label='Age (years)')
ax.scatter(df['BMI (kg/m²)'], df['Cholesterol Level (mg/dL)'], c='g', marker='x', label='BMI (kg/m²)')
ax.grid(True)
ax.set_xlabel('Feature value')
ax.set_ylabel('Cholesterol Level (mg/dL)')
ax.legend()
plt.savefig('../figures/lin_reg_example/age_and_bmi_vs_cholesterol.pdf')
plt.close(fig)

##############################################################################
# Computing the optimal parameters theta* and mean square error
##############################################################################

def lin_reg_opt_params(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

def lin_reg_output(parameters, sample):
    return np.dot(parameters, sample)

# split dataset into training and testing datasets
X_train = np.array([
    [1, 25, 22.5],
    [1, 35, 26.5],
    [1, 45, 29.5],
    [1, 55, 31.5],
    [1, 65, 33.5]
])
y_train = np.array([180, 210, 240, 260, 280])

X_test = np.array([
    [1, 30, 24.0],
    [1, 40, 28.0],
    [1, 50, 30.0],
    [1, 60, 32.0],
    [1, 70, 34.0]
])
y_test = np.array([190, 220, 250, 270, 290])

# optimal parameters
opt_params = lin_reg_opt_params(X_train, y_train)

# mse on training dataset
mse = 0
for k in range(X_train.shape[0]):
    mse += (lin_reg_output(opt_params, X_train[k]) - y_train[k]) ** 2
print("Train MSE:", mse / X_train.shape[0])

# mse on testing dataset
mse = 0
for k in range(X_test.shape[0]):
    mse += (lin_reg_output(opt_params, X_test[k]) - y_test[k]) ** 2
print("Test MSE:", mse / X_test.shape[0])

# predict trian and test
train_preds = [
    opt_params[0] + opt_params[1] * age + opt_params[2] * bmi
    for age, bmi in zip(train_data['Age (years)'], train_data['BMI (kg/m²)'])
]

test_preds = [
    opt_params[0] + opt_params[1] * age + opt_params[2] * bmi
    for age, bmi in zip(test_data['Age (years)'], test_data['BMI (kg/m²)'])
]

# plot model on training set
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(train_data['Age (years)'], train_data['BMI (kg/m²)'], np.array(train_preds), c='r', marker='o')
ax.plot(train_data['Age (years)'], train_data['BMI (kg/m²)'], train_data['Cholesterol Level (mg/dL)'], c='b', marker='o')
ax.set_xlabel('Age (years)')
ax.set_ylabel('BMI (kg/m²)')
ax.set_zlabel('Cholesterol Level (mg/dL)')
plt.savefig('../figures/lin_reg_example/model_predicts_train.pdf')
plt.close(fig)

# plot model on testing set
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(test_data['Age (years)'], test_data['BMI (kg/m²)'], np.array(test_preds), c='r', marker='o')
ax.plot(test_data['Age (years)'], test_data['BMI (kg/m²)'], test_data['Cholesterol Level (mg/dL)'], c='b', marker='o')
ax.set_xlabel('Age (years)')
ax.set_ylabel('BMI (kg/m²)')
ax.set_zlabel('Cholesterol Level (mg/dL)')
plt.savefig('../figures/lin_reg_example/model_predicts_test.pdf')
plt.close(fig)