import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Load the dataset
sonar_data = pd.read_csv('/content/copy.csv', header=None)

# EDA: Data summary
print(f"Shape of the dataset: {sonar_data.shape}")
print(sonar_data.describe())

# EDA: Class distribution
print("Class distribution:")
print(sonar_data[60].value_counts())

# Visualize class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=sonar_data[60])
plt.title("Class Distribution: Rock vs Mine")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# Correlation Matrix (excluding target column)
correlation_matrix = sonar_data.drop(columns=60).corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title("Correlation Matrix of Features")
plt.show()

# Pairplot for feature relationships (select first 5 features for clarity)
sns.pairplot(sonar_data.iloc[:, :5])
plt.suptitle("Pairplot of First 5 Features", y=1.02)
plt.show()

# Boxplot of features (example: first 5 features)
plt.figure(figsize=(10, 6))
sns.boxplot(data=sonar_data.iloc[:, :5])
plt.title("Boxplot of First 5 Features")
plt.xlabel("Features")
plt.ylabel("Values")
plt.show()

# Separating data and labels
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

# Encode target variable ('R' -> 0, 'M' -> 1)
Y = Y.map({'R': 0, 'M': 1})

# Splitting data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Comparing multiple models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    "SVM": SVC(kernel='linear', probability=True),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

results = {}
for model_name, model in models.items():
    # Perform 5-fold cross-validation
    scores = cross_val_score(model, X_train_scaled, Y_train, cv=5, scoring='accuracy')
    results[model_name] = {
        "mean_accuracy": np.mean(scores),
        "std_dev": np.std(scores)
    }
    # Train and test on the held-out test set
    model.fit(X_train_scaled, Y_train)
    Y_test_pred = model.predict(X_test_scaled)
    test_accuracy = accuracy_score(Y_test, Y_test_pred)
    results[model_name]["test_accuracy"] = test_accuracy

# Convert results to a DataFrame for visualization
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values(by="mean_accuracy", ascending=False)

# Plot the model comparison
plt.figure(figsize=(12, 6))
sns.barplot(x=results_df.index, y="mean_accuracy", data=results_df)
plt.title("Model Comparison (Cross-Validation Accuracy)")
plt.ylabel("Mean Accuracy")
plt.xticks(rotation=45)
plt.show()

# Train the best model (Random Forest, based on earlier results)
best_model = RandomForestClassifier(random_state=42, n_estimators=100)
best_model.fit(X_train_scaled, Y_train)

# Confusion Matrix
Y_test_pred_best = best_model.predict(X_test_scaled)
conf_matrix = confusion_matrix(Y_test, Y_test_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Rock", "Mine"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix for Best Model (Random Forest)")
plt.show()

# Function for user input
def user_input_prediction(model, scaler):
    print("Enter 60 features as comma-separated values (e.g., 0.02,0.03,0.04,...):")
    try:
        input_data = input("Enter features: ")
        input_data = list(map(float, input_data.split(',')))
        if len(input_data) != 60:
            print("Error: Please enter exactly 60 features.")
            return

        # Convert input to numpy array and reshape for prediction
        input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
        input_data_scaled = scaler.transform(input_data_as_numpy_array)
        prediction = model.predict(input_data_scaled)
        prediction_label = 'Rock' if prediction[0] == 0 else 'Mine'
        print(f"The object is classified as: {prediction_label}")
    except ValueError:
            print("Invalid input. Please enter numeric values separated by commas.")

# Call the user input function
user_input_prediction(best_model,scaler)