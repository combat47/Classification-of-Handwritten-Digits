import tensorflow as tf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer

# Load the MNIST dataset
(x, y), (_, _) = tf.keras.datasets.mnist.load_data()

# Use the first 6000 rows of the dataset
x = x[:6000]
y = y[:6000]

# Reshape the features array to 2D array with n rows and m columns
x_flattened = x.reshape(x.shape[0], -1)

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x_flattened, y, test_size=0.3, random_state=40)

# Normalize the data
normalizer = Normalizer()
x_train_norm = normalizer.fit_transform(x_train)
x_test_norm = normalizer.transform(x_test)


# Function to fit, predict and evaluate the model
def fit_predict_eval(model, features_train, features_test, target_train, target_test):
    model.fit(features_train, target_train)
    y_pred = model.predict(features_test)
    score = accuracy_score(target_test, y_pred)
    return score


# List of models to evaluate
models = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(random_state=40),
    LogisticRegression(random_state=40, max_iter=1000),
    RandomForestClassifier(random_state=40)
]

# Evaluate each model with normalized data
accuracies = {}
for model in models:
    score = fit_predict_eval(model, x_train_norm, x_test_norm, y_train, y_test)
    model_name = model.__class__.__name__
    accuracies[model_name] = score

# Determine if normalization had a positive impact by comparing to unnormalized results
unnormalized_accuracies = {
    model.__class__.__name__: fit_predict_eval(model, x_train, x_test, y_train, y_test)
    for model in models
}
positive_impact = any(accuracies[model] > unnormalized_accuracies[model] for model in accuracies)

# Find the best two models
sorted_accuracies = sorted(accuracies.items(), key=lambda item: item[1], reverse=True)
best_two_models = sorted_accuracies[:2]

# Answer the questions
print(f'The answer to the 1st question: {"yes" if positive_impact else "no"}')
print(
    f'The answer to the 2nd question: {best_two_models[0][0]}-{best_two_models[0][1]:.3f}, {best_two_models[1][0]}-{best_two_models[1][1]:.3f}')

# Hyperparameter tuning with GridSearchCV
param_grid_knn = {'n_neighbors': [3, 4], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'brute']}
param_grid_rf = {'n_estimators': [300, 500], 'max_features': ['sqrt', 'log2'],
                 'class_weight': ['balanced', 'balanced_subsample']}

# Using normalized data as it provided the best results
x_train_final = x_train_norm
x_test_final = x_test_norm

# K-nearest Neighbors tuning
knn_grid_search = GridSearchCV(KNeighborsClassifier(), param_grid_knn, scoring='accuracy', n_jobs=-1)
knn_grid_search.fit(x_train_final, y_train)
best_knn = knn_grid_search.best_estimator_
best_knn_accuracy = accuracy_score(y_test, best_knn.predict(x_test_final))

# Random Forest tuning
rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=40), param_grid_rf, scoring='accuracy', n_jobs=-1)
rf_grid_search.fit(x_train_final, y_train)
best_rf = rf_grid_search.best_estimator_
best_rf_accuracy = accuracy_score(y_test, best_rf.predict(x_test_final))

# Print the results
print(f'K-nearest neighbours algorithm\nbest estimator: {best_knn}\naccuracy: {best_knn_accuracy:.3f}')
print(f'Random forest algorithm\nbest estimator: {best_rf}\naccuracy: {best_rf_accuracy:.3f}')
