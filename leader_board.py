# -*- coding: utf-8 -*-
"""WE_for_SHE.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/16ryd4XiLVNm2_QF69PJLpk2QkbB0xGPP
"""

# Install required libraries in Google Colab
!pip install pandas numpy scikit-learn tensorflow keras-tuner

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt

# Load the data
file_path = '/content/fashion_data_2018_2022.csv'  # Adjust the path accordingly
data = pd.read_csv(file_path)

# Select relevant features and target
features = data[['category', 'pattern', 'color', 'season', 'price', 'material', 'sales_count', 'reviews_count', 'average_rating', 'out_of_stock_times', 'brand', 'discount', 'wish_list_count', 'month_of_sale']]
target = data[['product_name', 'gender', 'age_group']]

# Handle missing values
for column in features.select_dtypes(include=['number']).columns:
    features[column].fillna(features[column].median(), inplace=True)

for column in features.select_dtypes(include=['object']).columns:
    features[column].fillna(features[column].mode()[0], inplace=True)

# Encode categorical features and target variables
label_encoders = {}
for column in features.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    features[column] = le.fit_transform(features[column])
    label_encoders[column] = le

for column in target.columns:
    le = LabelEncoder()
    target[column] = le.fit_transform(target[column])
    label_encoders[column] = le

# Normalize numerical features
scaler = StandardScaler()
numerical_features = features.select_dtypes(include=['number']).columns
features[numerical_features] = scaler.fit_transform(features[numerical_features])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Convert targets to categorical (one-hot encoding)
y_train_product_name = to_categorical(y_train['product_name'])
y_train_gender = to_categorical(y_train['gender'])
y_train_age_group = to_categorical(y_train['age_group'])

y_test_product_name = to_categorical(y_test['product_name'])
y_test_gender = to_categorical(y_test['gender'])
y_test_age_group = to_categorical(y_test['age_group'])

# Define the hypermodel for Keras Tuner
def build_model(hp, output_units):
    model = Sequential()
    model.add(Dense(hp.Int('units', min_value=32, max_value=512, step=32), activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)))
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(Dense(hp.Int(f'units_{i}', min_value=32, max_value=512, step=32), activation='relu'))
        model.add(Dropout(hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(output_units, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Perform hyperparameter tuning
def perform_tuning(X_train, y_train, output_units, project_name):
    tuner = kt.RandomSearch(
        lambda hp: build_model(hp, output_units),
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=3,
        directory='my_dir',
        project_name=project_name
    )
    tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=3)])
    return tuner.get_best_models(num_models=1)[0]

# Tune and train the model for product_name
output_units_product_name = y_train_product_name.shape[1]
best_model_product_name = perform_tuning(X_train, y_train_product_name, output_units_product_name, 'product_name_tuning')

# Train the best model for product_name
history_product_name = best_model_product_name.fit(X_train, y_train_product_name, epochs=50, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=3)])

# Evaluate the model for product_name
loss_product_name, accuracy_product_name = best_model_product_name.evaluate(X_test, y_test_product_name)
print(f"Best model for product_name accuracy: {accuracy_product_name:.4f}")

# Repeat the process for gender and age_group
best_models = {'product_name': best_model_product_name}
accuracies = {'product_name': accuracy_product_name}

for target_name, y_train_target, y_test_target, project_name in zip(['gender', 'age_group'], [y_train_gender, y_train_age_group], [y_test_gender, y_test_age_group], ['gender_tuning', 'age_group_tuning']):
    output_units = y_train_target.shape[1]
    best_model = perform_tuning(X_train, y_train_target, output_units, project_name)
    history = best_model.fit(X_train, y_train_target, epochs=50, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=3)])
    loss, accuracy = best_model.evaluate(X_test, y_test_target)
    best_models[target_name] = best_model
    accuracies[target_name] = accuracy
    print(f"Best model for {target_name} accuracy: {accuracy:.4f}")

# Check if there is data for the year 2022
if data[data['year_of_sale'] == 2022].empty:
    print("No data available for the year 2022.")
else:
    # Predict for year_of_sale 2022
    data_2022 = data[data['year_of_sale'] == 2022].copy()
    features_2022 = data_2022[['category', 'pattern', 'color', 'season', 'price', 'material', 'sales_count', 'reviews_count', 'average_rating', 'out_of_stock_times', 'brand', 'discount', 'wish_list_count', 'month_of_sale']]

    # Handle missing values and encode categorical features
    for column in features_2022.select_dtypes(include=['number']).columns:
        features_2022[column].fillna(features[column].median(), inplace=True)

    for column in features_2022.select_dtypes(include=['object']).columns:
        features_2022[column].fillna(features[column].mode()[0], inplace=True)

    for column in features_2022.select_dtypes(include=['object']).columns:
        features_2022[column] = label_encoders[column].transform(features_2022[column])

    features_2022[numerical_features] = scaler.transform(features_2022[numerical_features])

    # Make predictions for 2022 data
    predictions_2022 = {}
    for target_name, model in best_models.items():
        predictions = model.predict(features_2022)
        predictions_2022[target_name] = np.argmax(predictions, axis=1)

    # Decode the predictions back to original labels
    decoded_predictions_2022 = {}
    for column in predictions_2022:
        decoded_predictions_2022[column] = label_encoders[column].inverse_transform(predictions_2022[column])

    # Create a detailed report
    report = data_2022[['category', 'pattern', 'color', 'season', 'price', 'material', 'sales_count', 'reviews_count', 'average_rating', 'out_of_stock_times', 'brand', 'discount', 'wish_list_count', 'month_of_sale', 'product_name', 'gender', 'age_group']].copy()
    report['predicted_product_name'] = decoded_predictions_2022['product_name']
    report['predicted_gender'] = decoded_predictions_2022['gender']
    report['predicted_age_group'] = decoded_predictions_2022['age_group']

    # Save the report to a CSV file
    report.to_csv('/content/prediction_report_2022.csv', index=False)

    # Display the report
    print(report.head())

    # Calculate the accuracy for 2022 predictions
    accuracy_product_name_2022 = (report['product_name'] == report['predicted_product_name']).mean()
    accuracy_gender_2022 = (report['gender'] == report['predicted_gender']).mean()
    accuracy_age_group_2022 = (report['age_group'] == report['predicted_age_group']).mean()

    print(f"2022 prediction accuracy for product_name: {accuracy_product_name_2022:.4f}")
    print(f"2022 prediction accuracy for gender: {accuracy_gender_2022:.4f}")
    print(f"2022 prediction accuracy for age_group: {accuracy_age_group_2022:.4f}")