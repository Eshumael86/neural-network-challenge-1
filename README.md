# neural-network-challenge-1
# Student Loan Repayment Prediction using Neural Networks

## Overview
This project develops a deep learning model to predict student loan repayment based on historical borrower data. The dataset includes various features about student loan recipients, and the target variable is the "credit_ranking" column, which indicates the likelihood of repayment.

## Dataset
The dataset is loaded from the following source:
[Student Loans Dataset](https://static.bc-edx.com/ai/ail-v-1-0/m18/lms/datasets/student-loans.csv)

### Features
- Various borrower-related attributes
- Target variable: `credit_ranking`

## Steps
1. **Data Preprocessing**
   - Load dataset into a Pandas DataFrame
   - Define features (`X`) and target (`y`)
   - Split data into training and testing sets
   - Standardize the feature data using `StandardScaler`

2. **Model Development**
   - Construct a neural network using TensorFlow/Keras
   - Define input, hidden, and output layers
   - Compile the model with Adam optimizer and binary cross-entropy loss function
   - Train the model using training data

3. **Model Evaluation**
   - Assess performance using test data
   - Measure accuracy and loss
   - Make predictions and compare with actual values

## Requirements
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- TensorFlow/Keras

## Running the Model
1. Clone this repository.
2. Install dependencies using:
   ```sh
   pip install pandas numpy scikit-learn tensorflow
   ```
3. Run the Jupyter Notebook or script.

## Results
The trained model predicts the likelihood of student loan repayment. The accuracy of the model is assessed on test data, and results are compared between actual and predicted values.

## Future Improvements
- Tune hyperparameters for better accuracy.
- Experiment with additional features.
- Implement alternative machine learning models for comparison.

## Author
Eshumael Manhanzva

