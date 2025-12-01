Auto MPG Prediction Project Using Random Forest Model
Project Objective:
This project aims to predict auto fuel consumption (MPG) based on various vehicle characteristics such as number of cylinders,
engine size, engine power, vehicle weight, acceleration, model year and country of origin.
For this, the Random Forest Regressor model has been used, which is one of the powerful machine learning models.
Project Steps:
Data Loading and Preprocessing:
Data was loaded from a CSV file named auto-mpg.csv, which included various vehicle characteristics and their fuel consumption.
The horsepower column data, which contained non-numeric values, was converted to numeric values ​​and all missing values ​​(NaN) were removed from the dataset.
Outlier Removal:
To prevent outliers from affecting the model performance,
the IQR (Interquartile Range) method was used to identify and remove outliers in important columns such as mpg, horsepower, acceleration, weight, and displacement.
Convert categorical features to numeric features:
The origin column, which specified the country of origin of the cars, was converted to binary features using the One-Hot Encoding technique.
This created three binary features for the country of origin of the cars (origin_2, origin_3) and removed the reference feature (origin_1).
Data Preparation:
Input features (X) were created from the columns cylinders, displacement, horsepower
, weight, acceleration, model_year, and new features origin_2, origin_3.
The target feature (y), which is MPG, was separated.
The data was divided into two sets of training and testing (80% training and 20% testing).
Data Standardization:
To ensure better model performance, StandardScaler was used to standardize the data.
This caused the features to reach a similar scale and the model was trained with higher accuracy.
Random Forest Model Training:
A Random Forest Regressor model with 200 decision trees was used to predict fuel consumption (MPG).
The model was able to learn the nonlinear and complex relationships in the data and achieve more accurate predictions.
Model Evaluation:
The model performance was evaluated using Mean Squared Error (MSE). This metric measures the amount of prediction error of the model.
The model made predictions based on the test data and the amount of error (MSE) was calculated.
Saving the Model and Scaler:
After training the model, the Random Forest model and the StandardScaler scaler were saved in .joblib files so that they could be loaded for later use.
User Interface (GUI) Using Tkinter:
To facilitate the use of the model, a graphical user interface (GUI) was built using Tkinter.
This interface allows the user to enter various vehicle characteristics and view the predicted fuel consumption.
The inputs to this user interface include: number of cylinders, engine displacement, engine power, vehicle weight, acceleration, model year, and country of origin.
After entering the values, the model displays the fuel consumption (MPG) prediction to the user.
Summary of Results:
The Mean Squared Error (MSE) of the model was calculated to check the accuracy of the fuel consumption (MPG) prediction.
A simple user interface was built that allows users to enter vehicle characteristics and view the fuel consumption prediction.
Technologies used:
Python: Main programming language.
Pandas and NumPy: For data processing.
Scikit-learn: For machine modeling and data standardization.
Tkinter: For creating a graphical user interface (GUI).
Joblib: For model storage and scaler.
End result:
This project well demonstrates the ability of machine learning models such as Random Forest to predict numerical values ​​(here, fuel consumption of cars).
Also, by using a graphical user interface, predicting car fuel consumption has become easily possible for any user.



