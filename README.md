# kaggle_sales_predictor

This repository contains a sales prediction model developed for the **Store Sales Prediction** competition on Kaggle. The model leverages **XGBRegressor** from the XGBoost library and employs **Grid Search** to optimize the hyperparameters, achieving accurate sales predictions based on historical sales data and various features.

## Problem Statement
The goal of the competition is to predict the future sales of different products across multiple stores based on historical sales data and other relevant features. Accurate sales predictions help businesses with inventory management, demand forecasting, and optimizing supply chains.

## Approach
### 1. Data Preprocessing
- Data Cleaning
- Feature Engineering
- Handling Missing Values
- Encoding Categorical Features
- Data Splitting into Train and Test Sets

### 2. Model Selection
The model selected for this task is **XGBRegressor** from the **XGBoost** library, known for its efficiency and high performance in regression tasks.

### 3. Hyperparameter Tuning
**Grid Search** was used to find the optimal hyperparameters for the XGBRegressor model, ensuring better performance and generalization.


## Model Training
1. Load the preprocessed dataset.
2. Split data into training and validation sets.
3. Define the XGBRegressor model.
4. Set up a parameter grid for Grid Search.
5. Perform Grid Search with Cross-Validation.
6. Train the best model on the training data.
7. Evaluate performance on validation data.

## Evaluation Metrics
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)


## Results
The final model demonstrated strong predictive performance on the validation set, achieving a score of 2.19078 on Kaggle

## Future Improvements
- Incorporate additional external features such as holidays, promotions, and weather data.
- Experiment with other advanced models like LightGBM, CatBoost, or ensemble methods.
- Implement more sophisticated feature engineering techniques.

## Acknowledgments
- **Kaggle** for hosting the [Store Sales Prediction](https://www.kaggle.com/competitions/store-sales-prediction) competition.
- Open-source libraries and the data science community for providing valuable tools and resources.

## License
This project is licensed under the MIT License.

