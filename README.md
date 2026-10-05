Overview
Task Overview
Problem Statement
The goal of this competition is to predict the future sale price of houses in London based on a range of features that capture the characteristics of the property and the context of the sale. The dataset includes various attributes of each property such as the number of bedrooms, bathrooms, property type, size, geographical location, and energy rating. Additionally, each sale transaction is time-stamped with the month and year of sale, providing an opportunity to incorporate temporal trends into the predictions.

Participants are tasked with developing a machine learning model that can accurately predict the sale price of properties based on these features. The models should take into account not only the static features of each property but also how market conditions and property characteristics evolve over time.

Start

Nov 9, 2024
Close

Nov 8, 2025
Description
Given the current state of the housing market and the features of each property, participants should:

Develop a regression model to predict future house prices based on the provided features.
Account for seasonal, regional, and market-specific trends that could affect property prices over time.
Handle missing data, outliers, and multicollinearity to improve model performance.
Provide insights into which factors most significantly affect house prices in London.
Evaluation
The performance of the model will be evaluated based on the Mean Absolute Error (MAE) between the predicted prices and the actual sale prices. MAE is a metric that measures the average magnitude of errors in a set of predictions, without considering their direction. It gives an idea of how far off the predictions are from the actual values, with lower values indicating better predictive accuracy.

Citation
jake wright. London House Price Prediction: Advanced Techniques. https://www.kaggle.com/competitions/london-house-price-prediction-advanced-techniques, 2024. Kaggle.


Dataset Description
London House Price Prediction Dataset Overview This dataset contains information about property sales in London, with each entry representing a sale transaction. Each property sale has detailed information such as the full address, geographical location, number of rooms, property type, energy rating, and the sale price. The dataset includes both numeric and categorical features that can be used to predict the sale price of properties based on various factors such as location, property type, and features of the house.

Columns fullAddress

Type: String Description: The full address of the property, including street name, city, and postal code. postcode

Type: String Description: The postal code of the property. country

Type: String Description: The country in which the property is located (e.g., "England"). outcode

Type: String Description: The outward part of the postcode, which typically represents a district or region. latitude

Type: Float Description: The geographical latitude of the property. longitude

Type: Float Description: The geographical longitude of the property. bathrooms

Type: Float Description: The number of bathrooms in the property. Missing values are represented by NaN. bedrooms

Type: Float Description: The number of bedrooms in the property. floorAreaSqM

Type: Float Description: The floor area of the property in square meters. livingRooms

Type: Float Description: The number of living rooms in the property. Missing values are represented by NaN. tenure

Type: String Description: The type of ownership of the property, such as "Freehold" or "Leasehold". propertyType

Type: String Description: The type of property, such as "Flat/Maisonette" or "Detached House". currentEnergyRating

Type: String Description: The current energy rating of the property, such as "A", "B", "C", or "None" if not available. sale_month

Type: Integer Description: The month in which the property was sold (1–12). sale_year

Type: Integer Description: The year in which the property was sold. price

Type: Float Description: The sale price of the property in GBP.

# London-House-Price-Prediction-CS-4661-02-Project
https://www.kaggle.com/competitions/london-house-price-prediction-advanced-techniques
