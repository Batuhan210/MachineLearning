import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split      
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# DATA PREPARATION
data = {
    'HouseSize': [120, 250, 175, 300, 220],
    'Price': [2400000, 5000000, 3500000, 6000000, 4400000]    
    }

df = pd.DataFrame(data)

x = df[['HouseSize']]     # It's an input
y = df['Price']           # It's an output

# Testing and training the data. Use 20% for testing and 80% for training. random state: divide the data exactly by 42.
xTrain, xTest, yTrain, yTest = train_test_split(x,y, test_size = 0.2, random_state = 42)

# Create Model - Training the data
model = LinearRegression()         
model.fit(xTrain, yTrain)           


# TEST - MEAN SQUARE ERROR: 0. EVERYTHING IS OK. 
# Smaller the error, the better the prediction.
yPred = model.predict(xTest)
mse = mean_squared_error(yTest, yPred)       
rmse = np.sqrt(mse)                       # get square root
print(f"Mean square error: {rmse} 🎉 🎉 🎉")

houseSize = float(input("Enter the size of house in( m²): "))

# Create a DataFrame for the prediction to preserve the feature name.
inputData = pd.DataFrame({'HouseSize': [houseSize]})

estimatedPrice = model.predict(inputData)                  # guess the price
print(f"Estimated the price of house: {estimatedPrice[0]:.2f} ₺")
