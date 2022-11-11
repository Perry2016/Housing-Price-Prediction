import pandas as pd

df_housing = pd.read_csv("house.csv")

df_housing.head

# Creat feature x and label y

x = df_housing.drop("median_house_value", axis = 1)

y = df_housing.median_house_value

# Split the dataset

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# train the model

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x_train, y_train)

# predict housing price

y_pred = model.predict(x_test)

print('housing actual price', y_test)

print('housing predicted price', y_pred)

print('prediction score:', model.score(x_test, y_test))

# Visualization

import matplotlib.pyplot as plt

plt.scatter(x_test.median_income, y_test, color = 'brown')

plt.plot(x_test.median_income, y_pred, color = 'green', linewidth = 1)

plt.xlabel('Median Income')

plt.ylabel('Median House Vaule')

plt.show()
