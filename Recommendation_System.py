import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset
data_file = 'C:/Users/tahao/Downloads/archive(4)/X_train.csv'
df = pd.read_csv(data_file)

# Handle missing values
df = df.dropna()

# Create a score based on mileage, tax, and year
max_year = df['year'].max()
df['score'] = df['mileage'] + df['tax'] + df['price'] - (max_year - df['year'])

# Split the dataset into features and target
X = df.drop(columns=['carID', 'score'])
y = df['score']

# Preprocess the categorical features
X = pd.get_dummies(X, drop_first=True)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the random forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean squared error: {mse}")

# Make predictions for the entire dataset
df['predicted_score'] = model.predict(pd.get_dummies(df.drop(columns=['carID', 'score']), drop_first=True))

# Sort the cars by their predicted scores
sorted_cars = df.sort_values(by='predicted_score')

# Display the top recommended cars
top_n = 10
print(f"Top {top_n} recommended cars:")
print(sorted_cars.head(top_n))


import matplotlib.pyplot as plt

# Get the top recommended cars
top_cars = sorted_cars.head(10)

# Prepare the data for the scatter plot
x_data = top_cars['mileage']
y_data = top_cars['price']

# Create the scatter plot
plt.scatter(x_data, y_data)
plt.xlabel("mileage")
plt.ylabel("price")
plt.title(" Top 10 Recommended Cars")

# Show the plot
plt.show()

