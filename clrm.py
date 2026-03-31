import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv("msme_gdp_india.csv")

print(data.head())

# Correlation
correlation = data["MSME_Count"].corr(data["GDP"])
print("Correlation between MSME and GDP:", correlation)

# Regression
X = data[["MSME_Count"]]
Y = data["GDP"]

model = LinearRegression()
model.fit(X, Y)

# Prediction line
y_pred = model.predict(X)

# Plot
plt.figure(figsize=(8,6))
plt.scatter(data["MSME_Count"], data["GDP"])
plt.plot(data["MSME_Count"], y_pred)
plt.xlabel("MSME Count")
plt.ylabel("GDP")
plt.title("MSME vs GDP Regression in India")
plt.show()

# Regression equation
print("Intercept:", model.intercept_)
print("Slope:", model.coef_[0])