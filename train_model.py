import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from housing_data import load_data
# Load dataset
df = load_data()
# Separate input and output
X = df[["area", "bedrooms", "age"]]
y = df["price"]
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.2, random_state=42
)
# Train model
model = LinearRegression()
model.fit(X_train, y_train)
# Save trained model
joblib.dump(model, "model.pkl")
print("Model trained and saved successfully.")