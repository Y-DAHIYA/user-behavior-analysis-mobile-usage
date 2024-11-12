import streamlit as st
import pickle
import pandas as pd

# Load saved models and encoders using pickle
with open('random_forest_model.pkl', 'rb') as f:
    rf_clf = pickle.load(f)
with open('gradient_boosting_model.pkl', 'rb') as f:
    gb_clf = pickle.load(f)
with open('device_model_encoder.pkl', 'rb') as f:
    device_model_encoder = pickle.load(f)
with open('os_encoder.pkl', 'rb') as f:
    os_encoder = pickle.load(f)
with open('gender_encoder.pkl', 'rb') as f:
    gender_encoder = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the column order
with open("columns_order.pkl", "rb") as f:
    columns_order = pickle.load(f)

# Define the function to make predictions
def predict(model, input_data):
    prediction = model.predict(input_data)
    return prediction

# Set the title of the app
st.title("User Behavior Prediction Based on Mobile Usage")

# Add sidebar for user input
st.sidebar.header("Input Features")

# Input fields for each feature
device_model = st.sidebar.selectbox("Device Model", device_model_encoder.classes_)
os = st.sidebar.selectbox("Operating System", os_encoder.classes_)
usage_min = st.sidebar.number_input("Usage Time (minutes)", min_value=0, max_value=1000, value=30)
screen_time_hr = st.sidebar.number_input("Screen Time (hours)", min_value=0, max_value=24, value=3)
data_usage_mb = st.sidebar.number_input("Data Usage (MB)", min_value=0, max_value=5000, value=500)
num_apps = st.sidebar.number_input("Number of Apps Installed", min_value=0, max_value=100, value=15)
age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=25)
gender = st.sidebar.selectbox("Gender", gender_encoder.classes_)

# Encode categorical variables using the saved LabelEncoders
device_model_encoded = device_model_encoder.transform([device_model])[0]
os_encoded = os_encoder.transform([os])[0]
gender_encoded = gender_encoder.transform([gender])[0]

# Arrange and scale data
input_data = pd.DataFrame([[device_model_encoded, os_encoded, usage_min, screen_time_hr, data_usage_mb, num_apps, age, gender_encoded]],
                          columns=["device_model", "os", "usage_min", "screen_time_hr", "data_usage_mb", "num_apps", "age", "gender"])

# Reorder input_data to match the column order used during training
input_data = input_data.reindex(columns=columns_order)


# Scale numerical features
scaled_features = scaler.transform(input_data[["usage_min", "screen_time_hr", "data_usage_mb", "num_apps", "age"]])
input_data[["usage_min", "screen_time_hr", "data_usage_mb", "num_apps", "age"]] = scaled_features

# Model selection
model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "Gradient Boosting"])

# Prediction button
if st.sidebar.button("Predict"):
    # Use the selected model for prediction
    model = rf_clf if model_choice == "Random Forest" else gb_clf
    prediction = predict(model, input_data)

    # Display the prediction result
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.write("User is likely to be a **high-engagement** user.")
    else:
        st.write("User is likely to be a **low-engagement** user.")

    # Display input data
    st.subheader("User Input Data")
    st.write(input_data)
