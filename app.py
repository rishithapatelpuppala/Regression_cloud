import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.title("Simple Linear Regression - Salary Prediction")
st.header("Step 1: Load and Display Data")
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.dataframe(data)

    st.header("Step 2: Visualize Salary vs Experience")
    fig, ax = plt.subplots()
    ax.scatter(data["YearsExperience"], data["Salary"], color='blue')
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary")
    ax.set_title("Scatter Plot")
    st.pyplot(fig)

    st.header("Step 3: Train Linear Regression Model")
    X = data[["YearsExperience"]]
    y = data["Salary"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    st.success("Model Trained Successfully!")

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    st.info(f"Model Accuracy (R² Score): {r2:.2f}")

    st.header("Step 4: Visualize Regression Line")
    fig2, ax2 = plt.subplots()
    ax2.scatter(X, y, color='blue', label='Actual Data')
    ax2.plot(X, model.predict(X), color='red', label='Regression Line')
    ax2.set_xlabel("Years of Experience")
    ax2.set_ylabel("Salary")
    ax2.legend()
    st.pyplot(fig2)

    st.header("Step 5: Predict Salary")
    experience = st.number_input("Enter years of experience:", min_value=0.0, step=0.1)
    if st.button("Predict Salary"):
        prediction = model.predict([[experience]])
        st.success(f"Predicted Salary: ${prediction[0]:,.2f}")
        