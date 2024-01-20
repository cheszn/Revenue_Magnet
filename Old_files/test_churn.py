import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Set the tab title for your Streamlit app
st.set_page_config(page_title="Revenue Magnet")

# Load the saved model
churn_model = joblib.load('Churn_files/random_forest_classifier.joblib')

# Initialize session states for login
if 'username' not in st.session_state:
    st.session_state['username'] = None
if 'password' not in st.session_state:
    st.session_state['password'] = None
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

# User credentials
user_credentials = {
    'sales': 'sales',
    'marketing': 'marketing',
    'csops': 'csops'
}

# User authentication function


def authenticate(username, password):
    return username in user_credentials and user_credentials[username] == password

# Login form


def show_login_form():
    st.sidebar.header("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type='password')
    if st.sidebar.button("Login"):
        if authenticate(username, password):
            st.session_state['username'] = username
            st.session_state['password'] = password
            st.session_state['authenticated'] = True
            st.experimental_rerun()
        else:
            st.sidebar.error("Incorrect username or password")

# Online input function


def online_input():
    st.sidebar.header("User Input for Online Prediction")

    # Collecting user input
    points_in_wallet = st.sidebar.number_input('Points in Wallet')
    opportunity_size_usd = st.sidebar.number_input('Opportunity Size USD')
    days_since_last_login = st.sidebar.number_input('Days Since Last Login')
    feedback = st.sidebar.selectbox("Last Feedback", ['Support Response Delays', 'Frequent Software Downtime', 'Unhelpful Online Resources',
                                    'Reliable Service Availability', 'Proactive Issue Resolution', 'Value for Money Solutions', 'Excessive Marketing Communications', 'Intuitive User Interface'])
    joined_through_referral = st.sidebar.selectbox(
        "Joined Through Referral", ['Yes', 'No'])
    preferred_offer_types = st.sidebar.selectbox("Preferred Offer Types", [
                                                 'Gift Vouchers/Coupons', 'Credit/Debit Card Offers', 'Without Offers'])
    technology_primary = st.sidebar.selectbox("Technology Primary", [
                                              'Analytics', 'ERP Implementation', 'Legacy Modernization', 'Technical Business Solutions'])
    past_complaint = st.sidebar.selectbox("Past Complaint", ['Yes', 'No'])

    # Create a DataFrame from user inputs
    user_input_data = pd.DataFrame({
        'points_in_wallet': [points_in_wallet],
        'opportunity_size_usd': [opportunity_size_usd],
        'days_since_last_login': [days_since_last_login],
        'feedback': [feedback],
        'joined_through_referral': [joined_through_referral],
        'preferred_offer_types': [preferred_offer_types],
        'technology_primary': [technology_primary],
        'past_complaint': [past_complaint]
    })

    # Process user input and predict
    if st.sidebar.button('Predict'):
        # Directly pass the user input data to the model for prediction
        prediction = churn_model.predict(user_input_data)[0]
        prediction_text = "The Customer will terminate the contract" if prediction == 1 else "The Customer will not terminate the contract"

        # Define font size and color based on prediction
        font_size = 24  # You can adjust the font size
        font_color = "red" if prediction == 1 else "green"

        # Display the prediction with customized font size and color
        st.markdown(
            f"<p style='font-size:{font_size}px; color:{font_color};'>{prediction_text}</p>",
            unsafe_allow_html=True
        )

# Batch data function


def batch_prediction_interface():
    st.sidebar.header("Batch Prediction")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV for Batch Prediction", type=["csv"])

    if uploaded_file is not None:
        # Saving the uploaded file to session state so that it persists
        st.session_state['uploaded_file'] = uploaded_file

        if st.sidebar.button('Predict'):
            try:
                # Read the data from the session state
                batch_data = pd.read_csv(st.session_state['uploaded_file'])
                predictions = churn_model.predict(batch_data)

                # Save the predictions to session state
                st.session_state['batch_data'] = batch_data
                st.session_state['predictions'] = predictions

                # Adjust the ID column to start at 1 for batch_data
                batch_data.index = np.arange(1, len(batch_data) + 1)

                # Create a separate DataFrame for predictions
                prediction_data = pd.DataFrame({
                    'ID': range(1, len(predictions) + 1),
                    'Prediction': ['The Customer will terminate the contract' if pred == 1 else 'The Customer will not terminate the contract' for pred in predictions]
                })

                # Display the original data and prediction data separately without the index column
                st.write("Batch Data:")
                st.dataframe(batch_data, use_container_width=True)

                st.write("Predictions:")
                st.dataframe(prediction_data.set_index(
                    'ID'), use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred: {e}")

# Visualization section


def show_visualizations():
    if 'batch_data' in st.session_state and not st.session_state['batch_data'].empty:
        batch_data = st.session_state['batch_data']
        st.subheader("Data Visualizations")

        col1, col2 = st.columns(2)

        # Histogram for Points in Wallet
        with col1:
            try:
                fig_hist = px.histogram(
                    batch_data, x='points_in_wallet', nbins=20, title='Distribution of Points in Wallet')
                fig_hist.update_layout(
                    xaxis_title='Points in Wallet', yaxis_title='Count')
                st.plotly_chart(fig_hist, use_container_width=True)
            except Exception as e:
                st.error(
                    f"An error occurred while creating the histogram: {e}")

        # Box plot for Opportunity Size USD
        with col2:
            try:
                fig_box = px.box(batch_data, y='opportunity_size_usd',
                                 title='Distribution of Opportunity Size USD')
                fig_box.update_layout(yaxis_title='Opportunity Size USD')
                st.plotly_chart(fig_box, use_container_width=True)
            except Exception as e:
                st.error(f"An error occurred while creating the box plot: {e}")

        # Bar chart for Feedback Distribution
        try:
            if 'feedback' in batch_data.columns:
                fig_count = px.bar(batch_data, x='feedback',
                                   title='Distribution of Feedback Types')
                fig_count.update_layout(xaxis_title='Feedback', yaxis_title='Count', xaxis={
                                        'categoryorder': 'total descending'})
                fig_count.update_traces(marker_color='blue')
                st.plotly_chart(fig_count, use_container_width=True)
        except Exception as e:
            st.error(f"An error occurred while creating the bar chart: {e}")
    else:
        st.error(
            "No data available to show visualizations. Please upload data and predict.")

# Main app function


def main_app():
    st.title("Churn Prediction Application")

    # Selection for prediction method
    prediction_method = st.sidebar.selectbox("Select Prediction Method", [
                                             "Online Prediction", "Batch Prediction"], key='prediction_method')

    if prediction_method == "Online Prediction":
        online_input()
    elif prediction_method == "Batch Prediction":
        batch_prediction_interface()

    if st.checkbox('Show visualizations'):
        if 'batch_data' in st.session_state and not st.session_state['batch_data'].empty:
            show_visualizations()
        else:
            st.warning(
                "No batch data available for visualization. Please upload data and predict.")


# Check if the user is authenticated
if not st.session_state['authenticated']:
    show_login_form()
else:
    main_app()
