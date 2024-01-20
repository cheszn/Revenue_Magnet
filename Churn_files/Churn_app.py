import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Load the saved model and SHAP values
customer_success = joblib.load('Churn_files/random_forest_classifier.joblib')

# defining online input function


def online_input():
    st.header("User Input for Input features")

    # Collecting user input
    points_in_wallet = st.number_input('Points in Wallet')
    opportunity_size_usd = st.number_input('Opportunity Size USD')
    days_since_last_login = st.number_input('Days Since Last Login')
    feedback = st.selectbox("Last Feedback", ['Support Response Delays', 'Frequent Software Downtime',
                                                      'Unhelpful Online Resources', 'Reliable Service Availability',
                                                      'Proactive Issue Resolution', 'Value for Money Solutions',
                                                      'Excessive Marketing Communications', 'Intuitive User Interface'])
    joined_through_referral = st.selectbox(
        "Joined Through Referral", ['Yes', 'No'])
    preferred_offer_types = st.selectbox("Preferred Offer Types", ['Gift Vouchers/Coupons',
                                                                           'Credit/Debit Card Offers',
                                                                           'Without Offers'])
    technology_primary = st.selectbox("Technology Primary", ['Analytics', 'ERP Implementation',
                                                                     'Legacy Modernization', 'Technical Business Solutions'])
    past_complaint = st.selectbox("Past Complaint", ['Yes', 'No'])

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
    if st.button('Predict'):
        # Directly pass the user input datas to the model for prediction
        prediction = customer_success.predict(user_input_data)[0]
        feedback = "The Customer will terminate the contract" if prediction == 1 else "The Customer will not terminate the contract"
        st.write(f"Prediction: {feedback}")


# Initialize batch_data at the beginning of your script (global scope)
batch_data = None

# defining batch data function
def batch_prediction_interface():
    st.header("Upload Data")
    uploaded_file = st.file_uploader(
        "Upload CSV for Upload Data", type=["csv"])

    if uploaded_file is not None:
        # Saving the uploaded file to session state so that it persists
        st.session_state['uploaded_file'] = uploaded_file

        if st.button('Predict'):
            try:
                # Read the data from the session state
                batch_data = pd.read_csv(st.session_state['uploaded_file'])
                predictions = customer_success.predict(batch_data)

                # Save the predictions to session state
                st.session_state['batch_data'] = batch_data
                st.session_state['predictions'] = predictions

                # Adjust the ID column to start at 1 for batch_data
                batch_data.index = np.arange(1, len(batch_data) + 1)

                # Create a separate DataFrame for predictions
                prediction_data = pd.DataFrame({
                    'ID': range(1, len(predictions) + 1),
                    'Prediction': [
                        'The Customer will terminate the contract' if pred == 1 else 'The Customer will not terminate the contract'
                        for pred in predictions]
                })

                # Display the original data and prediction data separately without the index column
                st.write("Batch Data:")
                st.dataframe(batch_data, use_container_width=True)

                st.write("Predictions:")
                st.dataframe(prediction_data.set_index(
                    'ID'), use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred: check file uploaded")

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
                    batch_data,
                    x='points_in_wallet',
                    nbins=20,
                    title='Distribution of Points in Wallet'
                )
                fig_hist.update_layout(
                    xaxis_title='Points in Wallet',
                    yaxis_title='Count'
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            except Exception as e:
                st.error(
                    f"An error occurred while creating the histogram: check file uploaded")

        # Box plot for Opportunity Size USD
        with col2:
            try:
                fig_box = px.box(
                    batch_data,
                    y='opportunity_size_usd',
                    title='Distribution of Opportunity Size USD'
                )
                fig_box.update_layout(
                    yaxis_title='Opportunity Size USD'
                )
                st.plotly_chart(fig_box, use_container_width=True)
            except Exception as e:
                st.error(f"An error occurred while creating the box plot: check file uploaded")

        # Bar chart for Feedback Distribution
        try:
            if 'feedback' in batch_data.columns:
                fig_count = px.bar(
                    batch_data,
                    x='feedback',
                    title='Distribution of Feedback Types'
                )
                fig_count.update_layout(
                    xaxis_title='Feedback',
                    yaxis_title='Count',
                    xaxis={'categoryorder': 'total descending'}
                )
                fig_count.update_traces(marker_color='blue')
                st.plotly_chart(fig_count, use_container_width=True)
        except Exception as e:
            st.error(f"An error occurred while creating the bar chart: check file uploaded")
    else:
        st.error(
            "No data available to show visualizations. Please upload data and predict.")


# Main execution
st.title("Churn Prediction Application")

# Selection for prediction method
prediction_method = st.selectbox("Select Prediction Method", [
                                         "Input features", "Upload Data"])

if prediction_method == "Input features":
    online_input()
elif prediction_method == "Upload Data":
    batch_prediction_interface()

# Show visualizations if checkbox is checked and batch_data is available
if st.checkbox('Show visualizations'):
    if 'batch_data' in st.session_state and not st.session_state['batch_data'].empty:
        show_visualizations()
    else:
        st.warning(
            "No batch data available for visualization. Please upload data and predict.")
