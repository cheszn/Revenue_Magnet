
import streamlit as st
import pandas as pd
import joblib
import altair as alt
import plotly.express as px
import xgboost as xgb

# Load the saved model - Ensure the model file is in the correct path
try:
    leads_model = joblib.load("Leads_files/xgboost_classifier.joblib")
except FileNotFoundError:
    st.error('Model file not found. Please check the file path.')

# Generate remarks based on prediction and data row


def generate_remarks(prediction, data_row):
    remarks = ""
    if prediction == 1:
        if data_row['Total Time Spent on Website'] > 1000:
            remarks = "Customers who spent longer time on website tend to convert."
    else:
        if data_row['Total Time Spent on Website'] < 1000:
            remarks = "Customers might not convert"
    return remarks

# Function to collect user input for online prediction


def online_input():
    st.sidebar.header("User Input for Online Prediction")

    # Collecting user input for various features
    # Categorical and numerical features are collected using sidebar widgets
    lead_origin = st.sidebar.selectbox("Lead Origin", [
                                       'API', 'Landing Page Submission', 'Lead Add Form', 'Lead Import', 'Quick Add Form'])
    lead_source = st.sidebar.selectbox("Lead Source", ['Drift', 'Organic Search', 'Direct Traffic', 'Google', 'Referral Sites',
                                       'LinkedIn', 'Reference', 'Facebook', 'blog', 'Pay per Click Ads', 'bing', 'Social Media', 'Click2call', 'Press_Release'])
    do_not_email = st.sidebar.selectbox("Do Not Email", ['Yes', 'No'])
    total_time_spent_on_website = st.sidebar.number_input(
        'Total Time Spent on Website (minutes)', step=1, format="%d")
    last_activity = st.sidebar.selectbox("Last Activity", ['Page Visited on Website', 'Email Opened', 'Unreachable', 'Converted to Lead', 'Drift Chat Conversation', 'Email Bounced', 'Email Link Clicked', 'Form Submitted on Website',
                                         'Unsubscribed', 'Had a Phone Conversation', 'View in browser link Clicked', 'Approached upfront', 'SMS Sent', 'Visited Booth in Tradeshow', 'Resubscribed to emails', 'Email Received', 'Email Marked Spam'])
    lead_quality = st.sidebar.selectbox("Lead Quality", [
                                        'Low in Relevance', 'Might be', 'Not Sure', 'Worst', 'High in Relevance'])
    lead_profile = st.sidebar.selectbox("Lead Profile", [
                                        'Select', 'Potential Lead', 'Other Leads', 'Lateral Student', 'Dual Specialization Student', 'Student of SomeSchool'])
    asymmetrique_profile_score = st.sidebar.number_input(
        'Asymmetrique Profile Score', min_value=0, max_value=200, value=0)

    # Creating a DataFrame from user inputs
    user_input_data = pd.DataFrame({
        'lead_origin': [lead_origin],
        'lead_source': [lead_source],
        'do_not_email': [do_not_email],
        'total_time_spent_on_website': [total_time_spent_on_website],
        'last_activity': [last_activity],
        'lead_quality': [lead_quality],
        'lead_profile': [lead_profile],
        'asymmetrique_profile_score': [asymmetrique_profile_score]
    })

    if st.sidebar.button('Predict'):
        # Predicting using the model and displaying results
        prediction = leads_model.predict(user_input_data)[0]
        feedback = "The lead will be converted" if prediction == 1 else "The lead will not be converted"
        st.write(f"Prediction: {feedback}")

# Function to handle batch prediction


def batch_prediction_interface():
    st.sidebar.header("Batch Prediction")
    uploaded_file = st.file_uploader(
        "Upload CSV for Batch Prediction", type=["csv"])

    if uploaded_file is not None:
        try:
            # Reading the uploaded file
            batch_data = pd.read_csv(uploaded_file)
            # Generating predictions
            predictions = leads_model.predict(batch_data)
            # Adding an ID column for reference
            batch_data['ID'] = range(1, len(batch_data) + 1)
            # Displaying the original data with ID
            st.subheader("Uploaded Data")
            st.write(batch_data)

            # Creating a DataFrame for displaying predictions
            predictions_df = pd.DataFrame(
                {'ID': batch_data['ID'], 'Prediction': predictions})
            predictions_df['Prediction'] = predictions_df['Prediction'].apply(
                lambda x: "The lead will be converted" if x == 1 else "The lead will not be converted")
            # Displaying the predictions
            st.subheader("Predictions by ID")
            st.write(predictions_df)
        except Exception as e:
            st.error(f"An error occurred: {e}")


# Selection for prediction method
prediction_method = st.sidebar.selectbox(
    "Select Prediction Method", ["Online", "Batch"])

if prediction_method == "Online":
    online_input()
    if st.checkbox('Show visualizations'):
        st.warning(
            "Visualizations are not available for online prediction. Please use Batch Prediction.")
elif prediction_method == "Batch":
    batch_prediction_interface()
    if st.checkbox('Show visualizations'):
        if 'batch_data' in globals():
            show_visualizations(batch_data)
        else:
            st.warning(
                "No batch data available for visualization. Please upload data in Batch Prediction.")


# Visualization section

def show_visualizations(batch_data: pd.DataFrame):
    st.subheader("Model Insights and Data Visualizations")

    # Histogram for Lead Quality
    if 'lead_quality' in batch_data.columns:
        fig_hist = px.histogram(
            batch_data,
            x='lead_quality',
            nbins=20,
            title='Distribution of Lead Quality'
        )
        fig_hist.update_layout(
            xaxis_title='Lead Quality',
            yaxis_title='Frequency'
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Histogram plot for Lead Profile
    if 'lead_profile' in batch_data.columns:
        fig_hist = px.histogram(
            batch_data,
            x='lead_profile',
            nbins=20,
            title='Distribution of Lead Profile'
        )
        fig_hist.update_layout(
            xaxis_title='Lead Profile',
            yaxis_title='Frequency'
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Bar chart for Lead Origin
    if 'lead_origin' in batch_data.columns:
        fig_count = px.bar(
            batch_data,
            x='lead_origin',
            title='Distribution of Lead Origin'
        )
        fig_count.update_layout(
            xaxis_title='Lead Origin',
            yaxis_title='Frequency',
            xaxis={'categoryorder': 'total descending'}
        )
        st.plotly_chart(fig_count, use_container_width=True)

    # Bar chart for Lead Source
    if 'lead_source' in batch_data.columns:
        fig_count = px.bar(
            batch_data,
            x='lead_source',
            title='Distribution of Lead Source'
        )
        fig_count.update_layout(
            xaxis_title='Lead Source',
            yaxis_title='Frequency',
            xaxis={'categoryorder': 'total descending'}
        )
        st.plotly_chart(fig_count, use_container_width=True)

    # Density plot for Total Time Spent on Website
    if 'total_time_spent_on_website' in batch_data.columns:
        fig_count = px.density_contour(
            batch_data,
            x='total_time_spent_on_website',
            title='Density Plot of Total Time Spent on Website'
        )
        fig_count.update_layout(
            xaxis_title='Total Time Spent on Website',
            yaxis_title='Density',
        )
        fig_count.update_traces(contours_coloring="fill",
                                contours_showlabels=True)
        st.plotly_chart(fig_count, use_container_width=True)

    # Density Plot of Asymmetrique Profile Score
    if 'asymmetrique_profile_score' in batch_data.columns:
        fig_count = px.density_contour(
            batch_data,
            x='asymmetrique_profile_score',
            title='Density Plot of Asymmetrique Profile Score'
        )
        fig_count.update_layout(
            xaxis_title='Asymmetrique Profile Score',
            yaxis_title='Density',
        )
        fig_count.update_traces(contours_coloring="fill",
                                contours_showlabels=True)
        st.plotly_chart(fig_count, use_container_width=True)


# Main execution
st.title("Lead Conversion Prediction Application")

try:
    # Load the saved model
    leads_model = joblib.load("Leads_files/xgboost_classifier.joblib")
except FileNotFoundError:
    st.error('Model file not found. Please check the file path.')
    # Stop further execution if the model is not loaded
    st.stop()

# Existing functions with try-except blocks for error handling...

if prediction_method == "Batch":
    batch_prediction_interface()

    # Show visualizations if checkbox is checked and batch_data is available
    if st.checkbox('Show visualizations'):
        # Check if batch_data is defined
        if 'batch_data' in globals() and batch_data is not None:
            try:
                show_visualizations(batch_data)
            except Exception as e:
                st.error(
                    f"An error occurred while generating visualizations: {e}")
        else:
            st.warning(
                "No batch data available for visualization. Please upload data in Batch Prediction.")
