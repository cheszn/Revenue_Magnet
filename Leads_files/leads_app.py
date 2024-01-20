
import streamlit as st
import pandas as pd
import joblib
import altair as alt
import plotly.express as px

# Load the saved model - Ensure the model file is in the correct path
# Load the saved model - Ensure the model file is in the correct path
try:
    marketing = joblib.load(r"./xgboost_classifier.joblib")
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
# Main app structure
def online_input():
    st.header("User Input for Input features")
    # Collecting user input
    # Numerical features
    # Categorical features
    lead_origin = st.selectbox("Lead Origin", ['API', 'Landing Page Submission', 'Lead Add Form', 'Lead Import',
                                                      'Quick Add Form'])
    lead_source = st.selectbox("Lead Source", ['Drift', 'Organic Search', 'Direct Traffic', 'Google',
                                                      'Referral Sites', 'LinkedIn', 'Reference', 'Facebook',
                                                      'blog', 'Pay per Click Ads', 'bing', 'Social Media', 'Click2call',
                                                      'Press_Release'])
    do_not_email = st.selectbox("Do Not Email", ['Yes', 'No'])
    total_time_spent_on_website = st.number_input('Total Time Spent on Website (minutes)', step=1, format="%d")
    last_activity = st.selectbox("Last Activity", ['Page Visited on Website', 'Email Opened', 'Unreachable',
                                                      'Converted to Lead', 'Drift Chat Conversation', 'Email Bounced',
                                                      'Email Link Clicked', 'Form Submitted on Website', 'Unsubscribed',
                                                      'Had a Phone Conversation', 'View in browser link Clicked',
                                                      'Approached upfront', 'SMS Sent', 'Visited Booth in Tradeshow',
                                                      'Resubscribed to emails', 'Email Received', 'Email Marked Spam'])
    lead_quality = st.selectbox("Lead Quality", ['Low in Relevance', 'Might be', 'Not Sure', 'Worst',
                                                      'High in Relevance'])
    lead_profile = st.selectbox("Lead Profile", ['Select', 'Potential Lead', 'Other Marketing', 'Lateral Student',
                                                      'Dual Specialization Student', 'Student of SomeSchool'])
    asymmetrique_profile_score = st.number_input('Asymmetrique Profile Score', min_value=0, max_value=200, value=0)
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


        
    if st.button('Predict'):
        # Directly pass the user input datas to the model for prediction
        prediction = marketing.predict(pd.DataFrame(user_input_data))[0]
        feedback = "Converted" if prediction == 1 else "Not Converted"
        st.write(f"Prediction: {feedback}")
# Initialize batch_data at the beginning of your script (global scope)
batch_data = None
def batch_prediction_interface():
    global batch_data  # Declare batch_data as global to modify the global instance
    st.header("Upload Data")
    uploaded_file = st.file_uploader(
        "Upload CSV for Upload Data", type=["csv"])
    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            predictions = marketing.predict(batch_data)
            batch_data['predictions'] = predictions
            st.write(batch_data)
        except Exception as e:
            st.error(f"An error occurred: check file uploaded")
# Visualization section
def show_visualizations(batch_data: pd.DataFrame):
    st.subheader("Model Insights and Data Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(bar_chart, use_container_width=True)
        # Histogram for Lead Quality
        fig_hist = px.histogram(
            batch_data,
            x='Lead Quality',
            nbins=20,
            title='Distribution of Lead Quality'
        )
        fig_hist.update_layout(
            xaxis_title='Lead Quality',
            yaxis_title='Frequency'
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    with col2:
        # Box plot for Lead Profiles
        fig_box = px.box(
            batch_data,
            y='Lead Profile',
            title='Distribution of Lead Profiles'
        )
        fig_box.update_layout(
            yaxis_title='Lead Profile'
        )
        st.plotly_chart(fig_box, use_container_width=True)
        # Bar chart for Lead Origin
        if 'Lead Origin' in batch_data.columns:
            fig_count = px.bar(
                batch_data,
                x='Lead Origin',
                title='Distribution of Lead Origin'
            )
            fig_count.update_layout(
                xaxis_title='Lead Origin',
                yaxis_title='Count',
                xaxis={'categoryorder': 'total descending'}
            )
            fig_count.update_traces(marker_color='blue')
            st.plotly_chart(fig_count, use_container_width=True)
# Main execution
st.title("Lead Conversion Prediction Application")
# Selection for prediction method
prediction_method = st.selectbox("Select Prediction Method", [
                                         "Online", "Batch"])
if prediction_method == "Online":
    online_input()
elif prediction_method == "Batch":
    batch_prediction_interface()
# Show visualizations if checkbox is checked and batch_data is available
if st.checkbox('Show visualizations'):
    try:
        if 'batch_data' in globals():
            show_visualizations(batch_data)
        else:
            st.warning(
                "No batch data available for visualization. Please upload data in Upload Data.")
    except NameError:
        st.warning(
            "No batch data available for visualization. Please upload data in Upload Data.")