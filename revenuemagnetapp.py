from PIL import Image
import streamlit as st
import pandas as pd
import joblib
import altair as alt
import plotly.express as px
import numpy as np
from PIL import Image

# Setting Application title
st.set_page_config(page_title="Revenue Magnet")

# Load the saved models
Marketing = joblib.load("Models/xgboost_classifier.joblib")
Sales = joblib.load("Models/best_model_xgb.sav")
Customer_Success = joblib.load('Models/random_forest_classifier.joblib')


# Initialize session states for login
if 'username' not in st.session_state:
    st.session_state['username'] = None
if 'password' not in st.session_state:
    st.session_state['password'] = None
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
# User credentials
user_credentials = {
    'admin': 'admin'
}
# User authentication function


def authenticate(username, password):
    return username in user_credentials and user_credentials[username] == password

# Login form


def show_login_form():
    st.header("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    if st.button("Login"):
        if authenticate(username, password):
            st.session_state['username'] = username
            st.session_state['password'] = password
            st.session_state['authenticated'] = True
            st.experimental_rerun()
        else:
            st.error("Incorrect username or password")


# Initialize batch_data at the beginning of your script (global scope)
batch_data = None

# Global variable to store the selected model
selected_model = Marketing  # Default to marketing

# Function to handle online input for both "Marketing" and "Sales"


def online_input():
    global selected_model  # Declare selected_model as global to modify the global instance
    st.markdown("##### Please enter the following:")

    # Check the selected model
    if selected_model == Marketing:

        lead_origin = st.selectbox("Lead Origin", ['API', 'Landing Page Submission', 'Lead Add Form', 'Lead Import',
                                                   'Quick Add Form'])
        lead_source = st.selectbox("Lead Source", ['Drift', 'Organic Search', 'Direct Traffic', 'Google',
                                                   'Referral Sites', 'LinkedIn', 'Reference', 'Facebook',
                                                   'blog', 'Pay per Click Ads', 'bing', 'Social Media', 'Click2call',
                                                   'Press_Release'])
        do_not_email = st.selectbox("Do Not Email", ['Yes', 'No'])
        total_time_spent_on_website = st.number_input(
            'Total Time Spent on Website (minutes)', step=1, format="%d")
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
        asymmetrique_profile_score = st.number_input(
            'Asymmetrique Profile Score', min_value=0, max_value=200, value=0)

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

    elif selected_model == Sales:
        Tech = st.selectbox('Primary Technology:', ('ERP Implementation',
                            'Technical Business Solutions', 'Analytics', 'Legacy Modernaization'))
        Country = st.selectbox('Country:', ('France', 'Germany',
                               'United Kingdom', 'Italy', 'Spain', 'Sweden', 'Netherlands'))
        sales_medium = st.selectbox('B2B Sales Medium:', (
            'Partners', 'Enterprise Sellers', 'Marketing', 'Online Marketing', 'Tele Sales'))
        Sales_Velocity = st.number_input(
            'Sales Velocity', min_value=0, max_value=200, value=0)
        Sales_stage = st.number_input(
            'Sales Stage Iteration', min_value=0, max_value=30, value=0)
        Opportunity_Usd = st.number_input(
            'The exact $ value', min_value=0, max_value=10000000, value=0)
        Revenue_size = st.selectbox('Company revenue:', ('100k or less',
                                    '100k to 250k', '250k to 500k', '500k to 1m', 'More than 1M'))
        Company_size = st.selectbox(
            'Company Size:', ('1k or less', '1k to 5k', '5K to 15k', '15k to 25k', 'More than 25k'))
        Previous_Business = st.selectbox('Previous Business:', (
            '0 (No business)', '0 - 25,000', '25,000 - 50,000', '50,000 to 100,000', 'More than 100,000'))
        Opportunity_sizing = st.selectbox('Deal_Valuation:', ('10k or less', '10k to 20k',
                                          '20k to 30k', '30k to 40k', '40k to 50k', '50k to 60k', 'More than 60k'))

        user_input_data = pd.DataFrame({
            'Technology': [Tech],
            'Country': [Country],
            'B2B Sales Medium': [sales_medium],
            'Sales Velocity': Sales_Velocity,
            'Sales Stage Iterations': [Sales_stage],
            'Opportunity Size (USD)': [Opportunity_Usd],
            'Client Revenue Sizing': [Revenue_size],
            'Client Employee Sizing': [Company_size],
            'Business from Client Last Year': [Previous_Business],
            'Opportunity Sizing': [Opportunity_sizing]
        })

    elif selected_model == Customer_Success:

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
    if selected_model == Marketing:
        prediction = None  # Initialize prediction
        if st.button('Predict'):
            # Directly pass the user input datas to the model for prediction
            prediction = Marketing.predict(user_input_data)[0]

            if prediction == 1:
                feedback = "The lead will be Converted"
                st.success(f"✅ {feedback}")
            else:
                feedback = "The lead will be Not Converted"
                st.error(f"⚠️ {feedback}")

    elif selected_model == Sales:
        prediction = None  # Initialize prediction
        if st.button('Predict'):
            # Directly pass the user input datas to the model for prediction
            prediction = Sales.predict(pd.DataFrame(user_input_data))[0]

            if prediction == 1:
                feedback = "The deal is expected to be successful"
                st.success(f"✅ {feedback}")
            else:
                feedback = "The deal is at risk of not closing successfully"
                st.error(f"⚠️ {feedback}")

    elif selected_model == Customer_Success:
        prediction = None  # Initialize prediction
        if st.button('Predict'):
            # Directly pass the user input datas to the model for prediction
            prediction = Customer_Success.predict(
                pd.DataFrame(user_input_data))[0]

            if prediction == 1:
                feedback = "The Customer will terminate the contract"
                st.error(f"⚠️ {feedback}")
            else:
                feedback = "The Customer will not terminate the contract"
                st.success(f"✅ {feedback}")


# Function to handle upload data for both "Marketing" and "Sales"

def batch_prediction_interface():
    # Declare batch_data and selected_model as global to modify the global instance
    global batch_data, selected_model
    st.header("Upload Data")
    uploaded_file = st.file_uploader(
        "Upload CSV for Upload Data", type=["csv"])

    try:
        if uploaded_file is not None:
            batch_data = pd.read_csv(uploaded_file)
            predictions = selected_model.predict(batch_data)
            batch_data['predictions'] = predictions

            # Save the predictions to session state
            st.session_state['batch_data'] = batch_data
            st.session_state['predictions'] = predictions

            # Adjust the ID column to start at 1 for batch_data
            if selected_model == Customer_Success:
                batch_data.index = np.arange(1, len(batch_data) + 1)
                # Create a separate DataFrame for predictions
                prediction_data = pd.DataFrame({
                    'ID': range(1, len(predictions) + 1),
                    'Prediction': [
                        'The Customer will terminate the contract' if pred == 1 else 'The Customer will not terminate the contract'
                        for pred in predictions]
                })
                st.write("Batch Data:")
                st.dataframe(batch_data, use_container_width=True)
                st.write("Predictions:")
                st.dataframe(prediction_data.set_index(
                    'ID'), use_container_width=True)

            elif selected_model == Sales:
                batch_data.index = np.arange(1, len(batch_data) + 1)
                # Create a separate DataFrame for predictions
                prediction_data = pd.DataFrame({
                    'ID': range(1, len(predictions) + 1),
                    'Prediction': [
                        'The deal is on track to be won' if pred == 1 else 'The deal is on track to be lost'
                        for pred in predictions]
                })
                st.write("Batch Data:")
                st.dataframe(batch_data, use_container_width=True)
                st.write("Predictions:")
                st.dataframe(prediction_data.set_index(
                    'ID'), use_container_width=True)

            elif selected_model == Marketing:
                batch_data.index = np.arange(1, len(batch_data) + 1)
                # Create a separate DataFrame for predictions
                prediction_data = pd.DataFrame({
                    'ID': range(1, len(predictions) + 1),
                    'Prediction': [
                        'The lead will be converted' if pred == 1 else 'The lead will not be converted'
                        for pred in predictions]
                })

                # Display the original data and prediction data separately without the index column
                st.write("Batch Data:")
                st.dataframe(batch_data, use_container_width=True)
                st.write("Predictions:")
                st.dataframe(prediction_data.set_index(
                    'ID'), use_container_width=True)

        # Show visualizations if checkbox is checked and batch_data is available
        if st.checkbox('Show visualizations'):
            if st.session_state['batch_data'] is not None and not st.session_state['batch_data'].empty:
                show_visualizations()
        else:
            st.warning(
                "No batch data available for visualization. Please upload data and predict.")

    except Exception as e:
        st.error(f"An error occurred: Check file uploaded")


def show_visualizations():
    global batch_data
    if selected_model == Sales:
        if 'batch_data' in st.session_state and not st.session_state['batch_data'].empty:
            batch_data = st.session_state['batch_data']
            # Move this line inside the if block
            st.subheader("Data Visualizations")

            col1, col5 = st.columns(2)

            # Histogram for Points in Wallet
            with col1:
                try:
                    fig_hist = px.histogram(
                        batch_data,
                        x='Technology',
                        color='Technology',
                        nbins=20,
                        title='Distribution of Tech in Data'
                    )
                    fig_hist.update_layout(
                        xaxis_title='Primary Technology',
                        yaxis_title='Count'
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                except Exception as e:
                    st.error(
                        f"An error occurred while creating the histogram: check file uploaded")

            # Bar Plot for Sales Stages
            with col5:
                try:
                    fig_bar = px.bar(
                        batch_data,
                        y='Sales Stage Iterations',
                        color='Sales Stage Iterations',
                        title='Stages of Sales'
                    )
                    fig_bar.update_layout(
                        yaxis_title='Sales Stage'
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                except Exception as e:
                    st.error(
                        f"An error occurred while creating the histogram: check file uploaded")

                # Bar chart for Feedback Distribution
            col3, col2 = st.columns(2)
            with col3:

                try:
                    if 'B2B Sales Medium' in batch_data.columns:
                        fig_count = px.bar(
                            batch_data,
                            x='B2B Sales Medium',
                            color='B2B Sales Medium',
                            title='Sales Medium'
                        )
                    fig_count.update_layout(
                        xaxis_title='Feedback',
                        yaxis_title='Count',
                        xaxis={'categoryorder': 'total descending'}
                    )

                    st.plotly_chart(fig_count, use_container_width=True)
                except Exception as e:
                    st.error(
                        f"An error occurred while creating the bar chart: check file uploaded")
            with col2:

                try:
                    if 'Country' in batch_data.columns:
                        fig_count = px.bar(
                            batch_data,
                            x='Country',
                            color='Country',
                            title='Distribution By Countries'
                        )
                    fig_count.update_layout(
                        xaxis_title='Feedback',
                        yaxis_title='Count',
                        xaxis={'categoryorder': 'total descending'}
                    )

                    st.plotly_chart(fig_count, use_container_width=True)
                except Exception as e:
                    st.error(
                        f"An error occurred while creating the bar chart: check file uploaded")

            # Filter button for batch_data index at the bottom
            selected_rows = st.multiselect(
                'Select Rows (Index)', batch_data.index.tolist(), default=batch_data.index.tolist())

            # Apply row filter
            filtered_batch_data = batch_data.loc[selected_rows]

            for selected_row in selected_rows:
                selected_data = batch_data.loc[[selected_row]]

                # Display x and y values for the selected row
                st.write(f"Technology : {selected_data['Technology'].iloc[0]}")
                st.write(
                    f"Sales Stage Iterations: {selected_data['Sales Stage Iterations'].iloc[0]}")
                st.write(f"Prediction: {selected_data['predictions'].iloc[0]}")

    if selected_model == Marketing:
        if 'batch_data' in st.session_state and not st.session_state['batch_data'].empty:
            batch_data = st.session_state['batch_data']
            st.subheader("Model Insights and Data Visualizations")
            col1, col7 = st.columns(2)
            with col1:
                try:
                    fig_hist = px.histogram(
                        batch_data,
                        x='lead_origin',
                        color='lead_origin',
                        nbins=20,
                        title='Distribution of Lead origin'
                    )
                    fig_hist.update_layout(
                        xaxis_title='Lead origin',
                        yaxis_title='Frequency'
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                except Exception as e:
                    st.error(
                        f"An error occurred while creating the histogram: check file uploaded")

            with col7:
                try:
                    # Box plot for Lead Profiles
                    fig_hist = px.histogram(
                        batch_data,
                        x='lead_profile',
                        color='lead_profile',
                        title='Distribution of Lead Profile'
                    )
                    fig_hist.update_layout(
                        yaxis_title='Lead Profile'
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                except Exception as e:
                    st.error(
                        f"An error occurred while creating the histogram: check file uploaded")

                try:
                    if 'lead_origin' in batch_data.columns:
                        fig_count = px.bar(
                            batch_data,
                            y='lead_quality',
                            color='lead_quality',
                            title='Distribution of Lead Origin'
                        )
                        fig_count.update_layout(
                            xaxis_title='Lead origin',
                            yaxis_title='Count',
                            xaxis={'categoryorder': 'total descending'}
                        )

                        st.plotly_chart(fig_count, use_container_width=True)
                except Exception as e:
                    st.error(
                        f"An error occurred while creating the bar chart: check file uploaded")

    if selected_model == Customer_Success:
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
                st.error(
                    f"An error occurred while creating the box plot: check file uploaded")

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
            st.error(
                f"An error occurred while creating the bar chart: check file uploaded")

        # Bar chart for Technology Distribution
        try:
            if 'technology_primary' in batch_data.columns:
                fig_count = px.bar(
                    batch_data,
                    x='technology_primary',
                    title='Distribution of Technology Types'
                )
                fig_count.update_layout(
                    xaxis_title='technology_primary',
                    yaxis_title='Count',
                    xaxis={'categoryorder': 'total descending'}
                )
                fig_count.update_traces(marker_color='blue')
                st.plotly_chart(fig_count, use_container_width=True)
        except Exception as e:
            st.error(
                f"An error occurred while creating the bar chart: check file uploaded")


# checking the autentication
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

# Show login form if not authenticated
if not st.session_state['authenticated']:
    show_login_form()

else:

    # Example local image file path
    image_path = 'images/image.png'
    # Open the image using PIL
    image = Image.open(image_path)
    # Resize the image
    resized_image = image.resize((image.width, 150))
    # Display the resized image
    st.image(resized_image, use_column_width=True)

    # Main() function
    st.markdown("""
        :dart: The application is functional for both manual prediction and uploading data prediction. \n
        """)
    st.markdown("<h3></h3>", unsafe_allow_html=True)

    # Initialize session state variables if they don't exist
    if 'selected_department' not in st.session_state:
        st.session_state['selected_department'] = None
    if 'last_selected_model' not in st.session_state:
        st.session_state['last_selected_model'] = None
    if 'batch_data' not in st.session_state:
        st.session_state['batch_data'] = None

    # Sidebar for selecting the department
    model = st.sidebar.radio(
        "Departments", ["Marketing", "Sales", "Customer Success"])

    # Check if the department has changed
    if st.session_state['selected_department'] != model:
        # Reset the uploaded dataset and other related session state variables
        st.session_state['batch_data'] = None
        st.session_state['predictions'] = None
        # Update the current department in session state
        st.session_state['selected_department'] = model

    # Update the selected model based on the user's choice
    if model == "Marketing":
        selected_model = Marketing
        st.title('Lead Prediction')
    elif model == "Sales":
        selected_model = Sales
        st.title('Sales Pipeline Prediction')
    elif model == "Customer Success":
        selected_model = Customer_Success
        st.title('Churn Prediction')

    # Check if the selected model has changed
    if st.session_state['last_selected_model'] != selected_model:
        # Reset session state variables related to batch data
        st.session_state['batch_data'] = None
        st.session_state['last_selected_model'] = selected_model

    prediction_method = st.sidebar.radio("Select Prediction Method", [
        "Online", "Upload Data"])

    if prediction_method == "Online":
        online_input()
    elif prediction_method == "Upload Data":
        batch_prediction_interface()

    # Sidebar logout button
    with st.sidebar:

        # Creating space using markdown (line breaks)
        st.markdown('<br>' * 3, unsafe_allow_html=True)

        # Logout button at the bottom
        if st.button('Logout'):
            # Reset user session states
            st.session_state['username'] = None
            st.session_state['password'] = None
            st.session_state['authenticated'] = False

            # Rerun the app to update the state and return to the login screen
            st.experimental_rerun()

# Sidebar Revenue Magent logo
with st.sidebar:
    # Creating space using markdown (line breaks)
    st.markdown('<br>' * 2, unsafe_allow_html=True)
    # Load Sidebar image
    image = Image.open('images/logo.png')
    # Display the image in the sidebar as the first element
    st.sidebar.image(image, use_column_width=True)
