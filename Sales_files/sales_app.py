#Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image

#import the saved
import joblib
model = joblib.load(r"./best_model_xgb.sav")

def Online_input():
    
    #Based on our optimal features selection
    st.subheader("Client data")
    
    
    Tech = st.selectbox('Primary Technology:', ('ERP Implementation', 'Technical Business Solutions', 'Analytics', 'Legacy Modernaization'))
    Country = st.selectbox('Country:', ('France','Germany', 'United Kingdom', 'Italy', 'Spain', 'Sweden','Netherlands'))
    sales_medium = st.selectbox('B2B Sales Medium:', ('Partners','Enterprise Sellers','Marketing','Online Marketing','Tele Sales'))
    Sales_Velocity = st.number_input('Sales Velocity', min_value=0, max_value=200, value=0)
    Sales_stage = st.number_input('Sales Stage Iteration',min_value=0, max_value=30, value=0)
    Opportunity_Usd = st.number_input('The exact $ value', min_value=0, max_value=10000000, value=0)
    Revenue_size = st.selectbox('Company revenue:', ('100k or less', '100k to 250k', '250k to 500k','500k to 1m','More than 1M'))
    Company_size = st.selectbox('Company Size:', ('1k or less', '1k to 5k','5K to 15k', '15k to 25k', 'More than 25k'))
    Previous_Business = st.selectbox('Previous Business:',('0 (No business)','0 - 25,000', '25,000 - 50,000', '50,000 to 100,000', 'More than 100,000'))


    

        
    user_data = pd.DataFrame({
                'Technology': [Tech],
                'Country': [Country],
                'B2B Sales Medium': [sales_medium],
                'Sales Velocity': Sales_Velocity,
                'Sales Stage Iterations': [Sales_stage],
                'Opportunity Size (USD)': [Opportunity_Usd],
                'Client Revenue Sizing': [Revenue_size],
                'Client Employee Sizing':[Company_size],
                'Business from Client Last Year': [Previous_Business],
                
                
                })
        

        # Process user input and predict
    if st.button('Predict'):
        # Directly pass the user input datas to the model for prediction
        prediction = model.predict(pd.DataFrame(user_data))[0]
        feedback = "Won" if prediction == 1 else "Loss"
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
            predictions = model.predict(batch_data)
            batch_data['predictions'] = predictions
# Save the predictions to session state
            st.session_state['batch_data'] = batch_data
            st.session_state['predictions'] = predictions

            st.write(batch_data)
        except Exception as e:
            st.error(f"An error occurred: check file uploaded")


def show_visualizations():
    if 'batch_data' in st.session_state and not st.session_state['batch_data'].empty:
        batch_data = st.session_state['batch_data']
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
            try:
                if 'B2B Sales Medium' in batch_data.columns:
                    fig_count = px.bar(
                        batch_data,
                        x='B2B Sales Medium',
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

        # Filter button for batch_data index at the bottom
        selected_rows = st.multiselect(
            'Select Rows (Index)', batch_data.index.tolist(), default=batch_data.index.tolist())

        # Apply row filter
        filtered_batch_data = batch_data.loc[selected_rows]

        for selected_row in selected_rows:
            selected_data = batch_data.loc[[selected_row]]

            # Display x and y values for the selected row
            st.write(f"Technology : {selected_data['Technology'].iloc[0]}")
            st.write(f"Sales Stage Iterations: {selected_data['Sales Stage Iterations'].iloc[0]}")
            st.write(f"Prediction: {selected_data['predictions'].iloc[0]}")

    else:
        st.error(
            "No data available to show visualizations. Please upload data and predict.")

            
#Setting Application title
st.title('Sales Prediction App')

#Setting Application description
st.markdown("""
     :dart:  This Streamlit app aims to predict if a case(client) will be won or lost .
    The application is functional for both manual prediction and uploading data prediction. \n
    """)
st.markdown("<h3></h3>", unsafe_allow_html=True)


prediction_method = st.selectbox("Select Prediction Method", [
                                         "Input features", "Upload Data"])

if prediction_method == "Input features":
    Online_input()
elif prediction_method == "Upload Data":
    batch_prediction_interface()

# Show visualizations if checkbox is checked and batch_data is available
if st.checkbox('Show visualizations'):
    if 'batch_data' in st.session_state and not st.session_state['batch_data'].empty:
        show_visualizations()
    else:
        st.warning(
            "No batch data available for visualization. Please upload data and predict.")