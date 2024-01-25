# Revenue Magnet App

**Test the App here:** https://revenuemagnetapp.streamlit.app/ 

### Credentials:
- **Username**: admin
- **Password**: admin

**Slides:** https://docs.google.com/presentation/d/1ofixjH__Zj6dc9bDJnEv8s6LRP02cRy0ZefMGC7-g3E/edit#slide=id.g1058c88b81f_0_26

## Overview
Revenue Magnet App is a sophisticated Streamlit-based web application designed to predict outcomes in three crucial business areas: Marketing, Sales, and Customer Success. Leveraging advanced machine learning models, the application provides insights that can drive strategic business decisions, enhance marketing efforts, streamline sales processes, and optimize customer success initiatives.

## Features
- **User Authentication**: Secure access with a simple login interface ensuring data confidentiality.
- **Department-Specific Prediction Models**:
  - **Marketing**: Predicts the likelihood of lead conversion based on various inputs such as lead origin, lead source, website engagement, and more.
  - **Sales**: Evaluates potential sales opportunities, forecasting the chances of winning a deal by analyzing factors like technology, country, sales medium, opportunity size, etc.
  - **Customer Success**: Determines the probability of a customer terminating their contract, helping in proactive customer retention strategies. Factors considered include points in wallet, opportunity size, user feedback, and others.
- **Data Input Flexibility**:
  - **Manual Prediction**: Allows users to input data manually for real-time predictions.
  - **Batch Prediction**: Users can upload a CSV file for bulk predictions, enhancing efficiency for larger datasets.
- **Interactive Data Visualizations**: Provides insightful visualizations for batch data, helping in the deeper analysis and understanding of the trends and patterns.

## Business Problem and Goals
The primary objective of the Revenue Magnet App is to empower businesses to make data-driven decisions across various departments:

- **In Marketing**: Enhancing lead conversion rates by predicting the likelihood of leads converting into customers.
- **In Sales**: Increasing sales efficiency and forecasting by identifying the deals most likely to succeed.
- **In Customer Success**: Proactively identifying at-risk customers, enabling timely interventions to improve customer retention.

## Technology Stack
- **Streamlit**: For building and hosting the web application.
- **Pandas**: For data manipulation and analysis.
- **Joblib**: For loading pre-trained machine learning models.
- **Altair & Plotly Express**: For creating interactive data visualizations.
- **NumPy**: For numerical operations.

## Security
The app includes a basic authentication mechanism to restrict access and protect sensitive data.

## Future Enhancements
- Implementing more robust user authentication and authorization.
- Expanding the analytics dashboard for more in-depth insights.
- Integration with live data sources for real-time data analysis.

## Getting Started

To run the application:
1. Activate the virtual environment.
2. Navigate to the application directory.
3. Run `streamlit run app.py`.

## Set up your Environment


### **`macOS`** type the following commands : 

- For installing the virtual environment you can either use the [Makefile](Makefile) and run `make setup` or install it manually with the following commands:

     ```BASH
    make setup
    ```
    After that active your environment by following commands:
    ```BASH
    source .venv/bin/activate
    ```
Or ....
- Install the virtual environment and the required packages by following commands:

    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
    
### **`WindowsOS`** type the following commands :

- Install the virtual environment and the required packages by following commands.

   For `PowerShell` CLI :

    ```PowerShell
    pyenv local 3.11.3
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    For `Git-bash` CLI :
    ```
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/Scripts/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    **`Note:`**
    If you encounter an error when trying to run `pip install --upgrade pip`, try using the following command:
    ```Bash
    python.exe -m pip install --upgrade pip
    ```


   
## Usage

In order to train the model and store test data in the data folder and the model in models run:

**`Note`**: Make sure your environment is activated.

```bash
python example_files/train.py  
```

In order to test that predict works on a test set you created run:

```bash
python example_files/predict.py models/linear_regression_model.sav data/X_test.csv data/y_test.csv
```

### **`Linux Users`**  : You know what to do :sunglasses:
