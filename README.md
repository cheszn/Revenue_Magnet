# Revenue_Magnet

## Project Overview:

We are embarking on a project at 'Fiscal Fortune Finders,' a consulting firm specialising in the tech SaaS industry. Our primary objective is to enhance the efficiency of four core services:

1. Analytics
2. ERP Implementation
3. Legacy Modernization
4. Technical Business Solutions


Our project focuses on optimising the processes within these services and involves three critical teams:
1. Marketing: Responsible for lead generation and qualification, with a primary metric being lead conversion.
2. Sales: Tasked with converting prospects into opportunities and ultimately achieving Closed-Won status. The key metric here is opportunity status.
3. Customer Success: After successfully closing opportunities, the Customer Success team plays a vital role in retaining customers and preventing churn.
   
## Project Goals:
Our project's overarching goal is to leverage data analytics to predict outcomes in all three critical processes: Marketing, Sales, and Customer Success. By doing so, we aim to empower decision-makers, potentially including Revenue Operations Managers, to take informed actions to safeguard Fiscal Fortune Finder's revenue streams and prevent customer attrition.

In essence, our project will involve data analysis and predictive modelling to enhance lead conversion rates, optimise opportunity management, and minimise churn rates across the organisation's operations.

<--------------------------------------------------------------------------------------------------------------->

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

    ### **`Linux Users`**  : You know what to do :sunglasses:
