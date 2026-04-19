# E-commerce Sales and Profit Forecasting (Web Edition)

This project predicts future sales and profits based on historical e-commerce data. It features two core components: a classic Jupyter Notebook for data exploration, and a full-stack **Django Web Application** to serve predictions and dynamic charts utilizing machine learning (Facebook's Prophet model).

## Features

*   **Django Web Application:** A sleek web interface allowing users to select product categories and instantly view forecasted sales/profits charts.
*   **Data Preparation & Cleaning:** Loads historical sales data (e.g., `store.csv`), handles missing values, and prepares it for time-series analysis.
*   **Time-Series Forecasting:** Uses `Prophet` to train on the historical data and predict future trends.
*   **Model Persistence:** Saves the trained forecasting models as `.pkl` files (Joblib) for high performance caching in the backend.

## Web App Setup Instructions

1.  **Dependencies:** Install the project dependencies using the provided `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Dataset & Models:** Ensure your dataset and pickled models are present in the internal data structure (`forecast_project/mainapp/data` and `forecast_project/mainapp/model`).

3.  **Start the Web Server:** Launch the Django backend.
    ```bash
    cd forecast_project
    python manage.py runserver
    ```

4.  **Access:** Navigate to `http://127.0.0.1:8000/` in your browser to view the application!

## Jupyter Notebook Analysis (Research)
To see the step-by-step EDA and training logic, open and run the Jupyter Notebook `sales_profit_prediction.ipynb` located in the root folder.
