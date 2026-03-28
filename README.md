# E-commerce Sales and Profit Forecasting

This project is designed to predict future sales and profits based on historical e-commerce data. It utilizes machine learning for time-series forecasting, specifically leveraging Facebook's Prophet model.

## Features

*   **Data Preparation & Cleaning:** Loads historical sales data (e.g., `store.csv`), handles missing values, and prepares it for time-series analysis.
*   **Data Exploration (EDA):** Analyzes sales trends, profits, and overall data distributions using `pandas`, `matplotlib`, and `seaborn`.
*   **Time-Series Forecasting:** Uses `Prophet` to train on the historical data and predict future sales and profit trends.
*   **Model Persistence:** Saves the trained forecasting models as `.pkl` files (`sales_model.pkl` and `profit_model.pkl`) for future use without having to retrain.

## Setup Instructions

1.  **Dependencies:** Ensure you have Python installed. The required libraries include:
    *   `prophet`
    *   `pandas`
    *   `matplotlib`
    *   `scikit-learn`
    *   `numpy`
    *   `seaborn`

    You can install them via pip, or run the first cell of the notebook.

2.  **Dataset:** The project requires a dataset containing historical sales data named `store.csv` in the same directory. The data should have relevant columns like `Order Date`, `Sales`, and `Profit`.

3.  **Running the Project:** Open and run the Jupyter Notebook `sales_profit_prediction.ipynb`. The notebook steps through data loading, EDA, model training, and forecasting.

## Usage

After running the notebook and training the models, they will forecast sales for the upcoming periods. The visualizations generated will help you understand the seasonality and general trend of sales and profits over time.
