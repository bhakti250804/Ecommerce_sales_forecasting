

# Create your views here.
import pandas as pd
import joblib
from django.shortcuts import render
from prophet import Prophet
import matplotlib.pyplot as plt
import os

def home(request):
    return render(request, 'mainapp/index.html')

def predict(request):
    if request.method == 'POST':

        category = request.POST['category']
        df = pd.read_csv('mainapp/data/store.csv', encoding = 'latin1')  # path to your dataset
        
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        df = df[df['Category'] == category]
        df['Month'] = df['Order Date'].dt.to_period('M').dt.to_timestamp()

        monthly = df.groupby('Month')[['Sales', 'Profit']].sum().reset_index()
        monthly = monthly.rename(columns={'Month': 'ds', 'Sales': 'y_sales', 'Profit': 'y_profit'})

        sales_model = joblib.load('mainapp/model/sales_model.pkl')
        profit_model = joblib.load('mainapp/model/profit_model.pkl')

        future = sales_model.make_future_dataframe(periods=6, freq='ME')
        forecast_sales = sales_model.predict(future)
        forecast_profit = profit_model.predict(future)
        
        forecast_combined = pd.DataFrame({
    'Month': forecast_sales['ds'],
    'Predicted_Sale': forecast_sales['yhat'],
    'Predicted_Profit': forecast_profit['yhat']
        })
        
        # Use datetime Month for comparison
        future_data = forecast_combined[forecast_combined['Month'] > monthly['ds'].max()]
        past_data = monthly.tail(6)
        
        # Convert 'Month' to string BEFORE storing in session
        forecast_combined['Month'] = forecast_combined['Month'].dt.strftime('%Y-%m-%d')
        request.session['graph_data'] = forecast_combined.to_dict('records')
        
        return render(request, 'mainapp/result.html', {
            'past_data': past_data.to_dict('records'),
            'future_data': future_data.to_dict('records'),
            'category': category
        })

    return render(request, 'mainapp/index.html')

def show_graph(request):
    graph_data = pd.DataFrame(request.session.get('graph_data'))
    plt.figure(figsize=(14,6))
    plt.plot(graph_data['Month'], graph_data['Predicted_Sale'], label='Sales')
    plt.plot(graph_data['Month'], graph_data['Predicted_Profit'], label='Profit')
    plt.xlabel("Month")
    plt.ylabel("Value")
    plt.title("Sales and Profit Forecast")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    path = "mainapp/static/graph.png"
    plt.savefig(path)
    return render(request, 'mainapp/graph.html', {"graph": 'graph.png'})
