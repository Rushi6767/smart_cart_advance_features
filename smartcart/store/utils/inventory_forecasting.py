import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class InventoryForecasting:
    def __init__(self):
        self.data = None
        self.forecasts = {}
        self.inventory_recommendations = {}
        
    def load_data(self):
        """Load and preprocess the ABC Clothing Store data"""
        try:
            # Load the data
            self.data = pd.read_csv('inventory dataset/ABC_Clothing_Store_Sales_Data.csv')
            
            # Convert date column
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            
            # Create additional time features
            self.data['Year'] = self.data['Date'].dt.year
            self.data['Month'] = self.data['Date'].dt.month
            self.data['Day'] = self.data['Date'].dt.day
            self.data['DayOfWeek'] = self.data['Date'].dt.dayofweek
            self.data['Week'] = self.data['Date'].dt.isocalendar().week
            
            # Calculate daily demand for each category
            daily_demand = self.data.groupby(['Date', 'ItemCategory'])['Quantity'].sum().reset_index()
            daily_demand = daily_demand.pivot(index='Date', columns='ItemCategory', values='Quantity').fillna(0)
            
            print(f"Data loaded successfully! Shape: {self.data.shape}")
            print(f"Date range: {self.data['Date'].min()} to {self.data['Date'].max()}")
            print(f"Categories: {list(self.data['ItemCategory'].unique())}")
            
            return daily_demand
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def analyze_category_performance(self):
        """Analyze performance of each clothing category"""
        if self.data is None:
            return None
            
        # Category analysis
        category_stats = self.data.groupby('ItemCategory').agg({
            'Quantity': ['sum', 'mean', 'std', 'count'],
            'TotalAmount': ['sum', 'mean'],
            'Price': 'mean'
        }).round(2)
        
        category_stats.columns = ['Total_Quantity', 'Avg_Quantity', 'Std_Quantity', 'Sales_Count', 
                                'Total_Revenue', 'Avg_Revenue', 'Avg_Price']
        
        # Calculate demand variability (coefficient of variation)
        category_stats['Demand_Variability'] = (category_stats['Std_Quantity'] / category_stats['Avg_Quantity']).round(3)
        
        # Sort by total revenue
        category_stats = category_stats.sort_values('Total_Revenue', ascending=False)
        
        return category_stats
    
    def forecast_demand(self, category, forecast_days=30):
        """Forecast demand for a specific category using simple time series methods"""
        if self.data is None:
            return None
            
        # Get daily demand for the category
        category_data = self.data[self.data['ItemCategory'] == category].copy()
        daily_demand = category_data.groupby('Date')['Quantity'].sum().reset_index()
        daily_demand = daily_demand.set_index('Date').sort_index()
        
        if len(daily_demand) < 7:  # Need at least a week of data
            return None
        
        # Calculate moving averages
        daily_demand['MA_7'] = daily_demand['Quantity'].rolling(window=7).mean()
        daily_demand['MA_14'] = daily_demand['Quantity'].rolling(window=14).mean()
        
        # Simple exponential smoothing
        alpha = 0.3  # Smoothing factor
        daily_demand['SES'] = daily_demand['Quantity'].ewm(alpha=alpha).mean()
        
        # Linear trend projection
        X = np.arange(len(daily_demand)).reshape(-1, 1)
        y = daily_demand['Quantity'].values
        model = LinearRegression()
        model.fit(X, y)
        
        # Generate future dates
        last_date = daily_demand.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='D')
        
        # Predict future demand
        future_X = np.arange(len(daily_demand), len(daily_demand) + forecast_days).reshape(-1, 1)
        trend_forecast = model.predict(future_X)
        
        # Combine different forecasting methods
        avg_recent_demand = daily_demand['Quantity'].tail(7).mean()
        ses_forecast = daily_demand['SES'].iloc[-1]
        
        # Weighted average forecast
        forecast = (0.4 * trend_forecast + 0.4 * avg_recent_demand + 0.2 * ses_forecast)
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Forecasted_Demand': forecast.round(2),
            'Trend_Forecast': trend_forecast.round(2),
            'Moving_Avg_Forecast': avg_recent_demand,
            'SES_Forecast': ses_forecast
        })
        
        return {
            'historical': daily_demand,
            'forecast': forecast_df,
            'model_performance': {
                'avg_demand': avg_recent_demand,
                'trend_slope': model.coef_[0],
                'last_actual': daily_demand['Quantity'].iloc[-1]
            }
        }
    
    def calculate_inventory_recommendations(self, category, service_level=0.95, lead_time=7):
        """Calculate optimal inventory levels for a category"""
        if self.data is None:
            return None
            
        # Get demand data for the category
        category_data = self.data[self.data['ItemCategory'] == category]['Quantity']
        
        # Calculate demand statistics
        avg_demand = category_data.mean()
        std_demand = category_data.std()
        
        # Safety stock calculation (using normal distribution)
        z_score = 1.645  # For 95% service level
        safety_stock = z_score * std_demand * np.sqrt(lead_time)
        
        # Reorder point
        reorder_point = (avg_demand * lead_time) + safety_stock
        
        # Economic Order Quantity (EOQ) - simplified version
        # Assuming ordering cost = $50, holding cost = 20% of unit cost
        avg_price = self.data[self.data['ItemCategory'] == category]['Price'].mean()
        ordering_cost = 50
        holding_cost_rate = 0.20
        annual_demand = avg_demand * 365
        
        eoq = np.sqrt((2 * annual_demand * ordering_cost) / (avg_price * holding_cost_rate))
        
        # Maximum inventory level
        max_inventory = reorder_point + eoq
        
        return {
            'category': category,
            'avg_daily_demand': round(avg_demand, 2),
            'demand_std': round(std_demand, 2),
            'safety_stock': round(safety_stock, 2),
            'reorder_point': round(reorder_point, 2),
            'economic_order_quantity': round(eoq, 2),
            'max_inventory_level': round(max_inventory, 2),
            'service_level': service_level,
            'lead_time_days': lead_time
        }
    
    def generate_forecast_report(self, categories=None):
        """Generate comprehensive forecasting report for all categories"""
        if self.data is None:
            return None
            
        if categories is None:
            categories = self.data['ItemCategory'].unique()
        
        report = {
            'category_performance': self.analyze_category_performance(),
            'forecasts': {},
            'inventory_recommendations': {}
        }
        
        for category in categories:
            # Generate demand forecast
            forecast_result = self.forecast_demand(category)
            if forecast_result:
                report['forecasts'][category] = forecast_result
            
            # Generate inventory recommendations
            inventory_rec = self.calculate_inventory_recommendations(category)
            if inventory_rec:
                report['inventory_recommendations'][category] = inventory_rec
        
        return report
    
    def get_top_performing_categories(self, top_n=5):
        """Get top performing categories by revenue"""
        if self.data is None:
            return None
            
        category_performance = self.analyze_category_performance()
        return category_performance.head(top_n)
    
    def get_low_stock_alerts(self, current_inventory=None):
        """Generate low stock alerts based on forecasted demand"""
        if self.data is None:
            return None
            
        alerts = []
        
        for category in self.data['ItemCategory'].unique():
            inventory_rec = self.calculate_inventory_recommendations(category)
            if inventory_rec:
                current_stock = current_inventory.get(category, 0) if current_inventory else 0
                reorder_point = inventory_rec['reorder_point']
                
                if current_stock <= reorder_point:
                    alerts.append({
                        'category': category,
                        'current_stock': current_stock,
                        'reorder_point': reorder_point,
                        'recommended_order': inventory_rec['economic_order_quantity'],
                        'urgency': 'HIGH' if current_stock == 0 else 'MEDIUM'
                    })
        
        return alerts

# Global instance
inventory_forecaster = InventoryForecasting() 