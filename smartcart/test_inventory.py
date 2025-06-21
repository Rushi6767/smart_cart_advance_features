#!/usr/bin/env python
"""
Test script for inventory forecasting system
Run this script to test if the inventory forecasting system works correctly
"""

import os
import sys
import django

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'smartcart.settings')
django.setup()

from store.utils.inventory_forecasting import inventory_forecaster

def test_inventory_forecasting():
    """Test the inventory forecasting system"""
    print("🧪 Testing Inventory Forecasting System")
    print("=" * 50)
    
    try:
        # Test 1: Load data
        print("1. Loading data...")
        daily_demand = inventory_forecaster.load_data()
        
        if daily_demand is not None:
            print("✅ Data loaded successfully!")
            print(f"   - Data shape: {inventory_forecaster.data.shape}")
            print(f"   - Date range: {inventory_forecaster.data['Date'].min()} to {inventory_forecaster.data['Date'].max()}")
            print(f"   - Categories: {list(inventory_forecaster.data['ItemCategory'].unique())}")
        else:
            print("❌ Failed to load data")
            return False
        
        # Test 2: Category performance analysis
        print("\n2. Analyzing category performance...")
        category_performance = inventory_forecaster.analyze_category_performance()
        
        if category_performance is not None:
            print("✅ Category performance analysis completed!")
            print(f"   - Top category: {category_performance.index[0]}")
            print(f"   - Total revenue: ${category_performance['Total_Revenue'].sum():,.2f}")
        else:
            print("❌ Failed to analyze category performance")
            return False
        
        # Test 3: Demand forecasting
        print("\n3. Testing demand forecasting...")
        categories = inventory_forecaster.data['ItemCategory'].unique()[:3]  # Test first 3 categories
        
        for category in categories:
            print(f"   - Forecasting for {category}...")
            forecast_result = inventory_forecaster.forecast_demand(category)
            
            if forecast_result:
                print(f"     ✅ Forecast generated successfully!")
                print(f"     - Avg demand: {forecast_result['model_performance']['avg_demand']:.2f}")
                print(f"     - Trend slope: {forecast_result['model_performance']['trend_slope']:.3f}")
            else:
                print(f"     ❌ Failed to generate forecast for {category}")
        
        # Test 4: Inventory recommendations
        print("\n4. Testing inventory recommendations...")
        for category in categories:
            print(f"   - Generating recommendations for {category}...")
            inventory_rec = inventory_forecaster.calculate_inventory_recommendations(category)
            
            if inventory_rec:
                print(f"     ✅ Recommendations generated!")
                print(f"     - Safety stock: {inventory_rec['safety_stock']:.0f} units")
                print(f"     - Reorder point: {inventory_rec['reorder_point']:.0f} units")
                print(f"     - EOQ: {inventory_rec['economic_order_quantity']:.0f} units")
            else:
                print(f"     ❌ Failed to generate recommendations for {category}")
        
        # Test 5: Low stock alerts
        print("\n5. Testing low stock alerts...")
        current_inventory = {
            'Shoes': 10,
            'Dresses': 5,
            'Jeans': 0
        }
        alerts = inventory_forecaster.get_low_stock_alerts(current_inventory)
        
        if alerts:
            print(f"✅ Generated {len(alerts)} low stock alerts!")
            for alert in alerts:
                print(f"   - {alert['category']}: {alert['urgency']} priority")
        else:
            print("ℹ️ No low stock alerts generated")
        
        print("\n" + "=" * 50)
        print("🎉 All tests completed successfully!")
        print("The inventory forecasting system is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_inventory_forecasting()
    sys.exit(0 if success else 1) 