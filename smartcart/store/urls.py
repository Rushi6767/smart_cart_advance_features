from django.urls import path
from . import views


urlpatterns = [
    path('', views.store, name='store'),
    path('category/<slug:category_slug>/', views.store, name='products_by_category'),
    path('category/<slug:category_slug>/<slug:product_slug>/', views.product_detail, name='products_by_product'),
    path('search/', views.search, name='search'),
    path('submit_review/<int:product_id>/', views.submit_review, name='submit_review'),
    
    # Inventory Forecasting URLs
    path('inventory/dashboard/', views.inventory_dashboard, name='inventory_dashboard'),
    path('inventory/forecast/<str:category_name>/', views.category_forecast, name='category_forecast'),
    path('inventory/recommendations/', views.inventory_recommendations, name='inventory_recommendations'),
    path('inventory/demand-analysis/', views.demand_analysis, name='demand_analysis'),
    
    # Churn Prediction URLs
    path('churn/prediction/', views.churn_prediction, name='churn_prediction'),
    path('churn/train/', views.train_churn_model_view, name='train_churn_model'),
    path('churn/predict/', views.predict_customer_churn, name='predict_customer_churn'),
    path('churn/analysis/', views.churn_analysis, name='churn_analysis'),
    path('churn/check-model-status/', views.check_model_status, name='check_model_status'),
]
