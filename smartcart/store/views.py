from django.shortcuts import render, get_object_or_404, redirect
from .models import Product, ReviewRating, ProductGallery
from category.models import Category
from carts.models import CartItem
from django.db.models import Q

from carts.views import _cart_id
from django.core.paginator import EmptyPage, PageNotAnInteger, Paginator
from .forms import ReviewForm
from django.contrib import messages
from orders.models import OrderProduct
from store.utils.recommendation import recommend_products
from store.utils.inventory_forecasting import inventory_forecaster
import json


def store(request, category_slug=None):
    categories = None
    products = None

    if category_slug != None:
        categories = get_object_or_404(Category, slug=category_slug)
        products = Product.objects.filter(category=categories, is_available=True)
        paginator = Paginator(products, 1)
        page = request.GET.get('page')
        paged_products = paginator.get_page(page)
        product_count = products.count()
    else:
        products = Product.objects.all().filter(is_available=True).order_by('id')
        paginator = Paginator(products, 3)
        page = request.GET.get('page')
        paged_products = paginator.get_page(page)
        product_count = products.count()

    context = {
        'products': paged_products,
        'product_count': product_count,
    }
    return render(request, 'store/store.html', context)



def product_detail(request, category_slug, product_slug):
    try:
        single_product = Product.objects.get(category__slug=category_slug, slug=product_slug)
        in_cart = CartItem.objects.filter(cart__cart_id=_cart_id(request), product=single_product).exists()
    except Exception as e:
        raise e

    if request.user.is_authenticated:
        try:
            orderproduct = OrderProduct.objects.filter(user=request.user, product_id=single_product.id).exists()
        except OrderProduct.DoesNotExist:
            orderproduct = None
    else:
        orderproduct = None

    # Get the reviews
    reviews = ReviewRating.objects.filter(product_id=single_product.id, status=True)

    # Get the product gallery
    product_gallery = ProductGallery.objects.filter(product_id=single_product.id)

    # Get recommendations using Kaggle dataset
    recommendations = recommend_products(single_product.product_name)
    print("######################## recommended", recommendations)

    context = {
        'single_product': single_product,
        'in_cart': in_cart,
        'orderproduct': orderproduct,
        'reviews': reviews,
        'product_gallery': product_gallery,
        'recommendations': recommendations,
    }
    return render(request, 'store/product_detail.html', context)


def search(request):
    if 'keyword' in request.GET:
        keyword = request.GET['keyword']
        if keyword:
            products = Product.objects.order_by('-created_date').filter(Q(description__icontains=keyword) | Q(product_name__icontains=keyword))
            product_count = products.count()
    context = {
        'products': products,
        'product_count': product_count,
    }
    return render(request, 'store/store.html', context)


def submit_review(request, product_id):
    url = request.META.get('HTTP_REFERER')
    if request.method == 'POST':
        try:
            reviews = ReviewRating.objects.get(user__id=request.user.id, product__id=product_id)
            print("review function working")
            form = ReviewForm(request.POST, instance=reviews)
            form.save()
            messages.success(request, 'Thank you! Your review has been updated.')
            return redirect(url)
        except ReviewRating.DoesNotExist:
            print("review function is not working")
            form = ReviewForm(request.POST)
            if form.is_valid():
                data = ReviewRating()
                data.subject = form.cleaned_data['subject']
                data.rating = form.cleaned_data['rating']
                data.review = form.cleaned_data['review']
                data.ip = request.META.get('REMOTE_ADDR')
                data.product_id = product_id
                data.user_id = request.user.id
                data.save()
                messages.success(request, 'Thank you! Your review has been submitted.')
                return redirect(url)

def inventory_dashboard(request):
    """Inventory forecasting dashboard"""
    try:
        # Load data if not already loaded
        if inventory_forecaster.data is None:
            inventory_forecaster.load_data()
        
        # Get top performing categories
        top_categories = inventory_forecaster.get_top_performing_categories(top_n=5)
        
        # Get category performance
        category_performance = inventory_forecaster.analyze_category_performance()
        
        # Get low stock alerts (example current inventory)
        current_inventory = {
            'Shoes': 50,
            'Dresses': 30,
            'Jeans': 25,
            'T-Shirts': 40,
            'Jackets': 15,
            'Accessories': 60
        }
        low_stock_alerts = inventory_forecaster.get_low_stock_alerts(current_inventory)
        
        # Get top category name
        top_category_name = top_categories.index[0] if top_categories is not None and len(top_categories) > 0 else 'N/A'
        
        # Convert DataFrame to list of dictionaries for template iteration
        top_categories_list = []
        if top_categories is not None:
            for category, data in top_categories.iterrows():
                top_categories_list.append({
                    'category': category,
                    'total_revenue': data['Total_Revenue'],
                    'total_quantity': data['Total_Quantity'],
                    'avg_price': data['Avg_Price'],
                    'demand_variability': data['Demand_Variability']
                })
        
        context = {
            'top_categories': top_categories_list,
            'category_performance': category_performance,
            'low_stock_alerts': low_stock_alerts,
            'current_inventory': current_inventory,
            'total_categories': len(category_performance) if category_performance is not None else 0,
            'total_alerts': len(low_stock_alerts) if low_stock_alerts else 0,
            'top_category_name': top_category_name
        }
        
        return render(request, 'store/inventory_dashboard.html', context)
        
    except Exception as e:
        messages.error(request, f"Error loading inventory data: {e}")
        return render(request, 'store/inventory_dashboard.html', {})

def category_forecast(request, category_name):
    """Detailed forecast for a specific category"""
    try:
        if inventory_forecaster.data is None:
            inventory_forecaster.load_data()
        
        # Get forecast for the category
        forecast_result = inventory_forecaster.forecast_demand(category_name)
        
        # Get inventory recommendations
        inventory_rec = inventory_forecaster.calculate_inventory_recommendations(category_name)
        
        if forecast_result and inventory_rec:
            # Prepare data for charts
            historical_data = forecast_result['historical'].reset_index()
            forecast_data = forecast_result['forecast']
            
            # Convert to JSON for JavaScript charts
            historical_json = historical_data.to_json(orient='records', date_format='iso')
            forecast_json = forecast_data.to_json(orient='records', date_format='iso')
            
            context = {
                'category_name': category_name,
                'forecast_result': forecast_result,
                'inventory_rec': inventory_rec,
                'historical_data': historical_json,
                'forecast_data': forecast_json,
                'model_performance': forecast_result['model_performance']
            }
        else:
            context = {
                'category_name': category_name,
                'error': 'Insufficient data for forecasting'
            }
        
        return render(request, 'store/category_forecast.html', context)
        
    except Exception as e:
        messages.error(request, f"Error generating forecast: {e}")
        return render(request, 'store/category_forecast.html', {'category_name': category_name})

def inventory_recommendations(request):
    """Complete inventory recommendations for all categories"""
    try:
        if inventory_forecaster.data is None:
            inventory_forecaster.load_data()
        
        # Generate complete report
        report = inventory_forecaster.generate_forecast_report()
        
        if report:
            # Convert category performance DataFrame to list of dictionaries
            category_performance_list = []
            if report['category_performance'] is not None:
                for category, data in report['category_performance'].iterrows():
                    category_performance_list.append({
                        'category': category,
                        'total_revenue': data['Total_Revenue'],
                        'total_quantity': data['Total_Quantity'],
                        'avg_quantity': data['Avg_Quantity'],
                        'avg_price': data['Avg_Price'],
                        'sales_count': data['Sales_Count'],
                        'demand_variability': data['Demand_Variability']
                    })
            
            context = {
                'category_performance': category_performance_list,
                'inventory_recommendations': report['inventory_recommendations'],
                'forecasts': report['forecasts']
            }
        else:
            context = {'error': 'Unable to generate recommendations'}
        
        return render(request, 'store/inventory_recommendations.html', context)
        
    except Exception as e:
        messages.error(request, f"Error generating recommendations: {e}")
        return render(request, 'store/inventory_recommendations.html', {})

def demand_analysis(request):
    """Demand analysis and trends"""
    try:
        if inventory_forecaster.data is None:
            inventory_forecaster.load_data()
        
        # Get category performance
        category_performance = inventory_forecaster.analyze_category_performance()
        
        # Get demand trends for each category
        demand_trends = {}
        categories = inventory_forecaster.data['ItemCategory'].unique()
        
        for category in categories:
            forecast_result = inventory_forecaster.forecast_demand(category)
            if forecast_result:
                demand_trends[category] = {
                    'trend_slope': forecast_result['model_performance']['trend_slope'],
                    'avg_demand': forecast_result['model_performance']['avg_demand'],
                    'last_actual': forecast_result['model_performance']['last_actual']
                }
        
        # Convert category performance DataFrame to list of dictionaries
        category_performance_list = []
        if category_performance is not None:
            for category, data in category_performance.iterrows():
                category_performance_list.append({
                    'category': category,
                    'total_revenue': data['Total_Revenue'],
                    'avg_quantity': data['Avg_Quantity'],
                    'demand_variability': data['Demand_Variability']
                })
        
        context = {
            'category_performance': category_performance_list,
            'demand_trends': demand_trends,
            'categories': list(categories)
        }
        
        return render(request, 'store/demand_analysis.html', context)
        
    except Exception as e:
        messages.error(request, f"Error analyzing demand: {e}")
        return render(request, 'store/demand_analysis.html', {})