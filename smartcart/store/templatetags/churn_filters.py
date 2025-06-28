from django import template

register = template.Library()

@register.filter
def multiply(value, arg):
    """Multiply the value by the argument"""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return 0

@register.filter
def percentage(value, arg):
    """Convert value to percentage based on argument"""
    try:
        return (float(value) / float(arg)) * 100
    except (ValueError, TypeError):
        return 0 