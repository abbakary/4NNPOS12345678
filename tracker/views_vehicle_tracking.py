"""
Vehicle Tracking and Service Analytics Dashboard
Provides detailed tracking of vehicles by service period (daily, weekly, monthly)
with analytics, charts, and detailed invoice/order information.
"""

import logging
import json
from datetime import datetime, timedelta
from decimal import Decimal
from django.db.models import Count, Sum, Q, F, DecimalField
from django.db.models.functions import Cast, TruncDate, TruncWeek, TruncMonth
from django.http import JsonResponse
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods
from django.utils import timezone

from tracker.models import Vehicle, Order, Invoice, InvoiceLineItem, LabourCode, Customer
from tracker.utils.order_type_detector import _normalize_category_to_order_type
from tracker.utils.auth import get_user_branch

logger = logging.getLogger(__name__)


@login_required
def vehicle_tracking_dashboard(request):
    """
    Vehicle Tracking Dashboard - Shows vehicles that came for service
    with daily, weekly, and monthly analytics.
    """
    user_branch = get_user_branch(request.user)
    
    # Get filter parameters
    period = request.GET.get('period', 'monthly')  # daily, weekly, monthly
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    status_filter = request.GET.get('status', '')  # completed, pending, all
    order_type_filter = request.GET.get('order_type', '')  # service, sales, labour
    
    # Set default date range
    if not end_date:
        end_date = timezone.now().date()
    else:
        try:
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        except:
            end_date = timezone.now().date()
    
    if not start_date:
        if period == 'daily':
            start_date = end_date
        elif period == 'weekly':
            start_date = end_date - timedelta(days=7)
        else:  # monthly
            start_date = end_date - timedelta(days=30)
    else:
        try:
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        except:
            start_date = end_date - timedelta(days=30)
    
    context = {
        'period': period,
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'status_filter': status_filter,
        'order_type_filter': order_type_filter,
    }
    
    return render(request, 'tracker/vehicle_tracking_dashboard.html', context)


@login_required
@require_http_methods(["GET"])
def api_vehicle_tracking_data(request):
    """
    API endpoint for vehicle tracking data with filtering and aggregation.
    
    Query parameters:
    - period: daily|weekly|monthly
    - start_date: YYYY-MM-DD
    - end_date: YYYY-MM-DD
    - status: completed|pending|in_progress|all
    - order_type: service|sales|labour|mixed|all
    - search: search by plate number or customer name
    """
    user_branch = get_user_branch(request.user)
    
    try:
        period = request.GET.get('period', 'monthly')
        start_date_str = request.GET.get('start_date')
        end_date_str = request.GET.get('end_date')
        status_filter = request.GET.get('status', 'all')
        order_type_filter = request.GET.get('order_type', 'all')
        search_query = request.GET.get('search', '').strip()
        
        # Parse dates
        try:
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date() if end_date_str else timezone.now().date()
        except:
            end_date = timezone.now().date()
        
        try:
            if start_date_str:
                start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
            else:
                start_date = end_date - timedelta(days=30)
        except:
            start_date = end_date - timedelta(days=30)
        
        # Query vehicles that came for service:
        # 1. Vehicles with invoices (uploaded invoices = service reference)
        # 2. Vehicles with service-type orders
        # 3. Vehicles in both categories
        vehicles_query = Vehicle.objects.filter(
            Q(invoices__invoice_date__range=[start_date, end_date]) |
            Q(orders__created_at__date__range=[start_date, end_date], orders__type='service')
        ).distinct()
        
        if user_branch:
            vehicles_query = vehicles_query.filter(
                Q(invoices__branch=user_branch) | Q(orders__branch=user_branch)
            ).distinct()
        
        # Apply search filter
        if search_query:
            vehicles_query = vehicles_query.filter(
                Q(plate_number__icontains=search_query) |
                Q(customer__full_name__icontains=search_query)
            )
        
        vehicle_data = []
        
        for vehicle in vehicles_query:
            # Get all invoices for this vehicle in the date range
            invoices = vehicle.invoices.filter(
                invoice_date__range=[start_date, end_date]
            )
            
            if user_branch:
                invoices = invoices.filter(branch=user_branch)
            
            # Get all orders for this vehicle in the date range
            orders = vehicle.orders.filter(
                created_at__date__range=[start_date, end_date]
            )

            if user_branch:
                orders = orders.filter(branch=user_branch)

            # Also check for orders linked through invoices
            order_links_via_invoices = Order.objects.filter(
                invoices__vehicle=vehicle,
                invoices__invoice_date__range=[start_date, end_date]
            ).distinct()

            if user_branch:
                order_links_via_invoices = order_links_via_invoices.filter(branch=user_branch)

            # Combine orders from both sources
            all_orders = orders.union(order_links_via_invoices).order_by('-created_at')

            if not invoices.exists() and not all_orders.exists():
                continue

            # Calculate vehicle metrics
            total_spent = invoices.aggregate(total=Sum('total_amount'))['total'] or Decimal('0')
            invoice_count = invoices.count()

            # Get order statistics from all orders
            order_stats = {
                'completed': all_orders.filter(status='completed').count(),
                'in_progress': all_orders.filter(status='in_progress').count(),
                'pending': all_orders.filter(status='created').count(),
                'overdue': all_orders.filter(status='overdue').count(),
                'cancelled': all_orders.filter(status='cancelled').count(),
            }

            # Get order types and service types
            order_types = set()
            service_types = set()

            for order in all_orders:
                order_types.add(order.type)

                # Extract service types from service orders
                if order.type == 'service':
                    # Try to get service type from order description or mixed_categories
                    if order.mixed_categories:
                        try:
                            categories = json.loads(order.mixed_categories)
                            for cat in categories:
                                service_types.add(cat)
                        except:
                            pass
            
            # Get invoice data with line items
            invoice_list = []
            for invoice in invoices:
                line_items = InvoiceLineItem.objects.filter(invoice=invoice)

                # Get categories for line items
                categories = set()
                line_items_data = []

                for item in line_items:
                    # Try to find labor code for this item
                    category = 'Service'
                    labour_code = None

                    if item.code:
                        labour_code = LabourCode.objects.filter(code__iexact=item.code).first()
                        if labour_code:
                            category = labour_code.category
                            categories.add(category)

                    line_items_data.append({
                        'code': item.code or '',
                        'description': item.description,
                        'qty': float(item.quantity),
                        'unit_price': float(item.unit_price),
                        'total': float(item.line_total),
                        'category': category,
                        'tax_rate': float(item.tax_rate) if item.tax_rate else 0,
                        'tax_amount': float(item.tax_amount) if item.tax_amount else 0,
                    })

                invoice_dict = {
                    'invoice_number': invoice.invoice_number,
                    'invoice_date': invoice.invoice_date.isoformat(),
                    'total_amount': float(invoice.total_amount),
                    'subtotal': float(invoice.subtotal),
                    'tax_amount': float(invoice.tax_amount),
                    'reference': invoice.reference or '',
                    'status': invoice.status,
                    'order_id': invoice.order_id,
                    'order_number': invoice.order.order_number if invoice.order else '',
                    'line_items_count': line_items.count(),
                    'categories': sorted(list(categories)) if categories else ['Service'],
                    'line_items': line_items_data
                }
                invoice_list.append(invoice_dict)
            
            # Apply status filter
            if status_filter != 'all':
                if status_filter == 'completed' and order_stats['completed'] == 0:
                    continue
                elif status_filter == 'pending' and (order_stats['pending'] + order_stats['created']) == 0:
                    continue
            
            # Apply order type filter
            if order_type_filter != 'all':
                if order_type_filter not in order_types:
                    continue
            
            # Determine if returning vehicle (multiple visits/invoices)
            is_returning = invoice_count > 1
            
            vehicle_dict = {
                'id': vehicle.id,
                'plate_number': vehicle.plate_number,
                'make': vehicle.make or '',
                'model': vehicle.model or '',
                'vehicle_type': vehicle.vehicle_type or '',
                'customer_id': vehicle.customer.id,
                'customer_name': vehicle.customer.full_name,
                'customer_phone': vehicle.customer.phone or '',
                'total_spent': float(total_spent),
                'invoice_count': invoice_count,
                'is_returning': is_returning,
                'order_stats': order_stats,
                'order_types': sorted(list(order_types)),
                'service_types': sorted(list(service_types)) if service_types else [],
                'invoices': invoice_list,
                'order_count': all_orders.count(),
            }
            
            vehicle_data.append(vehicle_dict)
        
        # Sort by total spent (descending)
        vehicle_data.sort(key=lambda x: x['total_spent'], reverse=True)
        
        # Calculate summary statistics
        summary = {
            'total_vehicles': len(vehicle_data),
            'total_spent': sum(v['total_spent'] for v in vehicle_data),
            'total_invoices': sum(v['invoice_count'] for v in vehicle_data),
            'returning_vehicles': sum(1 for v in vehicle_data if v['is_returning']),
            'order_stats': {
                'completed': sum(v['order_stats']['completed'] for v in vehicle_data),
                'in_progress': sum(v['order_stats']['in_progress'] for v in vehicle_data),
                'pending': sum(v['order_stats']['pending'] for v in vehicle_data),
                'overdue': sum(v['order_stats']['overdue'] for v in vehicle_data),
            }
        }
        
        return JsonResponse({
            'success': True,
            'data': vehicle_data,
            'summary': summary,
            'filters': {
                'period': period,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'status': status_filter,
                'order_type': order_type_filter,
            }
        })
        
    except Exception as e:
        logger.error(f"Error fetching vehicle tracking data: {e}", exc_info=True)
        return JsonResponse({
            'success': False,
            'message': str(e)
        }, status=500)


@login_required
@require_http_methods(["GET"])
def api_vehicle_analytics(request):
    """
    API endpoint for vehicle analytics and trends.
    
    Returns:
    - Daily/weekly/monthly trends
    - Spending by order type
    - Vehicle visit frequency
    - Average spending per vehicle
    """
    user_branch = get_user_branch(request.user)
    
    try:
        period = request.GET.get('period', 'monthly')
        start_date_str = request.GET.get('start_date')
        end_date_str = request.GET.get('end_date')
        
        # Parse dates
        try:
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date() if end_date_str else timezone.now().date()
        except:
            end_date = timezone.now().date()
        
        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date() if start_date_str else (end_date - timedelta(days=30))
        except:
            start_date = end_date - timedelta(days=30)
        
        # Get invoices in date range
        invoices_qs = Invoice.objects.filter(
            invoice_date__range=[start_date, end_date]
        )
        
        if user_branch:
            invoices_qs = invoices_qs.filter(branch=user_branch)
        
        # Aggregate by time period
        if period == 'daily':
            trunc_field = TruncDate('invoice_date')
        elif period == 'weekly':
            trunc_field = TruncWeek('invoice_date')
        else:
            trunc_field = TruncMonth('invoice_date')
        
        trends = invoices_qs.annotate(
            period_date=trunc_field
        ).values('period_date').annotate(
            total_amount=Sum('total_amount'),
            invoice_count=Count('id'),
            vehicle_count=Count('vehicle', distinct=True)
        ).order_by('period_date')
        
        trends_data = [
            {
                'date': t['period_date'].isoformat() if t['period_date'] else '',
                'total_amount': float(t['total_amount'] or 0),
                'invoice_count': t['invoice_count'],
                'vehicle_count': t['vehicle_count'],
            }
            for t in trends
        ]
        
        # Spending by order type
        spending_by_type = invoices_qs.filter(
            order__type__isnull=False
        ).values('order__type').annotate(
            total=Sum('total_amount'),
            count=Count('id')
        ).order_by('-total')
        
        spending_by_type_data = [
            {
                'type': item['order__type'],
                'total': float(item['total'] or 0),
                'count': item['count'],
                'average': float((item['total'] or 0) / item['count']) if item['count'] > 0 else 0,
            }
            for item in spending_by_type
        ]
        
        # Top vehicles by spending
        top_vehicles = Vehicle.objects.filter(
            invoices__invoice_date__range=[start_date, end_date]
        ).annotate(
            total_spent=Sum('invoices__total_amount'),
            invoice_count=Count('invoices', distinct=True)
        ).filter(
            total_spent__isnull=False
        ).order_by('-total_spent')[:10]
        
        if user_branch:
            top_vehicles = top_vehicles.filter(
                invoices__branch=user_branch
            )
        
        top_vehicles_data = [
            {
                'plate_number': v.plate_number,
                'customer_name': v.customer.full_name,
                'total_spent': float(v.total_spent or 0),
                'invoice_count': v.invoice_count,
                'average_per_invoice': float((v.total_spent or 0) / v.invoice_count) if v.invoice_count > 0 else 0,
            }
            for v in top_vehicles
        ]
        
        return JsonResponse({
            'success': True,
            'trends': trends_data,
            'spending_by_type': spending_by_type_data,
            'top_vehicles': top_vehicles_data,
        })
        
    except Exception as e:
        logger.error(f"Error fetching vehicle analytics: {e}", exc_info=True)
        return JsonResponse({
            'success': False,
            'message': str(e)
        }, status=500)
