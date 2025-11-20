"""
Invoice upload and extraction endpoints.
Handles two-step process: extract preview → create/update records
"""

import json
import logging
from decimal import Decimal
from datetime import datetime
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.utils import timezone
from django.db import transaction

from .models import Order, Customer, Vehicle, Invoice, InvoiceLineItem, InvoicePayment, Branch
from .utils import get_user_branch
from .services import OrderService, CustomerService, VehicleService

logger = logging.getLogger(__name__)


@login_required
@require_http_methods(["POST"])
def api_extract_invoice_preview(request):
    """
    Step 1: Extract invoice data from uploaded PDF for preview.
    Returns extracted customer, order, and payment information.
    Does NOT create any records yet.
    
    POST fields:
      - file: PDF file to extract
      - selected_order_id (optional): Started order ID to link to
      - plate (optional): Vehicle plate number
      
    Returns:
      - success: true/false
      - header: Customer and payment info {invoice_no, customer_name, address, date, subtotal, tax, total}
      - items: Line items [{description, qty, value}]
      - raw_text: Full extracted text for reference
      - message: Error/status message
    """
    user_branch = get_user_branch(request.user)
    
    # Validate file upload
    uploaded = request.FILES.get('file')
    if not uploaded:
        return JsonResponse({
            'success': False,
            'message': 'No file uploaded'
        })
    
    try:
        file_bytes = uploaded.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        return JsonResponse({
            'success': False,
            'message': 'Failed to read uploaded file'
        })
    
    # Extract text from PDF (non-OCR extractor with filename)
    try:
        from tracker.utils.pdf_text_extractor import extract_from_bytes as extract_pdf_text
        extracted = extract_pdf_text(file_bytes, uploaded.name)
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return JsonResponse({
            'success': False,
            'message': f'Failed to extract invoice data: {str(e)}',
            'error': str(e)
        })
    
    # If extraction failed - still return partial data for manual completion
    if not extracted.get('success'):
        logger.info(f"Extraction failed: {extracted.get('error')} - {extracted.get('message')}")
        return JsonResponse({
            'success': False,
            'message': extracted.get('message', 'Could not extract data from PDF. Please enter invoice details manually.'),
            'error': extracted.get('error'),
            'raw_text': extracted.get('raw_text', ''),
            'header': extracted.get('header', {}),
            'items': extracted.get('items', [])
        })
    
    # Return extracted preview data
    header = extracted.get('header') or {}
    items = extracted.get('items') or []

    return JsonResponse({
        'success': True,
        'message': 'Invoice data extracted successfully',
        'header': {
            'invoice_no': header.get('invoice_no'),
            'code_no': header.get('code_no'),
            'customer_name': header.get('customer_name'),
            'phone': header.get('phone'),
            'email': header.get('email'),
            'address': header.get('address'),
            'reference': header.get('reference'),
            'date': header.get('date'),
            'subtotal': float(header.get('subtotal') or 0),
            'tax': float(header.get('tax') or 0),
            'total': float(header.get('total') or 0),
            'payment_method': header.get('payment_method'),
            'delivery_terms': header.get('delivery_terms'),
            'remarks': header.get('remarks'),
            'attended_by': header.get('attended_by'),
            'kind_attention': header.get('kind_attention'),
            'seller_name': header.get('seller_name'),
            'seller_address': header.get('seller_address'),
            'seller_phone': header.get('seller_phone'),
            'seller_email': header.get('seller_email'),
            'seller_tax_id': header.get('seller_tax_id'),
            'seller_vat_reg': header.get('seller_vat_reg'),
        },
        'items': [
            {
                'description': item.get('description', ''),
                'qty': int(item.get('qty', 1)) if isinstance(item.get('qty'), (int, float)) else 1,
                'unit': item.get('unit'),
                'code': item.get('code'),
                'value': float(item.get('value') or 0),
                'rate': float(item.get('rate') or 0)
            }
            for item in items
        ],
        'raw_text': extracted.get('raw_text', '')
    })


@login_required
@require_http_methods(["POST"])
def api_create_invoice_from_upload(request):
    """
    Step 2: Create/update customer, order, and invoice from extracted invoice data.
    This is called after user confirms extracted data.

    POST fields:
      - selected_order_id (optional): Existing started order to update
      - plate (optional): Vehicle plate number
      - pre_selected_customer_id (optional): Pre-selected customer ID (skip customer creation)

      Customer fields:
      - customer_name: Customer full name
      - customer_phone: Customer phone number
      - customer_email (optional): Customer email
      - customer_address (optional): Customer address
      - customer_type: personal|company|ngo|government

      Invoice fields:
      - invoice_number: Invoice number from invoice
      - invoice_date: Invoice date
      - subtotal: Subtotal amount
      - tax_amount: Tax/VAT amount
      - total_amount: Total amount
      - notes (optional): Additional notes

      Line items (arrays):
      - item_description[]: Item description
      - item_qty[]: Item quantity
      - item_price[]: Item unit price

    Returns:
      - success: true/false
      - invoice_id: Created invoice ID
      - order_id: Created/updated order ID
      - customer_id: Created/updated customer ID
      - redirect_url: URL to view created invoice
    """
    user_branch = get_user_branch(request.user)

    try:
        with transaction.atomic():
            # Priority 1: Use pre-selected customer (from started order detail page)
            # This prevents creating duplicate customers when uploading invoice for known customer
            customer_id = request.POST.get('pre_selected_customer_id') or request.POST.get('customer_id')
            customer_obj = None

            customer_name = request.POST.get('customer_name', '').strip()
            customer_phone = request.POST.get('customer_phone', '').strip()
            customer_email = request.POST.get('customer_email', '').strip() or None
            customer_address = request.POST.get('customer_address', '').strip() or None
            customer_type = request.POST.get('customer_type', 'personal')
            plate = (request.POST.get('plate') or '').strip().upper() or None

            if customer_id:
                try:
                    is_admin = getattr(request.user, 'is_superuser', False) or getattr(request.user, 'is_staff', False)
                    if user_branch and not is_admin:
                        customer_obj = Customer.objects.get(id=int(customer_id), branch=user_branch)
                    else:
                        customer_obj = Customer.objects.get(id=int(customer_id))
                    logger.info(f"Using pre-selected customer for invoice upload: {customer_obj.id}")
                except (Customer.DoesNotExist, ValueError):
                    return JsonResponse({'success': False, 'message': 'Selected customer not found'})

                # If the selected customer is a temporary placeholder from quick start (Plate/PLATE_),
                # check if the extracted customer details already exist as a real customer first.
                try:
                    is_temp = (customer_obj.full_name or '').startswith('Plate ') or (customer_obj.phone or '').startswith('PLATE_')
                    if is_temp and customer_name:
                        # CRITICAL: Before updating temp customer, check if extracted details already belong to another customer
                        # This prevents creating duplicate customers when same invoice is uploaded multiple times
                        existing_real_customer = None

                        # Check if customer with this phone exists (strong identifier)
                        if customer_phone:
                            existing_real_customer = Customer.objects.filter(
                                phone=customer_phone,
                                branch=user_branch
                            ).exclude(id=customer_obj.id).first()

                        # If found existing customer with same phone, use that instead
                        if existing_real_customer:
                            logger.info(f"Found existing customer {existing_real_customer.id} with phone {customer_phone}, using instead of temp customer {customer_obj.id}")
                            customer_obj = existing_real_customer
                        else:
                            # No existing customer found, update the temporary one in-place
                            # Map organization fields from either naming convention
                            org_name = (request.POST.get('organization_name') or request.POST.get('customer_organization_name') or '').strip() or None
                            tax_num = (request.POST.get('tax_number') or request.POST.get('customer_tax_number') or '').strip() or None
                            customer_obj.full_name = customer_name or customer_obj.full_name
                            if customer_phone:
                                customer_obj.phone = customer_phone
                            if customer_email:
                                customer_obj.email = customer_email
                            if customer_address:
                                customer_obj.address = customer_address
                            if customer_type:
                                customer_obj.customer_type = customer_type
                            if org_name:
                                customer_obj.organization_name = org_name
                            if tax_num:
                                customer_obj.tax_number = tax_num
                            customer_obj.save()
                            logger.info(f"Updated temporary customer {customer_obj.id} with extracted details from invoice")
                except Exception as e:
                    logger.warning(f"Failed to check/update temporary customer: {e}")
            else:
                # Require minimum customer info only when creating/looking up by details
                if not customer_name:
                    return JsonResponse({'success': False, 'message': 'Customer name is required'})

                # Map organization fields from either naming convention
                org_name = (request.POST.get('organization_name') or request.POST.get('customer_organization_name') or '').strip() or None
                tax_num = (request.POST.get('tax_number') or request.POST.get('customer_tax_number') or '').strip() or None

                # Use centralized service which does proper deduplication
                try:
                    # If phone missing, try to find by name first within branch
                    if not customer_phone:
                        customer_obj = CustomerService.find_customer_by_name_only(user_branch, customer_name)
                        created = False
                        if not customer_obj:
                            return JsonResponse({'success': False, 'message': 'Customer phone is required to create a new customer'})
                        # If found, update contact information
                        updated = False
                        if customer_email and (not customer_obj.email or customer_obj.email != customer_email):
                            customer_obj.email = customer_email; updated = True
                        if customer_address and (not customer_obj.address or customer_obj.address != customer_address):
                            customer_obj.address = customer_address; updated = True
                        if customer_type and (not customer_obj.customer_type or customer_obj.customer_type != customer_type):
                            customer_obj.customer_type = customer_type; updated = True
                        if org_name and (not customer_obj.organization_name or customer_obj.organization_name != org_name):
                            customer_obj.organization_name = org_name; updated = True
                        if tax_num and (not customer_obj.tax_number or customer_obj.tax_number != tax_num):
                            customer_obj.tax_number = tax_num; updated = True
                        if updated:
                            customer_obj.save()
                    else:
                        customer_obj, created = CustomerService.create_or_get_customer(
                            branch=user_branch,
                            full_name=customer_name,
                            phone=customer_phone,
                            email=customer_email,
                            address=customer_address,
                            customer_type=customer_type,
                            organization_name=org_name,
                            tax_number=tax_num,
                            create_if_missing=True
                        )

                    if not customer_obj:
                        return JsonResponse({
                            'success': False,
                            'message': 'Failed to create or find customer'
                        })

                    if created:
                        logger.info(f"Created new customer from invoice upload: {customer_obj.id} - {customer_name}")
                    else:
                        logger.info(f"Found existing customer for invoice upload: {customer_obj.id} - {customer_name}")

                except Exception as e:
                    logger.error(f"Error in customer creation/lookup for invoice: {e}")
                    return JsonResponse({
                        'success': False,
                        'message': f'Error processing customer: {str(e)}'
                    })

            # Update customer code with extracted Code No if available
            extracted_code_no = request.POST.get('code_no', '').strip()
            if extracted_code_no and customer_obj:
                try:
                    # Check if extracted code_no is different from the current code
                    if customer_obj.code != extracted_code_no:
                        # Check if this code is already used by another customer
                        existing_customer = Customer.objects.filter(code=extracted_code_no).exclude(id=customer_obj.id).first()
                        if not existing_customer:
                            old_code = customer_obj.code
                            customer_obj.code = extracted_code_no
                            customer_obj.save(update_fields=['code'])
                            logger.info(f"Updated customer {customer_obj.id} code from {old_code} to {extracted_code_no}")
                        else:
                            logger.warning(f"Code {extracted_code_no} already used by another customer {existing_customer.id}, keeping current code")
                except Exception as e:
                    logger.warning(f"Failed to update customer code with extracted code_no: {e}")

            # Get or create vehicle if plate provided
            vehicle = None
            if plate:
                try:
                    vehicle = VehicleService.create_or_get_vehicle(customer=customer_obj, plate_number=plate)
                    logger.info(f"Vehicle linked to customer {customer_obj.id}: {plate}")
                except Exception as e:
                    logger.warning(f"Failed to create/get vehicle for customer {customer_obj.id}: {e}")
                    vehicle = None

            # Get existing started order if provided
            selected_order_id = request.POST.get('selected_order_id')
            order = None
            if selected_order_id:
                try:
                    is_admin = getattr(request.user, 'is_superuser', False) or getattr(request.user, 'is_staff', False)
                    if user_branch and not is_admin:
                        order = Order.objects.get(id=int(selected_order_id), branch=user_branch)
                    else:
                        order = Order.objects.get(id=int(selected_order_id))
                    logger.info(f"Found existing order {order.id} to update")
                except Exception as e:
                    logger.warning(f"Could not find existing order {selected_order_id}: {e}")
                    pass

            # If no existing order, create new one
            if not order:
                try:
                    order = OrderService.create_order(
                        customer=customer_obj,
                        order_type='service',
                        branch=user_branch,
                        vehicle=vehicle,
                        description='Created from invoice upload'
                    )
                    logger.info(f"Created new order {order.id} for customer {customer_obj.id}")
                except Exception as e:
                    logger.error(f"Failed to create order for customer {customer_obj.id}: {e}")
                    return JsonResponse({
                        'success': False,
                        'message': f'Failed to create order: {str(e)}'
                    })
            else:
                # Update existing started order to ensure it's linked to the correct customer
                if order.customer_id != customer_obj.id:
                    order.customer = customer_obj
                    logger.info(f"Updated order {order.id} customer from {order.customer_id} to {customer_obj.id}")
                if vehicle and order.vehicle_id != vehicle.id:
                    order.vehicle = vehicle
                    logger.info(f"Updated order {order.id} vehicle to {vehicle.id}")
                order.save(update_fields=['customer', 'vehicle'] if vehicle else ['customer'])
            
            # Create or reuse invoice (enforce one invoice per order)
            inv = None
            if order:
                try:
                    inv = Invoice.objects.filter(order=order).first()
                except Exception:
                    inv = None
            if inv is None:
                inv = Invoice()
            inv.branch = order.branch if order and getattr(order, 'branch', None) else user_branch
            inv.order = order
            inv.customer = customer_obj

            # Parse invoice date
            invoice_date_str = request.POST.get('invoice_date', '')
            try:
                inv.invoice_date = datetime.strptime(invoice_date_str, '%Y-%m-%d').date() if invoice_date_str else timezone.localdate()
            except Exception:
                inv.invoice_date = timezone.localdate()

            # Set invoice fields
            inv.code_no = request.POST.get('code_no', '').strip() or None
            inv.reference = request.POST.get('reference', '').strip() or None

            # Collect all notes/remarks
            notes_parts = []
            if request.POST.get('notes', '').strip():
                notes_parts.append(request.POST.get('notes', '').strip())
            if request.POST.get('remarks', '').strip():
                notes_parts.append(request.POST.get('remarks', '').strip())
            if request.POST.get('delivery_terms', '').strip():
                notes_parts.append(f"Delivery: {request.POST.get('delivery_terms', '').strip()}")
            inv.notes = ' | '.join(notes_parts) if notes_parts else ''

            # Set additional fields
            inv.attended_by = request.POST.get('attended_by', '').strip() or None
            inv.kind_attention = request.POST.get('kind_attention', '').strip() or None
            inv.remarks = request.POST.get('remarks', '').strip() or None

            # Seller information (if provided via POST from extraction preview)
            inv.seller_name = (request.POST.get('seller_name') or '').strip() or None
            inv.seller_address = (request.POST.get('seller_address') or '').strip() or None
            inv.seller_phone = (request.POST.get('seller_phone') or '').strip() or None
            inv.seller_email = (request.POST.get('seller_email') or '').strip() or None
            inv.seller_tax_id = (request.POST.get('seller_tax_id') or '').strip() or None
            inv.seller_vat_reg = (request.POST.get('seller_vat_reg') or '').strip() or None

            # Parse amounts (support multiple possible field names)
            def _dec(val):
                s = str(val or '0')
                try:
                    return Decimal(s.replace(',', ''))
                except Exception:
                    return Decimal('0')

            subtotal = _dec(request.POST.get('subtotal') or request.POST.get('net_value'))
            tax_amount = _dec(request.POST.get('tax_amount') or request.POST.get('tax') or request.POST.get('vat'))
            total_amount = _dec(request.POST.get('total_amount') or request.POST.get('total') or request.POST.get('gross_value'))

            inv.subtotal = subtotal
            inv.tax_amount = tax_amount
            inv.total_amount = total_amount or (subtotal + tax_amount)
            inv.created_by = request.user

            if not getattr(inv, 'invoice_number', None):
                inv.generate_invoice_number()
            inv.save()

            # Save uploaded document if provided (optional in two-step flow)
            try:
                uploaded_file = request.FILES.get('file')
                if uploaded_file:
                    from django.core.files.base import ContentFile
                    try:
                        uploaded_file.seek(0)
                        bytes_ = uploaded_file.read()
                    except Exception:
                        bytes_ = None
                    if bytes_:
                        filename = uploaded_file.name or f"invoice_{inv.invoice_number}.pdf"
                        inv.document.save(filename, ContentFile(bytes_), save=True)
            except Exception:
                # Non-fatal
                pass

            # Create line items with extracted fields
            item_descriptions = request.POST.getlist('item_description[]')
            item_qtys = request.POST.getlist('item_qty[]')
            item_prices = request.POST.getlist('item_price[]')
            item_codes = request.POST.getlist('item_code[]')
            item_units = request.POST.getlist('item_unit[]')
            item_values = request.POST.getlist('item_value[]')

            # Create line items directly from extracted data (no aggregation to preserve extracted values)
            try:
                to_create = []
                for idx, desc in enumerate(item_descriptions):
                    if not desc or not desc.strip():
                        continue
                    try:
                        code = item_codes[idx].strip() if idx < len(item_codes) and item_codes[idx] else None
                        unit = item_units[idx].strip() if idx < len(item_units) and item_units[idx] else None

                        # Parse quantity
                        qty_val = item_qtys[idx] if idx < len(item_qtys) else 1
                        try:
                            qty = Decimal(str(qty_val or 1).replace(',', ''))
                        except Exception:
                            qty = Decimal('1')

                        # Parse unit price
                        price_val = item_prices[idx] if idx < len(item_prices) else Decimal('0')
                        try:
                            unit_price = Decimal(str(price_val or '0').replace(',', ''))
                        except Exception:
                            unit_price = Decimal('0')

                        # Use extracted line value directly (critical: NO RECALCULATION)
                        # This preserves the actual invoice data as extracted from the PDF
                        extracted_value = item_values[idx] if idx < len(item_values) else None
                        try:
                            line_total = Decimal(str(extracted_value or '0').replace(',', '')) if extracted_value else (qty * unit_price)
                        except Exception:
                            line_total = qty * unit_price

                        to_create.append(InvoiceLineItem(
                            invoice=inv,
                            code=code,
                            description=desc.strip(),
                            quantity=qty,
                            unit=unit,
                            unit_price=unit_price,
                            tax_rate=Decimal('0'),
                            line_total=line_total,
                            tax_amount=Decimal('0'),
                        ))
                    except Exception as e:
                        logger.warning(f"Failed to process line item {idx}: {e}")
                        continue

                if to_create:
                    InvoiceLineItem.objects.bulk_create(to_create)
                    logger.info(f"Created {len(to_create)} line items from extracted data with preserved values")
            except Exception as e:
                logger.warning(f"Failed to bulk create line items: {e}")

            # IMPORTANT: Preserve extracted Net, VAT, Gross values for uploaded invoices
            inv.subtotal = subtotal
            inv.tax_amount = tax_amount
            inv.total_amount = total_amount or (subtotal + tax_amount)
            inv.save(update_fields=['subtotal', 'tax_amount', 'total_amount'])

            # Create payment record if total > 0
            if inv.total_amount > 0:
                try:
                    payment = InvoicePayment()
                    payment.invoice = inv
                    payment.amount = Decimal('0')  # Default to unpaid (amount 0)

                    # Map extracted payment method or use form value or default
                    extracted_method = request.POST.get('payment_method', '').strip().lower() or 'on_delivery'
                    payment_method_map = {
                        'cash': 'cash',
                        'cheque': 'cheque',
                        'chq': 'cheque',
                        'bank': 'bank_transfer',
                        'transfer': 'bank_transfer',
                        'card': 'card',
                        'mpesa': 'mpesa',
                        'credit': 'on_credit',
                        'delivery': 'on_delivery',
                        'cod': 'on_delivery',
                        'on_delivery': 'on_delivery',
                    }

                    # Try to match the extracted method to a valid choice
                    payment.payment_method = 'on_delivery'  # Default
                    for key, val in payment_method_map.items():
                        if key in extracted_method:
                            payment.payment_method = val
                            break

                    payment.payment_date = None
                    payment.reference = None
                    payment.save()
                except Exception as e:
                    logger.warning(f"Failed to create payment record: {e}")
            
            # Update started order with invoice data
            try:
                order = OrderService.update_order_from_invoice(
                    order=order,
                    customer=customer_obj,
                    vehicle=vehicle,
                    description=order.description
                )
            except Exception as e:
                logger.warning(f"Failed to update order from invoice: {e}")

            # Handle additional order types/components
            try:
                additional_order_types_json = request.POST.get('additional_order_types', '[]')
                additional_order_types = json.loads(additional_order_types_json) if additional_order_types_json else []

                if additional_order_types:
                    from .models import OrderComponent

                    for component_data in additional_order_types:
                        component_type = component_data.get('type', '').strip()
                        reason = component_data.get('reason', '').strip()

                        # Validate component type
                        if component_type not in ['service', 'sales']:
                            logger.warning(f"Invalid component type: {component_type}")
                            continue

                        # Check if component already exists
                        component, created = OrderComponent.objects.get_or_create(
                            order=order,
                            type=component_type,
                            defaults={
                                'added_by': request.user,
                                'reason': reason,
                                'invoice': inv
                            }
                        )

                        # If component already existed, update it with new reason if provided
                        if not created and reason:
                            component.reason = reason
                            component.invoice = inv
                            component.save(update_fields=['reason', 'invoice'])
                        elif created:
                            logger.info(f"Created OrderComponent for order {order.id}: type={component_type}")
                        else:
                            logger.info(f"OrderComponent already exists for order {order.id}: type={component_type}")

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse additional_order_types JSON: {e}")
            except Exception as e:
                logger.warning(f"Failed to create order components: {e}")

            # Response - redirect to order detail page instead of invoice detail
            return JsonResponse({
                'success': True,
                'message': 'Invoice created and order updated successfully',
                'invoice_id': inv.id,
                'invoice_number': inv.invoice_number,
                'order_id': order.id,
                'customer_id': customer_obj.id,
                'redirect_url': f'/orders/started/{order.id}/'
            })
    
    except Exception as e:
        logger.error(f"Error creating invoice from upload: {e}", exc_info=True)
        return JsonResponse({
            'success': False,
            'message': f'Error: {str(e)}'
        })
