# Labour Codes Implementation Guide

## Overview

This document describes the advanced order type detection system that automatically classifies orders based on invoice item codes. The system supports multiple order types (Service, Sales, Labour, Inquiry) and can identify mixed-type orders when an invoice contains codes mapping to multiple categories.

## Architecture

### 1. Database Models

#### LabourCode Model
- **Location**: `tracker/models.py`
- **Purpose**: Stores the mapping of item codes to order type categories
- **Fields**:
  - `code`: Unique item code (e.g., "22001", "21035")
  - `description`: Item description (e.g., "BRAKE SERVICE", "WHEEL BALANCE")
  - `category`: Order type category (e.g., "labour", "tyre service")
  - `is_active`: Boolean flag for soft-delete capability
  - `created_at`, `updated_at`: Timestamps

#### Order Model Updates
- Added `mixed_categories` field: JSON array storing categories detected from invoice items
- Updated TYPE_CHOICES to include "labour" type
- Supports storing multiple categories for mixed-type orders

### 2. Order Type Detection Logic

#### Location
`tracker/utils/order_type_detector.py`

#### Main Function: `determine_order_type_from_codes(item_codes: List[str])`

**Logic Flow**:
1. Extract unique item codes from invoice
2. Query LabourCode table for matches
3. Collect unique categories from matched codes
4. For unmapped codes → treat as "sales" type
5. Determine final order type:
   - Single category → map to its type (labour, service)
   - Multiple categories → "mixed" type
   - No matches → "sales" type

**Returns**:
- `order_type`: Final order type ("labour", "service", "sales", "mixed")
- `categories`: List of unique categories found
- `mapping_info`: Detailed mapping information

**Example**:
```python
# Input with codes mapping to different categories
item_codes = ["22001", "21035", "UNMAPPED"]
# Code 22001 → "labour" category
# Code 21035 → "tyre service" category  
# Code UNMAPPED → treated as "sales"

order_type, categories, info = determine_order_type_from_codes(item_codes)
# Returns: ("mixed", ["labour", "tyre service"], {...})
```

### 3. Invoice Processing

#### Location
`tracker/views_invoice_upload.py` - `api_create_invoice_from_upload()` function

#### Flow
1. User uploads invoice and it's extracted
2. Item codes are extracted from line items
3. `determine_order_type_from_codes()` is called
4. Detected order type is set on the Order
5. If mixed type, `mixed_categories` field is populated with JSON array

#### Code Snippet
```python
item_codes = request.POST.getlist('item_code[]')
item_codes = [code.strip() for code in item_codes if code and code.strip()]

# Determine order type from item codes
from tracker.utils.order_type_detector import determine_order_type_from_codes
detected_order_type, categories, mapping_info = determine_order_type_from_codes(item_codes)

# Create or update order with detected type
order = OrderService.create_order(
    customer=customer_obj,
    order_type=detected_order_type,  # Uses detected type instead of hardcoded "service"
    ...
)

# Update order with mixed categories
if order and detected_order_type == 'mixed' and categories:
    order.mixed_categories = json.dumps(categories)
    order.save()
```

### 4. Frontend Display

#### Template Filters
- **Location**: `tracker/templatetags/order_filters.py`
- **Filters**:
  - `order_type_display`: Returns formatted order type (e.g., "Labour and Service")
  - `order_type_badge`: Returns HTML badge element

#### Usage in Templates
```django
{% load order_filters %}

<!-- Display order type -->
{{ order|order_type_display }}
<!-- Output: "Labour and Service" for mixed types -->

<!-- Display badge -->
{{ order|order_type_badge }}
<!-- Output: <span class="badge bg-purple">Labour & Service</span> -->
```

#### Order List Display (`orders_list.html`)
- Type filter dropdown includes "Labour" and "Mixed" options
- Mixed-type orders show purple badge with "Labour and Service" format
- Labour orders show yellow badge with tools icon

#### Order Detail Display (`order_detail.html`)
- Added Labour Details section for labour-type orders
- Mixed order types display an info alert explaining the combination
- Frontend shows which categories were detected

### 5. Admin Interface

#### Location
`tracker/admin.py` - `LabourCodeAdmin` class

#### Features
- Full CRUD operations for Labour Codes
- Searchable by code, description, and category
- Filterable by category and active status
- Read-only timestamps

#### Usage
1. Navigate to Django admin
2. Go to "Labour Codes" section
3. Create, update, or delete codes as needed

## Installation & Setup

### Step 1: Apply Migrations
```bash
python manage.py migrate
```

This creates:
- `LabourCode` table
- `mixed_categories` field in Order table

### Step 2: Seed Labour Codes
```bash
python manage.py seed_labour_codes
```

This populates the database with all codes from the provided spreadsheet (45+ codes).

#### Alternative: Seed with Clear
To replace existing codes:
```bash
python manage.py seed_labour_codes --clear
```

### Step 3: Verify Setup
```bash
python test_labour_code_implementation.py
```

## Data Reference

### Labour Codes Included
The system includes 45+ item codes with their categories:

#### Labour Category Codes
- 22001: BRAKE PAD FITMENT
- 22005: OIL SERVICE
- 22009-22016: AC/DIAGNOSTICS SERVICE
- 31005: SPARE SERVICE
- And 20+ more...

#### Tyre Service Category Codes
- 21002-21096: WHEEL BALANCE, TYRE FITTING, ALIGNMENT
- 31004: TYRE MAINTENANCE SERVICE
- And more...

## Usage Example: Invoice Upload Workflow

### Step 1: Upload Invoice
User uploads invoice with line items having codes:
- Code 22001 (mapped to "labour")
- Code 21035 (mapped to "tyre service")

### Step 2: Automatic Detection
System detects:
- Multiple categories → "mixed" type
- Creates order with `type="mixed"`
- Stores `mixed_categories='["labour", "tyre service"]'`

### Step 3: Display in UI
- Orders list shows: "Labour and Service" with purple badge
- Order detail page shows alert: "This order contains multiple types: Labour and Service"
- Categories list displayed: "labour, tyre service"

### Step 4: Admin Management
Admin can:
- View mixed-type orders via filter
- Edit order details if needed
- Manage Labour Codes in admin panel

## API Response Examples

### API: Determine Order Type
```python
# Request
determine_order_type_from_codes(["22001", "21035"])

# Response
(
    "mixed",  # order_type
    ["labour", "tyre service"],  # categories
    {
        "mapped": {"22001": "labour", "21035": "tyre service"},
        "unmapped": [],
        "categories_found": ["labour", "tyre service"],
        "order_types_found": ["labour", "service"]
    }
)
```

### API: Order Type Display
```python
# For single type
get_mixed_order_status_display("labour", ["labour"])
# Returns: "Labour"

# For mixed types
get_mixed_order_status_display("mixed", ["labour", "service"])
# Returns: "Labour and Service"
```

## Testing

### Automated Tests
Run the test suite:
```bash
python test_labour_code_implementation.py
```

Tests verify:
- ✓ LabourCode model CRUD
- ✓ Order type detection logic
- ✓ Mixed order status display
- ✓ Order model mixed_categories field

### Manual Testing
1. Create/edit Labour Codes in admin
2. Upload an invoice with mixed item codes
3. Verify order type is correctly detected
4. Check that mixed_categories is populated
5. Verify display in orders list and detail pages

## Troubleshooting

### Issue: Order type not updating from invoice
**Solution**: 
- Ensure item codes in invoice match codes in LabourCode table (case-sensitive)
- Check that LabourCode entries have `is_active=True`
- Verify migration was applied: `python manage.py migrate`

### Issue: Mixed order type not displaying correctly
**Solution**:
- Check that `mixed_categories` is stored as valid JSON
- Ensure `order_filters.py` template tag is loaded in template
- Verify `bg-purple` CSS class is defined (should be automatic)

### Issue: Labour Codes not appearing in admin
**Solution**:
- Run seed command: `python manage.py seed_labour_codes`
- Check that admin user has proper permissions
- Verify LabourCode is registered in admin.py

## Future Enhancements

Potential improvements:
1. Bulk upload Labour Codes via CSV
2. Import/Export functionality in admin
3. Audit trail for code changes
4. Auto-categorization based on code patterns
5. Integration with supplier systems
6. Multi-language support for descriptions

## Performance Considerations

- LabourCode table indexed on: `code`, `category`, `is_active`
- Order type detection uses efficient database query
- JSON storage for mixed_categories is minimal overhead
- Template filters cached by Django

## Support

For issues or questions:
1. Check the test file: `test_labour_code_implementation.py`
2. Review admin interface: Django admin → Labour Codes
3. Check order mixed_categories field in database
4. Review logs for order creation process

## Summary

The Labour Codes implementation provides:
- ✅ Automatic order type detection from invoice item codes
- ✅ Support for mixed-type orders with category tracking
- ✅ User-friendly admin interface for code management
- ✅ Frontend display of mixed order types
- ✅ Backward compatible with existing order types
- ✅ Fully tested and documented
