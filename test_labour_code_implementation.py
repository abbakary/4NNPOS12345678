#!/usr/bin/env python
"""
Test script for the Labour Code implementation.
Run with: python test_labour_code_implementation.py
"""

import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pos_tracker.settings')
django.setup()

from tracker.models import LabourCode, Order, Customer, Branch, Invoice, InvoiceLineItem
from tracker.utils.order_type_detector import determine_order_type_from_codes, get_mixed_order_status_display
from datetime import datetime
from decimal import Decimal
import json


def test_labour_code_model():
    """Test that LabourCode model works correctly"""
    print("\n" + "=" * 80)
    print("TEST 1: LabourCode Model")
    print("=" * 80)
    
    # Create a test labour code
    code, created = LabourCode.objects.update_or_create(
        code="TEST001",
        defaults={
            "description": "Test Labour Code",
            "category": "labour",
            "is_active": True
        }
    )
    
    print(f"✓ Created/Updated LabourCode: {code}")
    print(f"  Code: {code.code}")
    print(f"  Description: {code.description}")
    print(f"  Category: {code.category}")
    
    # Clean up
    code.delete()
    print("✓ Deleted test code")


def test_order_type_detection():
    """Test the order type detection logic"""
    print("\n" + "=" * 80)
    print("TEST 2: Order Type Detection")
    print("=" * 80)
    
    # Create test labour codes
    labour_codes = [
        {"code": "22001", "description": "BRAKE SERVICE", "category": "labour"},
        {"code": "21035", "description": "WHEEL BALANCE", "category": "service"},
    ]
    
    for lc in labour_codes:
        LabourCode.objects.update_or_create(
            code=lc["code"],
            defaults={
                "description": lc["description"],
                "category": lc["category"],
                "is_active": True
            }
        )
    
    # Test 1: Single labour code
    order_type, categories, info = determine_order_type_from_codes(["22001"])
    print(f"\nTest A - Single Labour Code:")
    print(f"  Input: ['22001']")
    print(f"  Order Type: {order_type}")
    print(f"  Categories: {categories}")
    assert order_type == "labour", f"Expected 'labour', got '{order_type}'"
    print("  ✓ PASS")
    
    # Test 2: Single service code
    order_type, categories, info = determine_order_type_from_codes(["21035"])
    print(f"\nTest B - Single Service Code:")
    print(f"  Input: ['21035']")
    print(f"  Order Type: {order_type}")
    print(f"  Categories: {categories}")
    assert order_type == "service", f"Expected 'service', got '{order_type}'"
    print("  ✓ PASS")
    
    # Test 3: Mixed codes (labour + service)
    order_type, categories, info = determine_order_type_from_codes(["22001", "21035"])
    print(f"\nTest C - Mixed Codes (Labour + Service):")
    print(f"  Input: ['22001', '21035']")
    print(f"  Order Type: {order_type}")
    print(f"  Categories: {categories}")
    assert order_type == "mixed", f"Expected 'mixed', got '{order_type}'"
    assert "labour" in categories and "service" in categories
    print("  ��� PASS")
    
    # Test 4: Unmapped codes (treated as sales)
    order_type, categories, info = determine_order_type_from_codes(["UNMAPPED001", "UNMAPPED002"])
    print(f"\nTest D - Unmapped Codes (Sales Default):")
    print(f"  Input: ['UNMAPPED001', 'UNMAPPED002']")
    print(f"  Order Type: {order_type}")
    print(f"  Categories: {categories}")
    assert order_type == "sales", f"Expected 'sales', got '{order_type}'"
    print("  ✓ PASS")
    
    # Test 5: Mixed - labour + unmapped (labour + sales)
    order_type, categories, info = determine_order_type_from_codes(["22001", "UNMAPPED"])
    print(f"\nTest E - Mixed Labour + Unmapped (Labour + Sales):")
    print(f"  Input: ['22001', 'UNMAPPED']")
    print(f"  Order Type: {order_type}")
    print(f"  Categories: {categories}")
    print(f"  Order Types Found: {info.get('order_types_found', [])}")
    assert order_type == "mixed", f"Expected 'mixed', got '{order_type}'"
    print("  ✓ PASS")
    
    # Clean up
    LabourCode.objects.filter(code__in=["22001", "21035"]).delete()
    print("\n✓ Cleaned up test codes")


def test_mixed_order_status_display():
    """Test the mixed order status display"""
    print("\n" + "=" * 80)
    print("TEST 3: Mixed Order Status Display")
    print("=" * 80)
    
    test_cases = [
        {
            "order_type": "labour",
            "order_types_found": ["labour"],
            "expected": "Labour"
        },
        {
            "order_type": "mixed",
            "order_types_found": ["labour", "sales"],
            "expected": "Labour and Sales"
        },
        {
            "order_type": "mixed",
            "order_types_found": ["service", "labour"],
            "expected": "Labour and Service"
        },
        {
            "order_type": "sales",
            "order_types_found": ["sales"],
            "expected": "Sales"
        },
    ]
    
    for idx, test in enumerate(test_cases, 1):
        result = get_mixed_order_status_display(test["order_type"], test["order_types_found"])
        print(f"\nTest {chr(64 + idx)}: {test['order_type']}")
        print(f"  Order Types: {test['order_types_found']}")
        print(f"  Display: {result}")
        print(f"  Expected: {test['expected']}")
        assert result == test["expected"], f"Expected '{test['expected']}', got '{result}'"
        print("  ✓ PASS")


def test_order_model_mixed_categories():
    """Test that Order model stores mixed categories correctly"""
    print("\n" + "=" * 80)
    print("TEST 4: Order Model Mixed Categories")
    print("=" * 80)
    
    # Create test branch and customer
    branch, _ = Branch.objects.get_or_create(
        name="Test Branch",
        defaults={"code": "TEST", "is_active": True}
    )
    
    customer, _ = Customer.objects.get_or_create(
        full_name="Test Customer",
        phone="0123456789",
        defaults={"code": "TESTCUST", "branch": branch}
    )
    
    # Create order with mixed type
    categories = ["labour", "service"]
    order = Order.objects.create(
        order_number="TEST-ORDER-001",
        customer=customer,
        branch=branch,
        type="mixed",
        mixed_categories=json.dumps(categories),
        status="created"
    )
    
    print(f"✓ Created Order: {order.order_number}")
    print(f"  Type: {order.type}")
    print(f"  Mixed Categories: {order.mixed_categories}")
    
    # Reload and verify
    order.refresh_from_db()
    stored_categories = json.loads(order.mixed_categories)
    print(f"  Reloaded Categories: {stored_categories}")
    assert stored_categories == categories
    print("  ✓ PASS")
    
    # Clean up
    order.delete()
    customer.delete()
    branch.delete()
    print("✓ Cleaned up test data")


def main():
    """Run all tests"""
    print("\n" + "█" * 80)
    print("LABOUR CODE IMPLEMENTATION TEST SUITE")
    print("█" * 80)
    
    try:
        test_labour_code_model()
        test_order_type_detection()
        test_mixed_order_status_display()
        test_order_model_mixed_categories()
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        print("\nImplementation Summary:")
        print("  ✓ LabourCode model created and working")
        print("  ✓ Order type detection logic functional")
        print("  ✓ Mixed order status display working")
        print("  ✓ Order model mixed_categories field working")
        print("\nNext Steps:")
        print("  1. Run: python manage.py migrate")
        print("  2. Run: python manage.py seed_labour_codes")
        print("  3. Upload an invoice to test the system")
        print("=" * 80 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
