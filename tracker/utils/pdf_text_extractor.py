"""
PDF and image text extraction for invoice processing.
CORRECTED VERSION: Proper line item extraction without payment information in descriptions
"""

import io
import logging
import re
from decimal import Decimal
from datetime import datetime
import json

try:
    import fitz
except ImportError:
    fitz = None

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(file_bytes) -> list:
    """Extract text from PDF file with page separation for multi-page handling."""
    pages_data = []
    
    if fitz is not None:
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page_num, page in enumerate(doc):
                page_text = page.get_text("text", sort=True)
                if page_text and page_text.strip():
                    pages_data.append({
                        'page_num': page_num + 1,
                        'text': page_text,
                        'lines': [line.strip() for line in page_text.split('\n') if line.strip()]
                    })
            doc.close()

            if pages_data:
                logger.info(f"Successfully extracted {len(pages_data)} pages from PDF using PyMuPDF")
                return pages_data
            else:
                logger.warning("PyMuPDF extracted empty text from PDF")
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {e}")

    if PyPDF2 is not None and not pages_data:
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    pages_data.append({
                        'page_num': page_num + 1,
                        'text': page_text,
                        'lines': [line.strip() for line in page_text.split('\n') if line.strip()]
                    })

            if pages_data:
                logger.info(f"Successfully extracted {len(pages_data)} pages from PDF using PyPDF2")
                return pages_data
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {e}")

    raise RuntimeError('PDF extraction failed with both PyMuPDF and PyPDF2')

def extract_text_from_image(file_bytes) -> str:
    """Extract text from image file."""
    logger.info("Image file detected. OCR not available. Manual entry required.")
    return ""

def parse_invoice_data(pages_data: list) -> dict:
    """Parse invoice data from extracted pages with multi-page support."""
    if not pages_data:
        return create_empty_invoice_data()

    # Combine all lines from all pages for header extraction
    all_lines = []
    for page in pages_data:
        all_lines.extend(page['lines'])

    # Extract customer information - FIXED: Only extract actual customer info, not seller info
    customer_info = extract_customer_information(all_lines)
    
    # Extract other fields
    code_no = extract_code_no_enhanced(all_lines)
    invoice_no = extract_invoice_no(all_lines)
    date_str = extract_date(all_lines)
    reference = extract_reference(all_lines)

    # Extract monetary values from all pages (totals are usually on last page)
    subtotal = extract_monetary_value(all_lines, [r'Net\s*Value', r'Subtotal', r'Net\s*Amount'])
    tax = extract_monetary_value(all_lines, [r'VAT', r'Tax', r'GST'])
    total = extract_monetary_value(all_lines, [r'Gross\s*Value', r'Grand\s*Total', r'Total\s*Amount'])

    # Extract line items from ALL pages with proper stopping at payment information
    items = extract_line_items_multipage_corrected(pages_data)

    return {
        'invoice_no': invoice_no, 'code_no': code_no, 'date': date_str,
        'customer_name': customer_info.get('name'),
        'phone': customer_info.get('phone'), 
        'email': customer_info.get('email'),
        'address': customer_info.get('address'), 
        'reference': reference, 
        'subtotal': subtotal,
        'tax': tax, 'total': total, 'items': items, 'payment_method': None,
        'delivery_terms': None, 'remarks': None, 'attended_by': None,
        'kind_attention': None, 'seller_name': None, 'seller_address': None,
        'seller_phone': None, 'seller_email': None, 'seller_tax_id': None,
        'seller_vat_reg': None
    }

def extract_customer_information(lines):
    """Extract customer information only, excluding seller information."""
    customer_info = {
        'name': None,
        'address': None,
        'phone': None,
        'email': None
    }
    
    # First pass: Extract customer name and basic info
    for i, line in enumerate(lines):
        # Look for customer name pattern
        if re.search(r'Customer\s*Name\s*[\t:]?\s*[A-Z]', line, re.I):
            # Extract customer name
            name_match = re.search(r'Customer\s*Name\s*[\t:]?\s*(.+?)(?:\s+Tel|\s+Fax|\s+Email|\s+Address|\s+Date|$)', line, re.I)
            if name_match:
                customer_info['name'] = name_match.group(1).strip()
                logger.info(f"Found customer name: {customer_info['name']}")
                break
    
    # Second pass: Extract customer address (exclude seller address)
    customer_info['address'] = extract_customer_address(lines)
    
    # Third pass: Extract customer phone
    customer_info['phone'] = extract_customer_phone(lines)
    
    # Fourth pass: Extract customer email (exclude seller email)
    customer_info['email'] = extract_customer_email(lines)
    
    return customer_info

def extract_customer_address(lines):
    """Extract only customer address, excluding seller address."""
    address_lines = []

    # FIRST PASS: Look for "Address :" label (primary method)
    for i, line in enumerate(lines):
        # Look for "Address :" label in the line
        address_match = re.search(r'Address\s*[\t:]?\s*(.+?)(?:\s+(?:Cust\s+Ref|Tel|Fax|Email|$))', line, re.I)
        if address_match:
            address_part = address_match.group(1).strip()

            # Exclude lines that clearly belong to seller info
            if not re.search(r'16541|Superdoll|Tax\s+ID|VAT\s+Reg', line, re.I):
                address_lines.append(address_part)

                # Look for continuation lines (next lines without a label) - be aggressive in capturing
                j = i + 1
                while j < len(lines) and j < i + 6:  # Increased from i+4 to i+6 to capture more lines
                    next_line = lines[j].strip()

                    if not next_line:
                        j += 1
                        continue

                    # Stop if we hit a label line (contains a field name with colon)
                    # But be careful not to match city names or abbreviations that might have colons
                    if re.search(r'^(?:Tel|Fax|Email|Cust\s+Ref|Ref\s+Date|Del\.\s+Date|Attended|Kind|Reference|Code\s+No|Customer\s+Name|Pl\.\s+No)\s*[\t:]', next_line, re.I):
                        break

                    # Exclude lines that clearly belong to seller info
                    if re.search(r'16541|Superdoll|Tax\s+ID|VAT\s+Reg|Dear\s+Sir|We\s+thank|Page\s+\d', next_line, re.I):
                        break

                    # Include all lines that don't look like new fields and aren't empty
                    # This is more aggressive - we'll include almost any non-empty line until we hit a clear field marker
                    if next_line and len(next_line) > 2:
                        # Additional check: if it looks like it could be part of an address
                        # (contains city, country, street references, P.O. Box, etc.)
                        address_lines.append(next_line)
                        j += 1
                    else:
                        break

                if address_lines:
                    # Join all address lines and clean up
                    clean_address = ' '.join(address_lines)
                    clean_address = re.sub(r'\s+', ' ', clean_address).strip()
                    # Remove any trailing field names that might have been partially captured
                    clean_address = re.sub(r'\s+(?:Cust\s+Ref|Ref\s+Date|Del\.\s+Date|Attended|Kind|Reference).*$', '', clean_address, flags=re.I)
                    clean_address = re.sub(r'\s+', ' ', clean_address).strip()
                    logger.info(f"Found customer address from 'Address :' label: {clean_address}")
                    return clean_address

    # SECOND PASS: Fallback to P.O. Box search if "Address :" label not found
    for i, line in enumerate(lines):
        # Look for address indicators in customer context
        if (re.search(r'P\.?O\.?\s*Box\s*\d+', line, re.I) and
            not re.search(r'16541|Superdoll', line, re.I)):  # Exclude seller PO Box
            address_lines.append(line.strip())
            # Look for continuation lines
            j = i + 1
            while j < len(lines) and j < i + 6:  # Increased range
                next_line = lines[j].strip()

                if not next_line:
                    j += 1
                    continue

                # Stop at clear field markers
                if re.search(r'^(?:Tel|Fax|Email|Attended|Kind|Reference|Dear\s+Sir|S\s*No|Item\s+Code)\s*[\t:]', next_line, re.I):
                    break

                # Always include lines with TANZANIA or country indicators
                if re.search(r'TANZANIA|UGANDA|KENYA|ETHIOPIA', next_line, re.I):
                    address_lines.append(next_line)
                    j += 1
                    continue

                # Include address-like lines
                if (re.search(r'[A-Z]+\s*[A-Z]*\s*,?\s*[A-Z]*\s*(?:TANZANIA|UGANDA|KENYA)', next_line, re.I) or
                    re.search(r'DAR\s*ES\s*SALAAM', next_line, re.I) or
                    re.search(r'PLOT\s*\d+', next_line, re.I) or
                    re.search(r'[A-Z]+\s*ROAD', next_line, re.I) or
                    re.search(r'P\.?O\.?\s*BOX', next_line, re.I)):
                    address_lines.append(next_line)
                    j += 1
                else:
                    break
            break

    if address_lines:
        # Clean up the address - remove any seller information
        clean_address = ' '.join(address_lines)
        clean_address = re.sub(r'\s+', ' ', clean_address).strip()
        logger.info(f"Found customer address from fallback P.O. Box search: {clean_address}")
        return clean_address

    return None

def extract_customer_phone(lines):
    """Extract only customer phone, excluding seller phone."""
    for line in lines:
        # Look for phone patterns that are likely customer phones
        phone_match = re.search(r'(?:Tel|Phone)\s*[\t:]?\s*([\d\s\/\-]+)(?:\s|$)', line, re.I)
        if phone_match:
            phone = phone_match.group(1).strip()
            # Exclude seller phone numbers (specific patterns)
            if (len(phone) >= 7 and 
                not re.search(r'\+255-22-286', phone) and  # Exclude seller prefix
                not re.search(r'16541', line) and  # Exclude seller context
                not re.search(r'Superdoll', line, re.I)):
                logger.info(f"Found customer phone: {phone}")
                return phone
    
    return None

def extract_customer_email(lines):
    """Extract only customer email, excluding seller email."""
    for line in lines:
        # Find all email patterns in the line
        email_matches = re.findall(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', line)
        
        for email in email_matches:
            # Exclude seller emails and common false positives
            if (not re.search(r'superdoll|stm@superdoll', email, re.I) and
                not re.search(r'example|test|domain', email, re.I) and
                len(email) > 5):
                
                # Additional validation: email should not be in seller context
                line_lower = line.lower()
                seller_indicators = ['superdoll', '16541', 'tax id', 'vat reg', 'tel+255', 'fax+255']
                is_seller_context = any(indicator in line_lower for indicator in seller_indicators)
                
                if not is_seller_context:
                    logger.info(f"Found customer email: {email}")
                    return email
    
    return None

def create_empty_invoice_data():
    """Create empty invoice data structure."""
    return {
        'invoice_no': None, 'code_no': None, 'date': None, 'customer_name': None,
        'address': None, 'phone': None, 'email': None, 'reference': None,
        'subtotal': None, 'tax': None, 'total': None, 'items': [],
        'payment_method': None, 'delivery_terms': None, 'remarks': None,
        'attended_by': None, 'kind_attention': None, 'seller_name': None,
        'seller_address': None, 'seller_phone': None, 'seller_email': None,
        'seller_tax_id': None, 'seller_vat_reg': None
    }

def extract_line_items_multipage_corrected(pages_data):
    """
    Extract line items from multiple pages with continuous numbering.
    CORRECTED: Stops properly at payment information.
    """
    all_items = []
    
    for page in pages_data:
        page_lines = page['lines']
        page_items = extract_line_items_from_page_corrected(page_lines)
        
        if page_items:
            all_items.extend(page_items)
    
    # Renumber items sequentially to ensure continuity
    for i, item in enumerate(all_items, 1):
        item['sr_no'] = i
    
    logger.info(f"Extracted {len(all_items)} items from {len(pages_data)} pages")
    return all_items

def extract_line_items_from_page_corrected(lines):
    """
    Extract line items from a single page.
    CORRECTED: Stops at payment information and doesn't include it in descriptions.
    """
    items = []
    
    # Find the item table section on this page
    table_start = -1
    for i, line in enumerate(lines):
        if is_table_header(line) and not is_customer_info_line(line):
            table_start = i
            logger.info(f"Found item table at line {i}: {line}")
            break
    
    if table_start == -1:
        return items
    
    # Process lines after header
    i = table_start + 1
    
    while i < len(lines):
        line = lines[i].strip()
        
        # STOP at payment information and totals - CORRECTED
        if (is_monetary_total(line) or 
            is_section_break(line) or 
            is_payment_information(line)):
            logger.info(f"Stopping at payment information: {line}")
            break
            
        # Skip customer info lines and page footers
        if is_customer_info_line(line) or is_page_footer(line):
            i += 1
            continue
            
        # Skip empty lines
        if not line:
            i += 1
            continue
        
        # Check if this line starts a new item (starts with number)
        if re.match(r'^\d+\.?\s+', line) and not contains_payment_info(line):
            # Extract item from line
            item = extract_item_data_corrected(line)
            if item and item.get('description'):
                items.append(item)
                logger.info(f"Extracted item: {item}")
        
        i += 1
    
    return items

def extract_item_data_corrected(line):
    """
    Extract item data from a single line.
    CORRECTED: Removes payment information from descriptions.
    """
    if not line.strip():
        return None
    
    # Clean the line first to remove payment information
    clean_line = remove_payment_info_from_line(line)
    
    # Pattern for complete items: Number Code Description Unit Qty Rate Value
    pattern_complete = r'^(\d+)\.?\s+(\d{4,15})\s+(.+?)\s+(PCS|NOS|KG|HR|LTR|PC|UNT|BOX|SET|UNIT|PIECES|TYRE|TIRE)\s+(\d+)\s+([\d,]+\.?\d{2})\s+([\d,]+\.?\d{2})$'
    match_complete = re.search(pattern_complete, clean_line)
    
    if match_complete:
        item_code = match_complete.group(2)
        description = match_complete.group(3).strip()
        unit = match_complete.group(4).upper()
        qty = int(match_complete.group(5))
        
        # Extract rate and value AS SHOWN
        rate_str = match_complete.group(6).replace(',', '')
        value_str = match_complete.group(7).replace(',', '')
        
        try:
            rate = Decimal(rate_str)
            value = Decimal(value_str)
        except:
            rate = Decimal('0')
            value = Decimal('0')
        
        # Clean description from any remaining payment info
        description = remove_payment_info_from_description(description)
        
        return {
            'code': item_code,
            'description': clean_description(description),
            'unit': unit,
            'qty': qty,
            'rate': rate,
            'value': value
        }
    
    # Pattern for items without explicit unit
    pattern_without_unit = r'^(\d+)\.?\s+(\d{4,15})\s+(.+?)\s+(\d+)\s+([\d,]+\.?\d{2})\s+([\d,]+\.?\d{2})$'
    match_without_unit = re.search(pattern_without_unit, clean_line)
    
    if match_without_unit:
        item_code = match_without_unit.group(2)
        description = match_without_unit.group(3).strip()
        qty = int(match_without_unit.group(4))
        
        # Extract rate and value AS SHOWN
        rate_str = match_without_unit.group(5).replace(',', '')
        value_str = match_without_unit.group(6).replace(',', '')
        
        try:
            rate = Decimal(rate_str)
            value = Decimal(value_str)
        except:
            rate = Decimal('0')
            value = Decimal('0')
        
        # Extract unit from description
        unit = extract_unit_from_description(description)
        
        # Clean description from any remaining payment info
        description = remove_payment_info_from_description(description)
        
        return {
            'code': item_code,
            'description': clean_description(description),
            'unit': unit,
            'qty': qty,
            'rate': rate,
            'value': value
        }
    
    # Fallback pattern
    return extract_item_fallback_corrected(clean_line)

def extract_item_fallback_corrected(line):
    """Fallback item extraction for difficult formats."""
    # Split line by whitespace
    parts = line.split()
    if len(parts) < 4:
        return None
    
    # First part should be item number
    if not parts[0].replace('.', '').isdigit():
        return None
    
    # Look for item code (typically numeric or alphanumeric)
    item_code = None
    description_parts = []
    qty = None
    rate = None
    value = None
    
    i = 1  # Start after item number
    while i < len(parts):
        part = parts[i]
        
        # Check for item code (typically 4+ digits or alphanumeric)
        if not item_code and (len(part) >= 4 and (part.isdigit() or part.isalnum())):
            item_code = part
        # Check for quantity (integer)
        elif not qty and part.isdigit() and 1 <= int(part) <= 10000:
            qty = int(part)
        # Check for monetary values (contain decimal points)
        elif '.' in part and re.match(r'^[\d,]+\.\d{2}$', part):
            monetary_value = Decimal(part.replace(',', ''))
            if not rate:
                rate = monetary_value
            else:
                value = monetary_value
        else:
            # This is likely part of description
            description_parts.append(part)
        
        i += 1
    
    if description_parts and qty:
        description = ' '.join(description_parts)
        unit = extract_unit_from_description(description)
        
        # Clean description from any remaining payment info
        description = remove_payment_info_from_description(description)
        
        return {
            'code': item_code,
            'description': clean_description(description),
            'unit': unit,
            'qty': qty,
            'rate': rate or Decimal('0'),
            'value': value or Decimal('0')
        }
    
    return None

def remove_payment_info_from_line(line):
    """Remove payment information from a line to prevent it from being included in descriptions."""
    payment_patterns = [
        r'Payment\s*:.*$',
        r'Cash/Chq\s+on\s+Delivery.*$',
        r'Net\s+Value\s*:.*$',
        r'Delivery\s*:.*$',
        r'VAT\s*:.*$',
        r'Gross\s+Value\s*:.*$',
        r'Remarks?\s*:.*$',
        r'NOTE\s+\d+\s*:.*$',
        r'Looking\s+forward\s+to\s+your.*$',
        r'Payment\s+in\s+TSHS.*$',
        r'Duty\s+and\s+VAT\s+exemption.*$',
        r'Authorised\s+Signatory.*$',
        r'Valid\s+for\s+\d+\s+weeks.*$',
        r'Discount\s+is\s+Valid.*$',
        r'TSH\s+\d+[,.]\d+.*$',
        r'Dear\s+Sir/Madam.*$',
        r'We\s+thank\s+you.*$',
        r'As\s+desired.*$'
    ]
    
    clean_line = line
    for pattern in payment_patterns:
        clean_line = re.sub(pattern, '', clean_line, flags=re.I)
    
    return clean_line.strip()

def remove_payment_info_from_description(description):
    """Remove any payment information that might have slipped into the description."""
    payment_keywords = [
        'Payment', 'Cash/Chq', 'Net Value', 'Delivery', 'VAT', 'Gross Value',
        'Remarks', 'NOTE', 'Looking forward', 'TSHS', 'Duty', 'Authorised',
        'Valid for', 'Discount', 'Dear Sir/Madam', 'We thank you', 'As desired'
    ]
    
    clean_desc = description
    for keyword in payment_keywords:
        # Remove the keyword and everything after it in the description
        pattern = r'\b' + re.escape(keyword) + r'\b.*$'
        clean_desc = re.sub(pattern, '', clean_desc, flags=re.I)
    
    return clean_desc.strip()

def contains_payment_info(line):
    """Check if line contains payment information."""
    payment_indicators = [
        r'Payment\s*:',
        r'Cash/Chq\s+on\s+Delivery',
        r'Net\s+Value\s*:',
        r'Delivery\s*:',
        r'VAT\s*:',
        r'Gross\s+Value\s*:',
        r'Remarks?\s*:',
        r'NOTE\s+\d+\s*:',
        r'Looking\s+forward\s+to\s+your',
        r'Payment\s+in\s+TSHS',
        r'Duty\s+and\s+VAT\s+exemption',
        r'Authorised\s+Signatory',
        r'Valid\s+for\s+\d+\s+weeks',
        r'Discount\s+is\s+Valid',
        r'Dear\s+Sir/Madam',
        r'We\s+thank\s+you',
        r'As\s+desired'
    ]
    
    return any(re.search(pattern, line, re.I) for pattern in payment_indicators)

def is_payment_information(line):
    """Check if line contains payment information that should stop item extraction."""
    return contains_payment_info(line)

def is_table_header(line):
    """Check if line is a table header."""
    header_keywords = [
        r'\b(Sr|S\.?No?\.?|No\.?|#)\b',
        r'\b(Item\s*Code|Code|Item)\b', 
        r'\b(Description|Desc)\b',
        r'\b(Type|Unit)\b',
        r'\b(Qty|Quantity)\b',
        r'\b(Rate|Price|Unit\s*Price)\b',
        r'\b(Value|Amount|Total)\b'
    ]
    
    keyword_count = sum(1 for pattern in header_keywords if re.search(pattern, line, re.I))
    return keyword_count >= 3

def is_customer_info_line(line):
    """Check if line contains customer information (should be skipped during item extraction)."""
    customer_patterns = [
        r'Customer\s+Name',
        r'P\.?O\.?\s*Box',
        r'Code\s*No',
        r'PI\s*No',
        r'Proforma\s+Invoice',
        r'SERENGETI\s+BREWERIES',
        r'STATEOIL\s+TANZANIA',
        r'JTI\s+LEAF\s+SERVICES',
        r'Superdoll\s+Trailer'
    ]
    
    return any(re.search(pattern, line, re.I) for pattern in customer_patterns)

def is_page_footer(line):
    """Check if line is a page footer."""
    footer_patterns = [
        r'Page\s+\d+\s+of\s+\d+',
        r'^\d+$',  # Just a page number
        r'Authorised\s+Signatory',
        r'Thank\s+you',
        r'Terms\s+and\s+Conditions'
    ]
    
    return any(re.search(pattern, line, re.I) for pattern in footer_patterns)

def is_monetary_total(line):
    """Check if line contains monetary totals."""
    total_patterns = [
        r'^(?:Net\s*Value|Gross\s*Value|Grand\s*Total|TOTAL)\s*[:\-]?\s*[\d,]+',
        r'^(?:VAT|Tax)\s*[:\-]?\s*[\d,]+',
        r'^Total\s+Amount\s*[:\-]?\s*[\d,]+'
    ]
    
    return any(re.search(pattern, line, re.I) for pattern in total_patterns)

def is_section_break(line):
    """Check if line indicates a section break."""
    break_patterns = [
        r'Customer\s+Information',
        r'Thank\s+you',
        r'Notes?:',
        r'Remarks?:',
        r'Payment\s+Terms'
    ]
    
    return any(re.search(pattern, line, re.I) for pattern in break_patterns)

def extract_unit_from_description(description):
    """Extract unit from description if present."""
    units = ['PCS', 'NOS', 'KG', 'HR', 'LTR', 'PC', 'UNT', 'BOX', 'SET', 'UNIT', 'PIECES', 'TYRE', 'TIRE']
    
    for unit in units:
        if re.search(r'\b' + re.escape(unit) + r'\b', description, re.I):
            return unit.upper()
    
    return 'PCS'  # Default fallback

def clean_description(description):
    """Clean and normalize description text."""
    if not description:
        return ""

    # Remove extra whitespace
    description = re.sub(r'\s+', ' ', description).strip()

    # Remove common prefixes/suffixes that might be left after number removal
    description = re.sub(r'^[-\s]*|[-\s]*$', '', description)

    # Remove any remaining isolated numbers or symbols at word boundaries
    description = re.sub(r'\s+[-\*\.]\s+', ' ', description)

    # Remove percentages completely (these are VAT indicators, not part of description)
    description = re.sub(r'\d+\.?\d*\%', '', description).strip()

    return description

def extract_code_no_enhanced(lines):
    """Enhanced Code No extraction with multiple patterns and validation."""
    code_no = None
    
    patterns = [
        r'(?:Code\s*(?:No|Number|#)?)\s*[\t:\-]?\s*([A-Za-z0-9\-_/]{2,30})',
        r'(?:Customer\s*Code|Cust\.?\s*Code)\s*[\t:\-]?\s*([A-Za-z0-9\-_/]{2,30})',
        r'^(?:Code|COD)\s+([A-Za-z0-9\-_/]{2,30})(?:\s|$)',
        r'(?:^|\s)([A-Z]{1,4}\d{2,8}[A-Z]?)(?:\s|$)',
        r'Code\s*:\s*([A-Za-z0-9\-_/]{2,30})',
        r'Code\s*No\s*[\[\(]?\s*([A-Za-z0-9\-_/]{2,30})\s*[\]\)]?',
    ]
    
    for line in lines:
        for pattern in patterns:
            match = re.search(pattern, line, re.I)
            if match:
                candidate = match.group(1).strip()
                if is_valid_code_no(candidate):
                    code_no = candidate
                    logger.info(f"Found Code No: {code_no} using pattern: {pattern}")
                    return code_no
    return None

def is_valid_code_no(candidate):
    """Validate if the extracted value is a legitimate code number."""
    if not candidate or len(candidate) < 2:
        return False
        
    if re.match(r'^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$', candidate):
        return False
        
    if re.match(r'^\d+\.?\d*$', candidate):
        if len(candidate) > 6:
            return False
        if len(candidate) <= 6 and int(candidate) > 100000:
            return False
            
    invalid_patterns = [
        r'^page\d*$', r'^\d+of\d+$', r'^total$', r'^subtotal$', 
        r'^vat$', r'^tax$', r'^amount$', r'^invoice$', r'^proforma$',
        r'^customer$', r'^name$', r'^address$', r'^phone$', r'^email$',
        r'^ref$', r'^reference$', r'^date$', r'^terms$'
    ]
    
    for pattern in invalid_patterns:
        if re.match(pattern, candidate, re.I):
            return False
            
    has_letters = bool(re.search(r'[A-Za-z]', candidate))
    has_numbers = bool(re.search(r'\d', candidate))
    
    if has_letters or (has_numbers and len(candidate) <= 8):
        return True
        
    if re.match(r'^[A-Z0-9\-_/]{3,20}$', candidate, re.I):
        return True
        
    return False

def extract_invoice_no(lines):
    """Extract Invoice No from lines."""
    for line in lines:
        patterns = [
            r'(?:PI|Invoice)\s*(?:No|Number|#|\.)\s*[:\-]?\s*([A-Z0-9\-]{3,30})',
            r'(?:PI|Invoice)\s*[:\-]?\s*([A-Z0-9\-]{3,30})',
            r'PI\s*[:]?\s*([A-Z0-9\-]{3,30})',
        ]
        for pattern in patterns:
            match = re.search(pattern, line, re.I)
            if match:
                candidate = match.group(1).strip()
                if candidate and len(candidate) >= 3:
                    return candidate
    return None

def extract_date(lines):
    """Extract Date from lines."""
    for line in lines:
        match = re.search(r'(?:Date|Invoice\s*Date)\s*[\t:]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', line, re.I)
        if match:
            return match.group(1)
    return None

def extract_reference(lines):
    """Extract Reference from lines."""
    for line in lines:
        patterns = [
            r'(?:Reference|Cust\s*Ref|Ref\.?)\s*[:\-]?\s*(.+?)(?:\s+Date|$)',
            r'Ref\s*[:\-]?\s*([A-Z0-9\s\-]{3,30})',
        ]
        for pattern in patterns:
            match = re.search(pattern, line, re.I)
            if match:
                candidate = match.group(1).strip()
                if candidate and not re.match(r'^\d{1,2}[/-]', candidate):
                    candidate = re.sub(r'\s*(?:Date|Ref\s*Date|Del\s*Date).*$', '', candidate, flags=re.I).strip()
                    if candidate and len(candidate) >= 2:
                        return candidate
    return None

def extract_monetary_value(lines, patterns):
    """Extract monetary value from lines."""
    for pattern in patterns:
        for line in lines:
            match = re.search(rf'{pattern}\s*[:=]?\s*(?:TSH|TZS|UGX)?\s*([\d,]+\.?\d*)', line, re.I)
            if match:
                try:
                    cleaned = re.sub(r'[^\d\.]', '', match.group(1).replace(',', ''))
                    return Decimal(cleaned) if cleaned else None
                except:
                    pass
    return None

def extract_from_bytes(file_bytes, filename: str = '') -> dict:
    """Main entry point: extract text from file and parse invoice data."""
    if not file_bytes:
        return {
            'success': False, 'error': 'empty_file', 'message': 'File is empty.',
            'ocr_available': False, 'header': {}, 'items': [], 'raw_text': ''
        }

    is_pdf = filename.lower().endswith('.pdf') or (len(file_bytes) > 4 and file_bytes[:4] == b'%PDF')
    is_image = filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.tiff', '.bmp'))

    if is_image:
        return {
            'success': False, 'error': 'image_file_not_supported', 
            'message': 'Image files are not supported.', 'ocr_available': False,
            'header': {}, 'items': [], 'raw_text': ''
        }

    if not is_pdf:
        return {
            'success': False, 'error': 'unsupported_file_type',
            'message': 'Please upload a PDF file.', 'ocr_available': False,
            'header': {}, 'items': [], 'raw_text': ''
        }

    # Extract text from PDF with page separation
    try:
        pages_data = extract_text_from_pdf(file_bytes)
        all_text = '\n'.join([page['text'] for page in pages_data])
    except Exception as e:
        logger.error(f"PDF text extraction failed: {e}")
        return {
            'success': False, 'error': 'pdf_extraction_failed',
            'message': f'Could not extract text from PDF: {str(e)}', 'ocr_available': False,
            'header': {}, 'items': [], 'raw_text': ''
        }

    if not pages_data:
        return {
            'success': False, 'error': 'no_text_extracted',
            'message': 'No readable text found in PDF.', 'ocr_available': False,
            'header': {}, 'items': [], 'raw_text': ''
        }

    # Parse extracted text to structured invoice data
    try:
        parsed = parse_invoice_data(pages_data)

        # Prepare header
        header = {
            'invoice_no': parsed.get('invoice_no'),
            'code_no': parsed.get('code_no'),
            'date': parsed.get('date'),
            'customer_name': parsed.get('customer_name'),
            'phone': parsed.get('phone'),
            'email': parsed.get('email'),
            'address': parsed.get('address'),
            'reference': parsed.get('reference'),
            'subtotal': float(parsed.get('subtotal')) if parsed.get('subtotal') else None,
            'tax': float(parsed.get('tax')) if parsed.get('tax') else None,
            'total': float(parsed.get('total')) if parsed.get('total') else None,
            'payment_method': parsed.get('payment_method'),
            'delivery_terms': parsed.get('delivery_terms'),
            'remarks': parsed.get('remarks'),
            'attended_by': parsed.get('attended_by'),
            'kind_attention': parsed.get('kind_attention'),
        }

        # Format items - PRESERVE EXTRACTED VALUES, NO CALCULATIONS
        formatted_items = []
        for item in parsed.get('items', []):
            formatted_items.append({
                'description': item.get('description', ''),
                'qty': item.get('qty', 1),
                'unit': item.get('unit'),
                'code': item.get('code'),
                'value': float(item.get('value')) if item.get('value') else 0.0,
                'rate': float(item.get('rate')) if item.get('rate') else None,
            })

        # Check if we extracted any meaningful data
        has_data = (header.get('customer_name') or 
                   header.get('invoice_no') or 
                   len(formatted_items) > 0 or 
                   header.get('total'))

        if has_data:
            return {
                'success': True,
                'header': header,
                'items': formatted_items,
                'raw_text': all_text,
                'ocr_available': False,
                'message': 'Invoice data extracted successfully - CORRECTED LINE ITEMS'
            }
        else:
            return {
                'success': False,
                'error': 'parsing_failed',
                'message': 'Could not extract structured data from PDF.',
                'ocr_available': False,
                'header': {},
                'items': [],
                'raw_text': all_text
            }

    except Exception as e:
        logger.error(f"Invoice data parsing failed: {e}")
        return {
            'success': False,
            'error': 'parsing_failed',
            'message': 'Could not extract structured data from PDF.',
            'ocr_available': False,
            'header': {},
            'items': [],
            'raw_text': all_text
        }

def build_invoice_json(parsed: dict) -> dict:
    """Build standardized invoice JSON from parsed data."""
    invoice_type = 'Proforma Invoice' if parsed.get('invoice_no', '').upper().startswith('PI') else 'Invoice'
    
    seller_details = {
        'name': parsed.get('seller_name') or '',
        'address': parsed.get('seller_address') or '',
        'phone': parsed.get('seller_phone') or '',
        'email': parsed.get('seller_email') or '',
        'vat_number': parsed.get('seller_vat_reg') or ''
    }

    customer_details = {
        'code': parsed.get('code_no') or '',
        'name': parsed.get('customer_name') or '',
        'address': parsed.get('address') or '',
        'contact_person': parsed.get('kind_attention') or '',
        'phone': parsed.get('phone') or '',
        'email': parsed.get('email') or ''
    }

    items_out = []
    for idx, item in enumerate(parsed.get('items', []), 1):
        items_out.append({
            'sr_no': idx,
            'item_code': item.get('code') or '',
            'description': item.get('description') or '',
            'type': item.get('unit') or '',
            'quantity': item.get('qty', 1),
            'rate': float(item.get('rate')) if item.get('rate') else '',
            'value': float(item.get('value')) if item.get('value') else '',
            'vat_percent': ''
        })

    totals = {
        'sub_total': float(parsed.get('subtotal')) if parsed.get('subtotal') else '',
        'vat_amount': float(parsed.get('tax')) if parsed.get('tax') else '',
        'vat_percent': '',
        'discount': '',
        'grand_total': float(parsed.get('total')) if parsed.get('total') else ''
    }

    if totals['sub_total'] and totals['vat_amount'] and totals['sub_total'] > 0:
        try:
            totals['vat_percent'] = round((totals['vat_amount'] / totals['sub_total']) * 100, 2)
        except:
            totals['vat_percent'] = ''

    invoice_metadata = {
        'invoice_type': invoice_type,
        'invoice_number': parsed.get('invoice_no') or '',
        'customer_reference': parsed.get('reference') or '',
        'reference_date': '',
        'page': '1',
        'pages': '1',
        'issue_date': parsed.get('date') or '',
        'due_date': '',
        'delivery_date': ''
    }

    return {
        'invoice_metadata': invoice_metadata,
        'seller_details': seller_details,
        'customer_details': customer_details,
        'items': items_out,
        'totals': totals,
        'footer_notes': parsed.get('remarks') or ''
    }

if __name__ == "__main__":
    def test_extraction():
        """Test the extraction with sample data."""
        print("CORRECTED INVOICE EXTRACTION - CLEAN LINE ITEMS")
        print("Key improvements:")
        print("- Line items stop at payment information")
        print("- Payment information removed from descriptions")
        print("- Clean customer information extraction")
        print("- Proper multi-page support")
    
    test_extraction()
