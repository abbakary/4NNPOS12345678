"""
PDF and image text extraction for invoice processing.
Uses PyMuPDF (fitz) for PDF text extraction with fallback patterns.
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

def extract_text_from_pdf(file_bytes) -> str:
    """Extract text from PDF file using PyMuPDF or PyPDF2."""
    text = ""
    fitz_error = None
    pdf2_error = None

    if fitz is not None:
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page in doc:
                page_text = page.get_text("text", sort=True)
                if page_text:
                    text += page_text + "\n"
            doc.close()

            if text and text.strip():
                logger.info(f"Successfully extracted {len(text)} characters from PDF using PyMuPDF")
                return text
            else:
                logger.warning("PyMuPDF extracted empty text from PDF")
                fitz_error = "No text found in PDF (PyMuPDF)"
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {e}")
            fitz_error = str(e)
            text = ""

    if PyPDF2 is not None and not text.strip():
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            if len(pdf_reader.pages) == 0:
                pdf2_error = "PDF has no pages"
            else:
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

                if text and text.strip():
                    logger.info(f"Successfully extracted {len(text)} characters from PDF using PyPDF2")
                    return text
                else:
                    logger.warning("PyPDF2 extracted empty text from PDF")
                    pdf2_error = "No text found in PDF (PyPDF2)"
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {e}")
            pdf2_error = str(e)

    if not fitz and not PyPDF2:
        error_msg = 'No PDF extraction library available. Install PyMuPDF or PyPDF2.'
    elif fitz_error and pdf2_error:
        error_msg = f'PDF extraction failed - PyMuPDF: {fitz_error}. PyPDF2: {pdf2_error}'
    elif fitz_error:
        error_msg = fitz_error
    else:
        error_msg = pdf2_error or 'Unknown PDF extraction error'

    raise RuntimeError(error_msg)

def extract_text_from_image(file_bytes) -> str:
    """Extract text from image file."""
    logger.info("Image file detected. OCR not available. Manual entry required.")
    return ""

def parse_invoice_data(text: str) -> dict:
    """Parse invoice data from extracted text using pattern matching."""
    if not text or not text.strip():
        return {
            'invoice_no': None, 'code_no': None, 'date': None, 'customer_name': None,
            'address': None, 'phone': None, 'email': None, 'reference': None,
            'subtotal': None, 'tax': None, 'total': None, 'items': [],
            'payment_method': None, 'delivery_terms': None, 'remarks': None,
            'attended_by': None, 'kind_attention': None
        }

    normalized_text = text.replace('\r\n', '\n').replace('\r', '\n')
    lines = [line.strip() for line in normalized_text.split('\n') if line.strip()]

    # Find the "Proforma Invoice" marker
    proforma_idx = -1
    for i, line in enumerate(lines):
        if re.search(r'Proforma\s+Invoice|PI\s*No|Code\s*No', line, re.I):
            proforma_idx = i
            break
    if proforma_idx == -1:
        proforma_idx = 0

    extraction_lines = lines[proforma_idx:] if proforma_idx >= 0 else lines

    # Extract basic fields
    code_no = extract_code_no_enhanced(extraction_lines)
    invoice_no = extract_invoice_no(extraction_lines)
    customer_name = extract_customer_name(extraction_lines)
    date_str = extract_date(extraction_lines)
    address = extract_address(extraction_lines)
    phone = extract_phone(extraction_lines)
    email = extract_email(extraction_lines)
    reference = extract_reference(extraction_lines)

    # Extract monetary values
    subtotal = extract_monetary_value(extraction_lines, [r'Net\s*Value', r'Subtotal', r'Net\s*Amount'])
    tax = extract_monetary_value(extraction_lines, [r'VAT', r'Tax', r'GST'])
    total = extract_monetary_value(extraction_lines, [r'Gross\s*Value', r'Grand\s*Total', r'Total\s*Amount'])

    # Extract line items - IMPROVED VERSION
    items = extract_line_items_corrected(extraction_lines)

    return {
        'invoice_no': invoice_no, 'code_no': code_no, 'date': date_str,
        'customer_name': customer_name, 'phone': phone, 'email': email,
        'address': address, 'reference': reference, 'subtotal': subtotal,
        'tax': tax, 'total': total, 'items': items, 'payment_method': None,
        'delivery_terms': None, 'remarks': None, 'attended_by': None,
        'kind_attention': None, 'seller_name': None, 'seller_address': None,
        'seller_phone': None, 'seller_email': None, 'seller_tax_id': None,
        'seller_vat_reg': None
    }

def extract_code_no_enhanced(lines):
    """Enhanced Code No extraction with multiple patterns and validation."""
    code_no = None
    
    # Multiple patterns to match different Code No formats
    patterns = [
        # Pattern 1: Explicit "Code No" with various separators
        r'(?:Code\s*(?:No|Number|#)?)\s*[\t:\-]?\s*([A-Za-z0-9\-_/]{2,30})',
        # Pattern 2: Customer Code patterns
        r'(?:Customer\s*Code|Cust\.?\s*Code)\s*[\t:\-]?\s*([A-Za-z0-9\-_/]{2,30})',
        # Pattern 3: Code at start of line followed by value
        r'^(?:Code|COD)\s+([A-Za-z0-9\-_/]{2,30})(?:\s|$)',
        # Pattern 4: Alphanumeric codes in specific positions
        r'(?:^|\s)([A-Z]{1,4}\d{2,8}[A-Z]?)(?:\s|$)',
        # Pattern 5: Simple code pattern after label
        r'Code\s*:\s*([A-Za-z0-9\-_/]{2,30})',
        # Pattern 6: Code No with parentheses or brackets
        r'Code\s*No\s*[\[\(]?\s*([A-Za-z0-9\-_/]{2,30})\s*[\]\)]?',
    ]
    
    for line in lines:
        for pattern in patterns:
            match = re.search(pattern, line, re.I)
            if match:
                candidate = match.group(1).strip()
                
                # Enhanced validation
                if is_valid_code_no(candidate):
                    code_no = candidate
                    logger.info(f"Found Code No: {code_no} using pattern: {pattern}")
                    return code_no
    
    # Fallback: Check in customer details section
    if not code_no:
        code_no = extract_code_no_from_customer_section(lines)
    
    return code_no

def is_valid_code_no(candidate):
    """Validate if the extracted value is a legitimate code number."""
    if not candidate or len(candidate) < 2:
        return False
        
    # Exclude dates
    if re.match(r'^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$', candidate):
        return False
        
    # Exclude pure numbers that look like amounts or page numbers
    if re.match(r'^\d+\.?\d*$', candidate):
        if len(candidate) > 6:  # Too long for a code, probably amount or phone
            return False
        if len(candidate) <= 6 and int(candidate) > 100000:  # Likely amount
            return False
            
    # Exclude common labels and false positives
    invalid_patterns = [
        r'^page\d*$', r'^\d+of\d+$', r'^total$', r'^subtotal$', 
        r'^vat$', r'^tax$', r'^amount$', r'^invoice$', r'^proforma$',
        r'^customer$', r'^name$', r'^address$', r'^phone$', r'^email$',
        r'^ref$', r'^reference$', r'^date$', r'^terms$'
    ]
    
    for pattern in invalid_patterns:
        if re.match(pattern, candidate, re.I):
            return False
            
    # Valid codes typically have mix of letters/numbers or specific formats
    has_letters = bool(re.search(r'[A-Za-z]', candidate))
    has_numbers = bool(re.search(r'\d', candidate))
    
    # Accept codes with letters, or numeric codes of reasonable length
    if has_letters or (has_numbers and len(candidate) <= 8):
        return True
        
    # Also accept codes that look like product codes (alphanumeric with dashes)
    if re.match(r'^[A-Z0-9\-_/]{3,20}$', candidate, re.I):
        return True
        
    return False

def extract_code_no_from_customer_section(lines):
    """Extract Code No from customer information section."""
    in_customer_section = False
    customer_section_lines = []
    
    for line in lines:
        # Detect customer section
        if re.search(r'customer|client|cust\.?', line, re.I) and not re.search(r'seller|vendor', line, re.I):
            in_customer_section = True
            continue
            
        if in_customer_section:
            # Stop when we hit other sections
            if re.search(r'description|items|qty|rate|amount|invoice|proforma', line, re.I):
                break
                
            customer_section_lines.append(line)
            
            # Look for code patterns in customer section
            code_patterns = [
                r'(?:code|ref)\s*[:#]?\s*([A-Z0-9\-_/]{3,20})',
                r'^(?:code|ref)\s+([A-Z0-9\-_/]{3,20})',
                r'([A-Z]{2,4}\d{3,8})',  # Pattern like AB12345
            ]
            
            for pattern in code_patterns:
                match = re.search(pattern, line, re.I)
                if match:
                    candidate = match.group(1).strip()
                    if is_valid_code_no(candidate):
                        return candidate
                    
    return None

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

def extract_customer_name(lines):
    """Extract Customer Name from lines."""
    for i, line in enumerate(lines):
        if re.search(r'Customer\s*Name', line, re.I):
            match = re.search(r'Customer\s*Name\s*[\t:]?\s*(.+?)(?:\s+Date|$)', line, re.I)
            if match:
                name = match.group(1).strip()
                if name and not re.match(r'^\d{1,2}[/-]', name):
                    return name
            elif i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and not re.match(r'^(?:Tel|Fax|Email|Phone|Address|Date)', next_line, re.I):
                    return next_line
    return None

def extract_date(lines):
    """Extract Date from lines."""
    for line in lines:
        match = re.search(r'(?:Date|Invoice\s*Date)\s*[\t:]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', line, re.I)
        if match:
            return match.group(1)
    return None

def extract_address(lines):
    """Extract Address from lines."""
    for i, line in enumerate(lines):
        if re.search(r'(?:P\.?O\.?\s*B(?:OX)?|Address)', line, re.I):
            match = re.search(r'(?:P\.?O\.?\s*B(?:OX)?|Address)\s*[\t:]?\s*(.+?)$', line, re.I)
            if match:
                addr_line = match.group(1).strip()
                address_parts = [addr_line] if addr_line else []
                for j in range(i + 1, min(i + 6, len(lines))):
                    next_line = lines[j].strip()
                    if re.match(r'^(?:Tel|Fax|Email|Phone|Cust|Ref|Date|Del|Kind)', next_line, re.I):
                        break
                    if not re.match(r'^\d{1,2}[/-]', next_line):
                        cleaned_line = re.sub(r'\s*(?:Cust\s*Ref|Ref\s*Date|Del\.?\s*Date)\s*:.*$', '', next_line, flags=re.I).strip()
                        if cleaned_line:
                            address_parts.append(cleaned_line)
                
                address = ' '.join(filter(None, address_parts))
                address = re.sub(r'\s+(?:Cust\s*Ref|Ref\s*Date|Del\.?\s*Date)\s*:.*?(?=\s+[A-Z]|$)', '', address, flags=re.I)
                address = re.sub(r'\s+', ' ', address).strip()
                if address:
                    return address
    return None

def extract_phone(lines):
    """Extract Phone from lines."""
    for line in lines:
        tel_match = re.search(r'(?:Tel|Telephone|Phone)\s*[\t:]?\s*([\+\d][\d\s\-/\(\)\.\,]{5,})', line, re.I)
        if tel_match:
            phone = tel_match.group(1).strip()
            if phone and len(phone) >= 7:
                return phone
    return None

def extract_email(lines):
    """Extract Email from lines."""
    for line in lines:
        email_match = re.search(r'([\w\.-]+@[\w\.-]+\.\w+)', line)
        if email_match:
            return email_match.group(1)
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

def extract_line_items_corrected(lines):
    """
    CORRECTED line item extraction with proper column separation.
    """
    items = []
    
    # Find the item table section
    table_start = -1
    for i, line in enumerate(lines):
        # Comprehensive header detection
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
        if keyword_count >= 4:
            table_start = i
            logger.info(f"Found item table header at line {i}: {line}")
            break
    
    if table_start == -1:
        logger.warning("No item table header found")
        return items
    
    # Process lines after header
    i = table_start + 1
    current_item_lines = []
    current_item_number = None
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Stop at totals or section breaks
        if re.search(r'(Net\s*Value|Gross\s*Value|Grand\s*Total|Total\s*Amount|Sub.*Total|Page\s*\d+|Customer\s*Information|Thank|Notes?)', line, re.I):
            logger.info(f"Stopping at section break: {line}")
            break
        
        # Skip empty lines
        if not line:
            i += 1
            continue
        
        # Check if this line starts a new item (starts with a number followed by item code pattern)
        item_start_match = re.match(r'^(\d+)\.?\s+(\d{4,15})\s+(.+)$', line)
        if item_start_match:
            # If we have a current item being built, save it first
            if current_item_lines and current_item_number is not None:
                parsed_item = parse_item_complete(current_item_lines, current_item_number)
                if parsed_item and parsed_item.get('description'):
                    items.append(parsed_item)
                    logger.info(f"Completed item {len(items)}: {parsed_item}")
            
            # Start new item
            current_item_number = item_start_match.group(1)
            item_code = item_start_match.group(2)
            rest_of_line = item_start_match.group(3)
            current_item_lines = [{'code': item_code, 'line': rest_of_line}]
            
        elif re.match(r'^(\d+)\.?\s+(.+)$', line) and not re.match(r'^(\d+)\.?\s+(\d{4,15})\s+', line):
            # This line starts with a number but doesn't have an item code pattern
            item_simple_match = re.match(r'^(\d+)\.?\s+(.+)$', line)
            if item_simple_match:
                # If we have a current item being built, save it first
                if current_item_lines and current_item_number is not None:
                    parsed_item = parse_item_complete(current_item_lines, current_item_number)
                    if parsed_item and parsed_item.get('description'):
                        items.append(parsed_item)
                        logger.info(f"Completed item {len(items)}: {parsed_item}")
                
                # Start new item without predefined code
                current_item_number = item_simple_match.group(1)
                rest_of_line = item_simple_match.group(2)
                current_item_lines = [{'code': None, 'line': rest_of_line}]
        
        elif current_item_lines:
            # This is a continuation line for the current item
            current_item_lines.append({'code': None, 'line': line})
        
        i += 1
    
    # Don't forget the last item
    if current_item_lines and current_item_number is not None:
        parsed_item = parse_item_complete(current_item_lines, current_item_number)
        if parsed_item and parsed_item.get('description'):
            items.append(parsed_item)
            logger.info(f"Completed final item {len(items)}: {parsed_item}")
    
    logger.info(f"Total items extracted: {len(items)}")
    return items

def parse_item_complete(item_lines, item_number):
    """
    Parse a complete item with proper column separation and value calculation.
    Handles cases where VAT% appears after rate value.
    """
    if not item_lines:
        return None

    # Extract item code from first line if available
    item_code = item_lines[0]['code']

    # Combine all lines into one text block
    full_text = ' '.join([line['line'] for line in item_lines])

    # Try to extract item code from the text if not already found
    if not item_code:
        code_match = re.search(r'\b(\d{4,15})\b', full_text)
        if code_match:
            item_code = code_match.group(1)
            logger.info(f"Extracted item code from text: {item_code}")

    # CRITICAL: Extract values exactly as shown in the document - NO CALCULATIONS
    # This preserves the actual invoice data without risk of recalculation errors

    # Remove VAT percentages temporarily for pattern matching, but keep track of them
    cleaned_text = re.sub(r'\s*\d+\.?\d*%\s*', ' ', full_text).strip()

    # Primary Pattern: Description Unit Qty Rate Value (most complete)
    # Matches: "DESCRIPTION UNIT QTY RATE VALUE"
    pattern_complete = r'^(.+?)\s+(PCS|NOS|KG|HR|LTR|PC|UNT|BOX|SET|UNIT|PIECES|TYRE|TIRE)\s+(\d+)\s+([\d,]+\.?\d{1,2})\s+([\d,]+\.?\d{1,2})$'
    match_complete = re.search(pattern_complete, cleaned_text)

    if match_complete:
        description = match_complete.group(1).strip()
        unit = match_complete.group(2).upper()
        qty = int(match_complete.group(3))
        rate = Decimal(match_complete.group(4).replace(',', ''))
        value = Decimal(match_complete.group(5).replace(',', ''))

        # Remove item code from description if present
        if item_code and item_code in description:
            description = description.replace(item_code, '', 1).strip()

        logger.info(f"Complete pattern match - Code: {item_code}, Desc: {description[:50]}, Unit: {unit}, Qty: {qty}, Rate: {rate}, Value: {value} (extracted as-is)")

        return {
            'code': item_code,
            'description': clean_description(description),
            'unit': unit,
            'qty': qty,
            'rate': rate,
            'value': value
        }

    # Secondary Pattern: Description Unit Qty Rate (no Value found - extract what's available)
    # In this case, extract only what we have - do NOT calculate value
    pattern_with_unit = r'^(.+?)\s+(PCS|NOS|KG|HR|LTR|PC|UNT|BOX|SET|UNIT|PIECES|TYRE|TIRE)\s+(\d+)\s+([\d,]+\.?\d{1,2})'
    match_with_unit = re.search(pattern_with_unit, cleaned_text)

    if match_with_unit:
        description = match_with_unit.group(1).strip()
        unit = match_with_unit.group(2).upper()
        qty = int(match_with_unit.group(3))
        rate = Decimal(match_with_unit.group(4).replace(',', ''))

        # Try to find Value separately from the remaining text
        # Look for another large number after the rate
        remaining = cleaned_text[match_with_unit.end():].strip()
        value = None

        # Try to extract value from the remaining text (should be the next large number)
        if remaining:
            value_match = re.search(r'([\d,]+\.?\d{1,2})', remaining)
            if value_match:
                value = Decimal(value_match.group(1).replace(',', ''))
                logger.info(f"Unit pattern match + extracted Value separately - Code: {item_code}, Desc: {description[:50]}, Unit: {unit}, Qty: {qty}, Rate: {rate}, Value: {value}")

        # If value couldn't be extracted separately, use rate (as a last resort, not calculated)
        if value is None:
            value = rate
            logger.info(f"Unit pattern match (no Value found) - Code: {item_code}, Desc: {description[:50]}, Unit: {unit}, Qty: {qty}, Rate: {rate}, Value: {rate} (no calculation)")

        # Remove item code from description if present
        if item_code and item_code in description:
            description = description.replace(item_code, '', 1).strip()

        return {
            'code': item_code,
            'description': clean_description(description),
            'unit': unit,
            'qty': qty,
            'rate': rate,
            'value': value
        }
    
    # Pattern for basic quantity and rate - extract without calculation
    pattern_basic = r'(\d+)\s+([\d,]+\.?\d{2})'
    matches_basic = list(re.finditer(pattern_basic, full_text))

    if len(matches_basic) >= 3:
        # Try to extract: qty, rate, value (in that order) without calculation
        qty = int(matches_basic[0].group(1))
        rate = Decimal(matches_basic[1].group(2).replace(',', ''))
        value = Decimal(matches_basic[2].group(2).replace(',', ''))
        logger.info(f"Basic pattern match with Value - Code: {item_code}, Qty: {qty}, Rate: {rate}, Value: {value} (extracted as-is)")
    elif len(matches_basic) >= 2:
        # Only qty and rate found - don't calculate value
        qty = int(matches_basic[0].group(1))
        rate = Decimal(matches_basic[1].group(2).replace(',', ''))
        value = rate  # Use rate as fallback, not calculation
        logger.info(f"Basic pattern match (no Value found) - Code: {item_code}, Qty: {qty}, Rate: {rate}, Value: {value} (no calculation)")
        
        # Extract unit
        unit = None
        unit_match = re.search(r'\b(PCS|NOS|KG|HR|LTR|PC|UNT|BOX|SET|UNIT|PIECES|TYRE|TIRE)\b', full_text, re.I)
        if unit_match:
            unit = unit_match.group(1).upper()
        
        # Build description by removing numbers, units, and codes
        description = full_text
        
        # Remove item code
        if item_code and item_code in description:
            description = description.replace(item_code, '', 1).strip()
        
        # Remove quantities and rates
        for match in reversed(matches_basic):
            description = description.replace(match.group(0), '', 1).strip()
        
        # Remove unit
        if unit:
            description = re.sub(r'\b' + re.escape(unit) + r'\b', '', description, flags=re.I).strip()
        
        # Remove percentages and other noise
        description = re.sub(r'\d+\.?\d*\%', '', description).strip()
        
        logger.info(f"Basic pattern match - Code: {item_code}, Desc: {description[:50]}, Unit: {unit}, Qty: {qty}, Rate: {rate}, Value: {value}")
        
        return {
            'code': item_code,
            'description': clean_description(description),
            'unit': unit,
            'qty': qty,
            'rate': rate,
            'value': value
        }
    
    # Fallback: Extract numbers and try to make sense of them
    # Use cleaned_text that has VAT% removed
    numbers = []
    number_matches = re.finditer(r'([\d,]+\.?\d*)', cleaned_text)
    for match in number_matches:
        try:
            num_str = match.group(1).replace(',', '')
            num = float(num_str)
            numbers.append({
                'value': num,
                'original': match.group(1),
                'is_integer': num == int(num) and num < 10000
            })
        except:
            continue
    
    # Extract unit
    unit = None
    unit_match = re.search(r'\b(PCS|NOS|KG|HR|LTR|PC|UNT|BOX|SET|UNIT|PIECES|TYRE|TIRE)\b', cleaned_text, re.I)
    if unit_match:
        unit = unit_match.group(1).upper()

    # Identify quantities and values
    # Only consider integers less than 100 as potential quantities (avoid product codes like 2132004135)
    quantities = [n for n in numbers if n['is_integer'] and 0 < n['value'] < 100]
    potential_rates = [n for n in numbers if not n['is_integer'] or (n['value'] >= 100 and n['value'] < 1000000)]

    # Determine quantity and rate
    quantity = 1
    rate = None
    value = None

    if quantities and potential_rates:
        # Use smallest integer (but > 0) as quantity
        quantity = int(min([q['value'] for q in quantities if q['value'] > 0]))
        largest_value = max([r['value'] for r in potential_rates])

        # Check if largest value makes sense as total value (qty * rate should match)
        if largest_value > 10000:  # Likely a total value (large number)
            value = Decimal(str(largest_value))
            rate = value / Decimal(quantity) if quantity > 0 else Decimal('0')
        else:
            rate = Decimal(str(largest_value))
            value = rate * Decimal(quantity) if quantity > 0 else Decimal('0')
    elif quantities:
        # Only quantities found, no rate
        quantity = int(min([q['value'] for q in quantities if q['value'] > 0]))
        rate = Decimal('0')
        value = Decimal('0')
    elif potential_rates:
        # Only rates found, assume quantity 1
        rate = Decimal(str(max([r['value'] for r in potential_rates])))
        value = rate
    else:
        # No numbers found
        rate = Decimal('0')
        value = Decimal('0')
    
    # Build description
    description = full_text
    
    # Remove item code
    if item_code and item_code in description:
        description = description.replace(item_code, '', 1).strip()
    
    # Remove numbers
    for num_info in sorted(numbers, key=lambda x: len(x['original']), reverse=True):
        description = description.replace(num_info['original'], '', 1).strip()
    
    # Remove unit
    if unit:
        description = re.sub(r'\b' + re.escape(unit) + r'\b', '', description, flags=re.I).strip()
    
    # Remove percentages
    description = re.sub(r'\d+\.?\d*\%', '', description).strip()
    
    if not description or len(description) < 2:
        return None
    
    logger.info(f"Fallback extraction - Code: {item_code}, Desc: {description[:50]}, Unit: {unit}, Qty: {quantity}, Rate: {rate}, Value: {value}")
    
    return {
        'code': item_code,
        'description': clean_description(description),
        'unit': unit,
        'qty': quantity,
        'rate': rate,
        'value': value
    }

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

    # Don't remove noise words - they may be legitimate parts of product descriptions
    # e.g., "WHEEL BALANCE ALLOYD RIMS" or "VALVE FOR CAR TUBELESS TYRES"

    return description

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

    # Extract text from PDF
    try:
        text = extract_text_from_pdf(file_bytes)
    except Exception as e:
        logger.error(f"PDF text extraction failed: {e}")
        return {
            'success': False, 'error': 'pdf_extraction_failed',
            'message': f'Could not extract text from PDF: {str(e)}', 'ocr_available': False,
            'header': {}, 'items': [], 'raw_text': ''
        }

    if not text or not text.strip():
        return {
            'success': False, 'error': 'no_text_extracted',
            'message': 'No readable text found in PDF.', 'ocr_available': False,
            'header': {}, 'items': [], 'raw_text': ''
        }

    # Parse extracted text to structured invoice data
    try:
        parsed = parse_invoice_data(text)

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

        # Format items
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
                'raw_text': text,
                'ocr_available': False,
                'message': 'Invoice data extracted successfully'
            }
        else:
            return {
                'success': False,
                'error': 'parsing_failed',
                'message': 'Could not extract structured data from PDF.',
                'ocr_available': False,
                'header': {},
                'items': [],
                'raw_text': text
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
            'raw_text': text
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

# Example usage and testing
if __name__ == "__main__":
    def test_extraction():
        """Test the extraction with sample data."""
        test_text = """
        Proforma Invoice
        PI No: 1765632
        Code No: A01696
        Date: 25/10/2025
        Customer Name: STATEOIL TANZANIA LIMITED
        
        P.O. BOX 15950 DAR ES SALAAM TANZANIA
        
        Sr  Item Code  Description                                                             Type  Qty  Rate (TSH)   Value (TSH)
        1   2132004135 BF GOODRICH TYRE LT265/65R17 116/113S TL ALL-TERRAIN T/A KO3 LRD RWL GO PCS   4    1,037,400.00 4,149,600.00
        2   3373119002 VALVE (1214 TR 414) FOR CAR TUBELESS TYRES                              PCS   4    1,300.00     5,200.00
        3   21004      WHEEL BALANCE ALLOY RIMS                                                PCS   4    12,712.00    50,848.00
        4   21019      WHEEL ALIGNMENT SMALL                                                   UNT   1    50,848.00    50,848.00
        
        Net Value: 4,256,496.00
        VAT: 765,169.28
        Grand Total: 5,021,665.28
        """
        
        lines = [line.strip() for line in test_text.split('\n') if line.strip()]
        parsed = parse_invoice_data(test_text)
        print("Test Data Extraction Results:")
        print(f"Invoice No: {parsed['invoice_no']}")
        print(f"Code No: {parsed['code_no']}")
        print(f"Customer: {parsed['customer_name']}")
        print(f"Total: {parsed['total']}")
        print(f"Items: {len(parsed['items'])}")
        
        for i, item in enumerate(parsed['items'], 1):
            print(f"  {i}. {item['description']}")
            print(f"     Code: {item['code']}, Unit: {item['unit']}, Qty: {item['qty']}, Rate: {item['rate']}, Value: {item['value']}")
    
    test_extraction()
