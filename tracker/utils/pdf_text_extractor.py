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
    """Extract text from PDF file using PyMuPDF or PyPDF2.

    Args:
        file_bytes: Raw bytes of PDF file

    Returns:
        Extracted text string

    Raises:
        RuntimeError: If no PDF extraction library is available or text extraction fails
    """
    text = ""
    fitz_error = None
    pdf2_error = None

    # Try PyMuPDF first (fitz) - best for text extraction
    if fitz is not None:
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page in doc:
                # Use layout preservation for better structure
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

    # Fallback to PyPDF2
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

    # If we get here, extraction failed with both libraries
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
    """Extract text from image file.
    Since OCR is not available, this returns empty string.
    
    Args:
        file_bytes: Raw bytes of image file
        
    Returns:
        Empty string (manual entry required for images)
    """
    logger.info("Image file detected. OCR not available. Manual entry required.")
    return ""

def _extract_items_with_coordinates(file_bytes):
    """Extract items using coordinate-based parsing for table structures."""
    items = []
    try:
        if fitz is None:
            return items
            
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page in doc:
            words = page.get_text("words")
            if not words:
                continue
                
            # Group by line_no for structure
            lines_map = {}
            for w in words:
                x0, y0, x1, y1, text, block_no, line_no, word_no = w
                lines_map.setdefault((block_no, line_no), []).append((x0, y0, x1, y1, text))

            # Sort words in each line left-to-right
            ordered_lines = []
            for key, line_words in lines_map.items():
                line_words.sort(key=lambda t: t[0])
                ordered_lines.append((key, line_words))
                
            # Sort lines by vertical position
            ordered_lines.sort(key=lambda kv: kv[1][0][1])

            # Detect header line by keywords
            header_idx = None
            header_words = None
            for i, (_, lw) in enumerate(ordered_lines):
                joined = ' '.join([t[4] for t in lw])
                kw_hits = 0
                for pat in (r'\bSr\b|\bS\.?\s*No\.?', r'Item\s*Code|Code', r'Description|Desc', r'Qty|Quantity', r'Rate|Unit\s*Price|Price', r'Value|Amount'):
                    if re.search(pat, joined, re.I):
                        kw_hits += 1
                if kw_hits >= 3:
                    header_idx = i
                    header_words = lw
                    break
                    
            if header_idx is None:
                continue

            # Determine column x positions from header keywords
            columns = []  # list of (name, x_start)
            def find_word_x(prefixes):
                for x0, y0, x1, y1, text in header_words:
                    for p in prefixes:
                        if re.search(p, text, re.I):
                            return x0
                return None

            col_defs = [
                ("sr", [r'^sr$', r'^s\.?\s*no\.?$', r'^no\.?$']),
                ("code", [r'item\s*code', r'^code$']),
                ("description", [r'^description$', r'^desc$']),
                ("type", [r'^type$', r'^unit$']),
                ("qty", [r'^qty$', r'^quantity$']),
                ("rate", [r'^rate$', r'unit\s*price', r'^price$']),
                ("value", [r'^value$', r'^amount$', r'^total$']),
            ]
            
            for name, pats in col_defs:
                x = find_word_x(pats)
                if x is not None:
                    columns.append((name, x))
                    
            if not columns:
                continue
                
            columns.sort(key=lambda c: c[1])
            
            # Build column ranges using midpoints
            ranges = []  # list of (name, x_min, x_max)
            for idx, (name, x) in enumerate(columns):
                if idx == 0:
                    left = x - 2
                    right = (columns[idx + 1][1] + x) / 2 if idx + 1 < len(columns) else x + 2000
                elif idx == len(columns) - 1:
                    left = (columns[idx - 1][1] + x) / 2
                    right = x + 2000
                else:
                    left = (columns[idx - 1][1] + x) / 2
                    right = (columns[idx + 1][1] + x) / 2
                ranges.append((name, left, right))

            # Parse item lines until totals section
            header_y = header_words[0][1]
            for i in range(header_idx + 1, len(ordered_lines)):
                _, lw = ordered_lines[i]
                line_y = lw[0][1]
                
                # Stop if we reach summary keywords
                joined = ' '.join([t[4] for t in lw])
                if re.search(r'(Net\s*Value|Gross\s*Value|Grand\s*Total|Total\s*:)', joined, re.I):
                    break

                # Collect cell text by ranges
                cells = {name: [] for name, _, _ in ranges}
                for x0, y0, x1, y1, text in lw:
                    for name, xmin, xmax in ranges:
                        if x0 >= xmin and x0 < xmax:
                            cells[name].append(text)
                            break
                            
                # Build item record
                desc = ' '.join(cells.get('description') or []).strip()
                if not desc:
                    continue

                # Parse code
                code_text = ' '.join(cells.get('code') or []).strip()
                code_match = re.search(r'(\d{3,15})', code_text) if code_text else None
                item_code = code_match.group(1) if code_match else None
        
                # Parse qty
                qty_text = ' '.join(cells.get('qty') or []).replace(',', '').strip()
                qty_val = None
                if qty_text:
                    m = re.search(r'(\d{1,4})', qty_text)
                    if m:
                        try:
                            qty_val = int(m.group(1))
                        except Exception:
                            qty_val = None

                # Parse rate and value
                def to_dec(s):
                    try:
                        return Decimal(re.sub(r'[^\d\.-]', '', s).replace(',', ''))
                    except Exception:
                        return None
                        
                rate_text = ' '.join(cells.get('rate') or []).strip()
                rate_val = to_dec(rate_text) if rate_text else None
        
                value_text = ' '.join(cells.get('value') or []).strip()
                value_val = to_dec(value_text) if value_text else None

                # Parse unit/type
                type_text = ' '.join(cells.get('type') or []).strip()
                unit_val = None
                if type_text:
                    unit_match = re.search(r'\b(NOS|PCS|KG|HR|LTR|UNT|BOX|PCS|PC|NOS)\b', type_text, re.I)
                    unit_val = unit_match.group(1).upper() if unit_match else type_text.upper()

                # Skip non-item rows
                if not desc and not value_val and not rate_val:
                    continue

                # Finalize item
                items.append({
                    'description': desc[:255] if desc else None,
                    'qty': qty_val or 1,
                    'unit': unit_val,
                    'value': value_val,
                    'rate': rate_val,
                    'code': item_code,
                })
                
        doc.close()
        
    except Exception as e:
        logger.warning(f"Coordinate-based extraction failed: {e}")
        
    return items

def parse_invoice_data(text: str) -> dict:
    """Parse invoice data from extracted text using pattern matching.

    Args:
        text: Raw extracted text from PDF/image

    Returns:
        dict with extracted invoice data
    """
    if not text or not text.strip():
        return {
            'invoice_no': None,
            'code_no': None,
            'date': None,
            'customer_name': None,
            'address': None,
            'phone': None,
            'email': None,
            'reference': None,
            'subtotal': None,
            'tax': None,
            'total': None,
            'items': [],
            'payment_method': None,
            'delivery_terms': None,
            'remarks': None,
            'attended_by': None,
            'kind_attention': None
        }

    normalized_text = text.replace('\r\n', '\n').replace('\r', '\n')
    lines = [line.strip() for line in normalized_text.split('\n') if line.strip()]

    # Find the "Proforma Invoice" marker to start extraction from there
    proforma_idx = -1
    for i, line in enumerate(lines):
        if re.search(r'Proforma\s+Invoice|PI\s*No|Code\s*No', line, re.I):
            proforma_idx = i
            break

    # If no Proforma Invoice found, start from beginning but skip first few lines
    if proforma_idx == -1:
        proforma_idx = 0

    # Use only lines from Proforma Invoice marker onwards
    extraction_lines = lines[proforma_idx:] if proforma_idx >= 0 else lines

    # Extract seller information from original top of document (before Proforma)
    seller_name = None
    seller_address = None
    seller_phone = None
    seller_email = None
    seller_tax_id = None
    seller_vat_reg = None

    try:
        top_block = lines[:proforma_idx] if proforma_idx > 0 else lines[:8]

        seller_lines = top_block
        if seller_lines:
            seller_name = seller_lines[0] if seller_lines and seller_lines[0] else None
            if len(seller_lines) > 1:
                seller_address = ' '.join([ln for ln in seller_lines[1:] if ln])

            seller_block_text = '\n'.join(seller_lines)
            phone_match = re.search(r'(?:Tel\.?|Telephone|Phone)[:\s]*([\+\d][\d\s\-/\(\)\,]{4,}\d)', seller_block_text, re.I)
            if phone_match:
                seller_phone = phone_match.group(1).strip()

            email_match = re.search(r'([\w\.-]+@[\w\.-]+\.\w+)', seller_block_text)
            if email_match:
                seller_email = email_match.group(1).strip()

    except Exception:
        pass

    # Helper function to extract field values
    def extract_field_value(label_patterns, max_lines=3):
        patterns = label_patterns if isinstance(label_patterns, list) else [label_patterns]
        
        for pattern in patterns:
            # Look for pattern in text
            for i, line in enumerate(lines):
                if re.search(pattern, line, re.I):
                    # Try to extract value from same line
                    match = re.search(rf'{pattern}\s*[:=]?\s*(.+)', line, re.I)
                    if match:
                        value = match.group(1).strip()
                        if value:
                            return value
                    
                    # Look in next lines
                    for j in range(1, min(max_lines + 1, len(lines) - i)):
                        next_line = lines[i + j].strip()
                        if next_line and not re.match(r'^(?:Tel|Fax|Email|Address|Reference|PI|Date)', next_line, re.I):
                            return next_line
        return None

    # Extract Code No
    code_no = extract_field_value([
        r'Code\s*No',
        r'Code\s*#',
        r'^Code\b'
    ])

    # Extract Invoice No (PI No)
    invoice_no = extract_field_value([
        r'PI\s*No',
        r'Proforma\s*Invoice',
        r'Invoice\s*No',
        r'^PI\b'
    ])

    # Extract Customer Name
    customer_name = extract_field_value([
        r'Customer\s*Name',
        r'Bill\s*To',
        r'Sold\s*To'
    ])

    # Extract Date
    date_str = None
    for line in lines:
        date_match = re.search(r'(?:Date|Invoice\s*Date)\s*[:=]?\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})', line, re.I)
        if date_match:
            date_str = date_match.group(1)
            break

    # Extract Address (look for P.O.BOX patterns)
    address = None
    for i, line in enumerate(lines):
        if re.search(r'P\.?\s*O\.?\s*B|P\.?O\.?\s*BOX|POB', line, re.I):
            address_parts = [line]
            for j in range(i + 1, min(i + 4, len(lines))):
                next_line = lines[j]
                if re.match(r'^(?:Tel|Fax|Email|Phone)', next_line, re.I):
                    break
                address_parts.append(next_line)
            address = ' '.join(address_parts)
            break

    # Extract Phone
    phone = None
    for line in lines:
        tel_match = re.search(r'\bTel\s*[:=]?\s*([^\n]+?)(?:\s*(?:Fax|Email)|$)', line, re.I)
        if tel_match:
            phone = tel_match.group(1).strip()
            break

    # Extract Email
    email = None
    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', normalized_text)
    if email_match:
        email = email_match.group(0)

    # Extract Reference
    reference = extract_field_value([
        r'Reference',
        r'Ref\.?'
    ])

    # Extract monetary values
    def find_amount(label_patterns):
        for pattern in label_patterns:
            for line in lines:
                match = re.search(rf'{pattern}\s*[:=]?\s*(?:TSH|TZS|UGX)?\s*([\d,]+\.?\d*)', line, re.I)
                if match:
                    return match.group(1)
        return None

    def to_decimal(s):
        try:
            if s:
                cleaned = re.sub(r'[^\d\.]', '', s.replace(',', ''))
                return Decimal(cleaned)
        except Exception:
            pass
        return None

    subtotal = to_decimal(find_amount([r'Net\s*Value', r'Subtotal', r'Net\s*Amount']))
    tax = to_decimal(find_amount([r'VAT', r'Tax', r'GST']))
    total = to_decimal(find_amount([r'Gross\s*Value', r'Grand\s*Total', r'Total\s*Amount']))

    # Extract line items
    items = extract_line_items_from_text(normalized_text)

    # Extract additional fields
    payment_method = extract_field_value([r'Payment', r'Payment\s*Method'])
    delivery_terms = extract_field_value([r'Delivery', r'Delivery\s*Terms'])
    remarks = extract_field_value([r'Remarks', r'Notes', r'NOTE'])
    attended_by = extract_field_value([r'Attended\s*By'])
    kind_attention = extract_field_value([r'Kind\s*Attention'])

    return {
        'invoice_no': invoice_no,
        'code_no': code_no,
        'date': date_str,
        'customer_name': customer_name,
        'phone': phone,
        'email': email,
        'address': address,
        'reference': reference,
        'subtotal': subtotal,
        'tax': tax,
        'total': total,
        'items': items,
        'payment_method': payment_method,
        'delivery_terms': delivery_terms,
        'remarks': remarks,
        'attended_by': attended_by,
        'kind_attention': kind_attention,
        'seller_name': seller_name,
        'seller_address': seller_address,
        'seller_phone': seller_phone,
        'seller_email': seller_email,
        'seller_tax_id': seller_tax_id,
        'seller_vat_reg': seller_vat_reg
    }

def extract_line_items_from_text(text):
    """Extract line items from text using pattern matching."""
    items = []
    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    
    # Find item section
    item_section_start = -1
    for i, line in enumerate(cleaned_lines):
        keyword_count = sum([
            1 if re.search(r'\b(?:Sr|S\.N|Serial|No\.?)\b', line, re.I) else 0,
            1 if re.search(r'\b(?:Item|Code|Product)\b', line, re.I) else 0,
            1 if re.search(r'\b(?:Description|Desc)\b', line, re.I) else 0,
            1 if re.search(r'\b(?:Qty|Quantity)\b', line, re.I) else 0,
            1 if re.search(r'\b(?:Rate|Price|Value|Amount)\b', line, re.I) else 0,
        ])
        if keyword_count >= 3:
            item_section_start = i
            break

    if item_section_start == -1:
        return items

    # Process item lines
    for i in range(item_section_start + 1, len(cleaned_lines)):
        line = cleaned_lines[i]
        
        # Stop at totals
        if re.search(r'(Net\s*Value|Gross\s*Value|Grand\s*Total|Total\s*:)', line, re.I):
            break
            
        # Skip header lines
        if re.search(r'(Sr|Item|Code|Description|Qty|Rate|Value)', line, re.I):
            continue

        # Parse item line
        item = parse_item_line(line)
        if item and item.get('description'):
            items.append(item)

    return items

def parse_item_line(line):
    """Parse a single item line."""
    # Remove leading numbers (Sr No)
    line = re.sub(r'^\d{1,3}\s+', '', line).strip()
    
    # Extract item code
    code_match = re.search(r'^(\d{3,10})\s+', line)
    item_code = code_match.group(1) if code_match else None
    if code_match:
        line = line[len(code_match.group(0)):].strip()

    # Extract unit type
    unit_match = re.search(r'\b(PCS|NOS|KG|UNIT|BOX|SET|PC|UNT)\b', line, re.I)
    unit = unit_match.group(1).upper() if unit_match else None

    # Extract numbers (qty, rate, value)
    numbers = []
    number_matches = re.finditer(r'(\d+(?:,\d+)*(?:\.\d+)?)', line)
    for match in number_matches:
        try:
            num = float(match.group(1).replace(',', ''))
            numbers.append(num)
        except ValueError:
            continue

    # Determine description
    description = line
    if unit_match:
        description = description[:unit_match.start()].strip()
    
    # Clean up description
    description = re.sub(r'\s+\d+(?:,\d+)*(?:\.\d+)?\s*$', '', description).strip()
    description = re.sub(r'\s+(?:PCS|NOS|KG|UNIT|BOX|SET)\s*$', '', description, flags=re.I).strip()

    if not description or len(description) < 2:
        return None

    # Build item
    item = {
        'description': description[:255],
        'code': item_code,
        'unit': unit,
        'qty': 1,
        'rate': None,
        'value': None
    }

    # Assign numbers to qty, rate, value
    if len(numbers) == 1:
        item['value'] = Decimal(str(numbers[0]))
    elif len(numbers) == 2:
        # Assume first is qty, second is value
        if numbers[0] == int(numbers[0]) and numbers[0] < 1000:
            item['qty'] = int(numbers[0])
            item['value'] = Decimal(str(numbers[1]))
        else:
            item['value'] = Decimal(str(max(numbers)))
    elif len(numbers) >= 3:
        # Find qty (small integer)
        qty_candidate = None
        max_num = max(numbers)
        for num in numbers:
            if num == int(num) and 0 < num < 1000 and num != max_num:
                qty_candidate = int(num)
                break
                
        if qty_candidate:
            item['qty'] = qty_candidate
            item['value'] = Decimal(str(max_num))
            # Calculate rate
            if qty_candidate > 0:
                item['rate'] = Decimal(str(max_num / qty_candidate))

    return item

def extract_from_bytes(file_bytes, filename: str = '') -> dict:
    """Main entry point: extract text from file and parse invoice data.

    Args:
        file_bytes: Raw bytes of uploaded file
        filename: Original filename (to detect file type)

    Returns:
        dict with extraction results
    """
    if not file_bytes:
        return {
            'success': False,
            'error': 'empty_file',
            'message': 'File is empty. Please upload a valid PDF file.',
            'ocr_available': False,
            'header': {},
            'items': [],
            'raw_text': ''
        }

    # Detect file type
    is_pdf = filename.lower().endswith('.pdf') or (len(file_bytes) > 4 and file_bytes[:4] == b'%PDF')
    is_image = filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.tiff', '.bmp'))

    if is_image:
        return {
            'success': False,
            'error': 'image_file_not_supported',
            'message': 'Image files are not supported. Please convert to PDF or enter details manually.',
            'ocr_available': False,
            'header': {},
            'items': [],
            'raw_text': ''
        }

    if not is_pdf:
        return {
            'success': False,
            'error': 'unsupported_file_type',
            'message': 'Please upload a PDF file.',
            'ocr_available': False,
            'header': {},
            'items': [],
            'raw_text': ''
        }

    # Extract text from PDF
    try:
        text = extract_text_from_pdf(file_bytes)
    except Exception as e:
        logger.error(f"PDF text extraction failed: {e}")
        return {
            'success': False,
            'error': 'pdf_extraction_failed',
            'message': f'Could not extract text from PDF: {str(e)}',
            'ocr_available': False,
            'header': {},
            'items': [],
            'raw_text': ''
        }

    # Validate that we got text
    if not text or not text.strip():
        return {
            'success': False,
            'error': 'no_text_extracted',
            'message': 'No readable text found in PDF.',
            'ocr_available': False,
            'header': {},
            'items': [],
            'raw_text': ''
        }

    # Parse extracted text to structured invoice data
    try:
        parsed = parse_invoice_data(text)

        # Try coordinate-based extraction for items
        coord_items = _extract_items_with_coordinates(file_bytes)
        
        # Use coordinate-based items if available, otherwise use text-based
        final_items = coord_items if coord_items else parsed.get('items', [])

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
        for item in final_items:
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
    """Build standardized invoice JSON from parsed data.
    
    Args:
        parsed: Parsed invoice data from parse_invoice_data
        
    Returns:
        Standardized invoice JSON structure
    """
    # Determine invoice type
    invoice_type = ''
    inv_no = parsed.get('invoice_no') or ''
    if inv_no and inv_no.upper().startswith('PI'):
        invoice_type = 'Proforma Invoice'
    else:
        invoice_type = 'Invoice'

    # Seller details
    seller_details = {
        'name': parsed.get('seller_name') or '',
        'address': parsed.get('seller_address') or '',
        'phone': parsed.get('seller_phone') or '',
        'email': parsed.get('seller_email') or '',
        'vat_number': parsed.get('seller_vat_reg') or ''
    }

    # Customer details
    customer_details = {
        'code': parsed.get('code_no') or '',
        'name': parsed.get('customer_name') or '',
        'address': parsed.get('address') or '',
        'contact_person': parsed.get('kind_attention') or '',
        'phone': parsed.get('phone') or '',
        'email': parsed.get('email') or ''
    }

    # Items
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

    # Totals
    totals = {
        'sub_total': float(parsed.get('subtotal')) if parsed.get('subtotal') else '',
        'vat_amount': float(parsed.get('tax')) if parsed.get('tax') else '',
        'vat_percent': '',
        'discount': '',
        'grand_total': float(parsed.get('total')) if parsed.get('total') else ''
    }

    # Calculate VAT percent if possible
    if totals['sub_total'] and totals['vat_amount'] and totals['sub_total'] > 0:
        try:
            totals['vat_percent'] = round((totals['vat_amount'] / totals['sub_total']) * 100, 2)
        except (ZeroDivisionError, TypeError):
            totals['vat_percent'] = ''

    # Invoice metadata
    invoice_metadata = {
        'invoice_type': invoice_type,
        'invoice_number': inv_no,
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

# Example usage
if __name__ == "__main__":
    # Example of how to use the functions
    def example_usage():
        # Read a PDF file
        with open('invoice.pdf', 'rb') as f:
            file_bytes = f.read()
        
        # Extract data
        result = extract_from_bytes(file_bytes, 'invoice.pdf')
        
        if result['success']:
            print("Extraction successful!")
            print(f"Invoice No: {result['header']['invoice_no']}")
            print(f"Customer: {result['header']['customer_name']}")
            print(f"Total: {result['header']['total']}")
            print(f"Items: {len(result['items'])}")
            
            # Build standardized JSON
            invoice_json = build_invoice_json(parse_invoice_data(result['raw_text']))
            print("\nStandardized JSON:")
            print(json.dumps(invoice_json, indent=2, default=str))
        else:
            print(f"Extraction failed: {result['message']}")
    
    # Uncomment to test
    # example_usage()
