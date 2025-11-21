import csv
import io
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required, permission_required
from django.contrib import messages
from django.http import JsonResponse
from django.db import transaction
from django.views.decorators.http import require_http_methods
from .models import LabourCode
from .forms import LabourCodeForm, LabourCodeCSVImportForm


@login_required
@permission_required('tracker.view_labourcode', raise_exception=True)
def labour_codes_list(request):
    """List all labour codes with search and filter functionality"""
    labour_codes = LabourCode.objects.all().order_by('code')
    
    # Search functionality
    search_query = request.GET.get('search', '').strip()
    if search_query:
        labour_codes = labour_codes.filter(
            models.Q(code__icontains=search_query) |
            models.Q(description__icontains=search_query) |
            models.Q(category__icontains=search_query)
        )
    
    # Filter by category
    category_filter = request.GET.get('category', '').strip()
    if category_filter:
        labour_codes = labour_codes.filter(category__icontains=category_filter)
    
    # Filter by active status
    active_filter = request.GET.get('active', '')
    if active_filter == 'true':
        labour_codes = labour_codes.filter(is_active=True)
    elif active_filter == 'false':
        labour_codes = labour_codes.filter(is_active=False)
    
    # Get distinct categories for filter dropdown
    categories = LabourCode.objects.values_list('category', flat=True).distinct().order_by('category')
    
    context = {
        'labour_codes': labour_codes,
        'categories': categories,
        'search_query': search_query,
        'category_filter': category_filter,
        'active_filter': active_filter,
        'total_count': LabourCode.objects.count(),
    }
    
    return render(request, 'tracker/labour_codes_list.html', context)


@login_required
@permission_required('tracker.add_labourcode', raise_exception=True)
def labour_code_create(request):
    """Create a new labour code manually"""
    if request.method == 'POST':
        form = LabourCodeForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, f'Labour code {form.cleaned_data["code"]} created successfully!')
            return redirect('tracker:labour_codes_list')
    else:
        form = LabourCodeForm()
    
    context = {
        'form': form,
        'title': 'Add Labour Code',
        'action': 'Create',
    }
    return render(request, 'tracker/labour_code_form.html', context)


@login_required
@permission_required('tracker.change_labourcode', raise_exception=True)
def labour_code_edit(request, pk):
    """Edit an existing labour code"""
    labour_code = get_object_or_404(LabourCode, pk=pk)
    
    if request.method == 'POST':
        form = LabourCodeForm(request.POST, instance=labour_code)
        if form.is_valid():
            form.save()
            messages.success(request, f'Labour code {labour_code.code} updated successfully!')
            return redirect('tracker:labour_codes_list')
    else:
        form = LabourCodeForm(instance=labour_code)
    
    context = {
        'form': form,
        'title': f'Edit Labour Code {labour_code.code}',
        'action': 'Edit',
        'labour_code': labour_code,
    }
    return render(request, 'tracker/labour_code_form.html', context)


@login_required
@permission_required('tracker.delete_labourcode', raise_exception=True)
def labour_code_delete(request, pk):
    """Delete a labour code"""
    labour_code = get_object_or_404(LabourCode, pk=pk)
    
    if request.method == 'POST':
        code = labour_code.code
        labour_code.delete()
        messages.success(request, f'Labour code {code} deleted successfully!')
        return redirect('tracker:labour_codes_list')
    
    context = {
        'labour_code': labour_code,
    }
    return render(request, 'tracker/labour_code_confirm_delete.html', context)


@login_required
@permission_required('tracker.add_labourcode', raise_exception=True)
def labour_codes_import(request):
    """Import labour codes from CSV file"""
    import_stats = None
    
    if request.method == 'POST':
        form = LabourCodeCSVImportForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = request.FILES['csv_file']
            clear_existing = form.cleaned_data.get('clear_existing', False)
            
            try:
                import_stats = _process_csv_import(csv_file, clear_existing)
                
                if import_stats['success']:
                    messages.success(
                        request,
                        f"Import completed! Created: {import_stats['created']}, "
                        f"Updated: {import_stats['updated']}, "
                        f"Errors: {import_stats['errors']}"
                    )
                    if import_stats['error_details']:
                        messages.warning(request, f"Issues found: {'; '.join(import_stats['error_details'][:5])}")
                else:
                    messages.error(request, f"Import failed: {import_stats['error_message']}")
                
                if not import_stats.get('keep_form'):
                    return redirect('tracker:labour_codes_list')
            except Exception as e:
                messages.error(request, f"Error processing CSV file: {str(e)}")
                import_stats = None
    else:
        form = LabourCodeCSVImportForm()
    
    context = {
        'form': form,
        'import_stats': import_stats,
        'title': 'Import Labour Codes',
    }
    return render(request, 'tracker/labour_codes_import.html', context)


def _process_csv_import(csv_file, clear_existing=False):
    """Process CSV file and import labour codes"""
    try:
        # Read CSV file
        if isinstance(csv_file, str):
            csv_content = csv_file
        else:
            csv_content = csv_file.read().decode('utf-8-sig')
        
        # Parse CSV
        csv_reader = csv.DictReader(io.StringIO(csv_content))
        
        if not csv_reader.fieldnames:
            return {
                'success': False,
                'error_message': 'CSV file is empty or invalid format',
            }
        
        # Check required columns
        required_cols = {'code', 'description', 'category'}
        csv_cols = set(col.strip().lower() for col in csv_reader.fieldnames)
        
        if not required_cols.issubset(csv_cols):
            return {
                'success': False,
                'error_message': f'CSV must contain columns: code, description, category. Found: {", ".join(csv_reader.fieldnames)}',
            }
        
        with transaction.atomic():
            if clear_existing:
                LabourCode.objects.all().delete()
            
            created_count = 0
            updated_count = 0
            error_details = []
            
            for row_num, row in enumerate(csv_reader, start=2):
                try:
                    code = row.get('code', '').strip().upper()
                    description = row.get('description', '').strip()
                    category = row.get('category', '').strip().lower()
                    is_active = row.get('is_active', 'true').lower() in ['true', '1', 'yes']
                    
                    # Validate required fields
                    if not code:
                        error_details.append(f"Row {row_num}: Code is required")
                        continue
                    
                    if not description:
                        error_details.append(f"Row {row_num}: Description is required")
                        continue
                    
                    if not category:
                        error_details.append(f"Row {row_num}: Category is required")
                        continue
                    
                    # Create or update labour code
                    obj, created = LabourCode.objects.update_or_create(
                        code=code,
                        defaults={
                            'description': description,
                            'category': category,
                            'is_active': is_active,
                        }
                    )
                    
                    if created:
                        created_count += 1
                    else:
                        updated_count += 1
                
                except Exception as e:
                    error_details.append(f"Row {row_num}: {str(e)}")
        
        return {
            'success': True,
            'created': created_count,
            'updated': updated_count,
            'errors': len(error_details),
            'error_details': error_details,
        }
    
    except UnicodeDecodeError:
        return {
            'success': False,
            'error_message': 'File encoding error. Please use UTF-8 encoded CSV files.',
        }
    except Exception as e:
        return {
            'success': False,
            'error_message': f'Error processing file: {str(e)}',
        }


@login_required
@permission_required('tracker.view_labourcode', raise_exception=True)
@require_http_methods(['GET'])
def api_labour_codes(request):
    """API endpoint to get labour codes for JS usage"""
    codes = list(
        LabourCode.objects.filter(is_active=True).values('code', 'description', 'category')
    )
    return JsonResponse({'codes': codes})


from django.db import models
