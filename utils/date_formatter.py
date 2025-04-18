from datetime import datetime

def format_due_date(date_str):
    try:
        # Attempt to parse 'YYYY-MM-DD' format
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        # If it fails, try to parse a format with a suffix (e.g., '15th April 2025')
        date_obj = datetime.strptime(date_str.replace('th', '').replace('st', '').replace('nd', '').replace('rd', ''), '%d %B %Y')
    
    day = date_obj.day
    suffix = 'th' if 11 <= day <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
    
    # Format with suffix and month-year
    formatted = f"{day}{suffix} {date_obj.strftime('%B %Y')}"
    return formatted