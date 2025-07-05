import pandas as pd

def convert_time_to_minutes(time_str):
    if pd.isna(time_str):
        return -1
    try:
        hours, minutes = map(int, str(time_str).split(':'))
        return hours * 60 + minutes
    except:
        return -1
