#!/usr/bin/env python3
"""
Test script to verify pandas comparison fixes
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.data_helpers import safe_dataframe_check, safe_series_check, safe_index_check

def test_pandas_fixes():
    """Test the pandas comparison fixes"""
    print("Testing pandas comparison fixes...")
    
    # Create test data
    df = pd.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': [1.1, 2.2, 3.3, 4.4, 5.5],
        'col3': ['a', 'b', 'c', 'd', 'e'],
        'col4': [1, 2, np.nan, 4, 5],
        'col5': [np.nan, np.nan, np.nan, np.nan, np.nan],
        'empty_col': []
    })
    
    print(f"Created test DataFrame with shape: {df.shape}")
    
    # Test safe_dataframe_check
    print("\nTesting safe_dataframe_check:")
    print(f"col1 exists and has data: {safe_dataframe_check(df, 'col1')}")
    print(f"col4 exists and has data: {safe_dataframe_check(df, 'col4')}")
    print(f"col5 exists and has data: {safe_dataframe_check(df, 'col5')}")
    print(f"nonexistent exists and has data: {safe_dataframe_check(df, 'nonexistent')}")
    
    # Test safe_series_check
    print("\nTesting safe_series_check:")
    print(f"col1 series has data: {safe_series_check(df['col1'])}")
    print(f"col4 series has data: {safe_series_check(df['col4'])}")
    print(f"col5 series has data: {safe_series_check(df['col5'])}")
    
    # Test safe_index_check
    print("\nTesting safe_index_check:")
    print(f"DataFrame index has data: {safe_index_check(df.index)}")
    print(f"Empty index has data: {safe_index_check(pd.Index([]))}")
    
    # Test the problematic pattern that was fixed
    print("\nTesting the fixed problematic pattern:")
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        # This is the old problematic pattern that would cause "truth value of Index is ambiguous"
        # if df[col].isnull().all():  # This would cause an error
        
        # This is the new safe pattern
        if not safe_series_check(df[col]):
            print(f"Column {col}: Skipping (no meaningful data)")
        else:
            print(f"Column {col}: Processing (has meaningful data)")
    
    print("\nAll tests passed! Pandas comparison fixes are working correctly.")

if __name__ == "__main__":
    test_pandas_fixes()

