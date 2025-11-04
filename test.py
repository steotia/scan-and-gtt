# test_full_bhavcopy.py
from jugaad_data.nse import full_bhavcopy_save, bhavcopy_save
from datetime import date
import pandas as pd
import glob
import os

# Clean up old files
for f in glob.glob("*.csv"):
    try:
        os.remove(f)
    except:
        pass

test_date = date(2025, 10, 31)

print("=" * 50)
print("Testing FULL bhavcopy vs regular bhavcopy")
print("=" * 50)

# Test 1: Regular bhavcopy
try:
    print(f"\n1. Regular bhavcopy for {test_date}...")
    bhavcopy_save(test_date, ".")
    
    files = glob.glob("*.csv")
    if files:
        print(f"✅ Downloaded: {files[0]}")
        df = pd.read_csv(files[0])
        print(f"   Columns ({len(df.columns)}): {df.columns.tolist()[:10]}...")
        
        # Check for delivery
        delivery_cols = [c for c in df.columns if 'DELIV' in c.upper()]
        if delivery_cols:
            print(f"   🎯 DELIVERY COLUMNS: {delivery_cols}")
        else:
            print(f"   ❌ No delivery columns")
            
        # Clean up
        os.remove(files[0])
        
except Exception as e:
    print(f"❌ Regular bhavcopy failed: {e}")

# Test 2: FULL bhavcopy (might include delivery)
try:
    print(f"\n2. FULL bhavcopy for {test_date}...")
    full_bhavcopy_save(test_date, ".")
    
    files = glob.glob("*.csv")
    if files:
        print(f"✅ Downloaded: {files[0]}")
        df = pd.read_csv(files[0])
        print(f"   Columns ({len(df.columns)}): {df.columns.tolist()}")
        
        # Check for delivery columns
        delivery_cols = [c for c in df.columns if any(word in c.upper() for word in ['DELIV', 'QTY', 'PERCENT'])]
        if delivery_cols:
            print(f"   🎯 DELIVERY COLUMNS FOUND: {delivery_cols}")
            # Show sample delivery data
            print(f"\n   Sample delivery data:")
            print(df[['SYMBOL'] + delivery_cols].head())
        else:
            print(f"   ❌ No delivery columns even in full bhavcopy")
            
except Exception as e:
    print(f"❌ Full bhavcopy failed: {e}")