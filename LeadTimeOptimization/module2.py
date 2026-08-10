import os
import pandas as pd

print("="*60)
print("     STAGE 2: LAUNCHING EXPLORATORY DATA ANALYSIS (EDA)   ")
print("="*60)

# Target the data file
excel_filename = "historical_delivery_data_200_records.xlsx"

if not os.path.exists(excel_filename):
    print(f"[ERROR] Process halted. Cannot locate: '{excel_filename}'")
else:
    # 1. Load data
    print(f"[INFO] Accessing database for descriptive statistics...")
    df = pd.read_excel(excel_filename)
    
    # 2. Compute Global Descriptive Metrics
    total_logs = len(df)
    avg_supplier_rating = df['Supplier_Rating'].mean()
    avg_warehouse_time = df['Warehouse_Time_Days'].mean()
    avg_distance = df['Transit_Distance_KM'].mean()
    avg_lead_time = df['Actual_Lead_Time_Days'].mean()
    delay_rate = (df['Delay_Status'].sum() / total_logs) * 100
    
    # 3. Print Summary Report to Terminal
    print("\n" + "-"*45)
    print("         GLOBAL LOGISTICAL METRICS SUMMARY      ")
    print("-"*45)
    print(f"Total Logistical Logs Analysed   : {total_logs} Shipments")
    print(f"Average Supplier Rating (1-5)    : {avg_supplier_rating:.2f} Stars")
    print(f"Average Warehouse Process Time   : {avg_warehouse_time:.2f} Days")
    print(f"Average Transit Distance Covered : {avg_distance:.1f} KM")
    print(f"Average Actual Delivery Lead Time: {avg_lead_time:.2f} Days")
    print(f"Overall Supply Chain Delay Rate  : {delay_rate:.1f}%")
    print("-"*45)
    
    # 4. Statistical Correlation Matrix
    print("\n[PROCESSING] Computing Pearson Correlation Coefficients...")
    correlation_matrix = df.corr()
    lead_time_corr = correlation_matrix['Actual_Lead_Time_Days'].sort_values(ascending=False)
    
    print("\n[RESULT] Linear Correlation relative to Actual_Lead_Time_Days:")
    print(lead_time_corr.to_string())
    print("="*60)