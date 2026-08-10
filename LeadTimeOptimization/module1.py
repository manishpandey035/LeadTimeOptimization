import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

print("="*60)
print("     STAGE 1: LAUNCHING DATA PREPROCESSING PIPELINE     ")
print("="*60)

# Define the exact Excel workbook filename
excel_filename = "historical_delivery_data_200_records.xlsx"

# Check if the file exists in the folder
if not os.path.exists(excel_filename):
    print(f"[ERROR] Process halted. Cannot locate: '{excel_filename}'")
else:
    # 1. Excel Data Ingestion
    print(f"[INFO] Connecting to system ledger: '{excel_filename}'...")
    raw_df = pd.read_excel(excel_filename)
    print(f"[SUCCESS] Ingested database profile matrix: {raw_df.shape[0]} rows and {raw_df.shape[1]} columns.")
    
    # 2. Structural Integrity Validation (Duplicates Check)
    cleaned_df = raw_df.copy()
    duplicate_count = cleaned_df.duplicated().sum()
    if duplicate_count > 0:
        cleaned_df.drop_duplicates(inplace=True)
        print(f"[WARNING] Redundancy flagged. Purged {duplicate_count} duplicated entries.")
    else:
        print("[SUCCESS] Integrity confirmed. Zero structural duplicates found.")
        
    # 3. Missing Feature Block Isolation & Median Imputation
    for column in cleaned_df.columns:
        missing_count = cleaned_df[column].isnull().sum()
        if missing_count > 0:
            median_value = cleaned_df[column].median()
            cleaned_df[column].fillna(median_value, inplace=True)
            print(f"[REMEDIATION] Rectified {missing_count} missing fields in '{column}' using median ({median_value}).")
            
    # 4. Feature Extraction Boundary Configuration
    X = cleaned_df[['Supplier_Rating', 'Warehouse_Time_Days', 'Transit_Distance_KM']]
    y_reg = cleaned_df['Actual_Lead_Time_Days']
    
    # 5. Z-Score Scale Standardisation Processing
    print("[PROCESSING] Deploying StandardScaler normalization module...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    print("="*60)
    print("      STAGE 1 PROCESSING COMPLETE: MATRIX FORMATTED     ")
    print("="*60)
    print("\n[PREVIEW] Normalized Operational Feature Space Vector (First 5 Rows):")
    print(X_scaled_df.head(5).to_string(index=True))