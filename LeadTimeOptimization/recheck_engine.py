import os
import pickle
import pandas as pd

print("="*60)
print("     ENTERPRISE INFERENCE VERIFICATION: RUNNING 50 RECHECKS     ")
print("="*60)

test_file = "new_50_test_records.xlsx"

if not os.path.exists("trained_model.pkl") or not os.path.exists("trained_scaler.pkl"):
    print("[ERROR] Missing deployment assets! Run Module 6 first to save your model.")
elif not os.path.exists(test_file):
    print(f"[ERROR] Cannot locate the evaluation file: '{test_file}'")
else:
    # 1. Load your saved model from disk
    print("[INFO] Re-loading saved AI serialization binaries...")
    with open("trained_scaler.pkl", "rb") as sf:
        loaded_scaler = pickle.load(sf)
    with open("trained_model.pkl", "rb") as mf:
        loaded_model = pickle.load(mf)
        
    # 2. Read the 50 new rows
    print(f"[INFO] Ingesting '{test_file}' for dynamic execution...")
    new_data = pd.read_excel(test_file)
    
    # 3. Scale variables using rules learned from original data
    X_new = new_data[['Supplier_Rating', 'Warehouse_Time_Days', 'Transit_Distance_KM']]
    X_new_scaled = loaded_scaler.transform(X_new)
    
    # 4. Predict the delivery times
    predictions = loaded_model.predict(X_new_scaled)
    
    # 5. Format and present live analytical results
    new_data['Predicted_Lead_Time_Days'] = [round(p, 2) for p in predictions]
    
    def assign_risk(days):
        if days <= 3.0: return "Low Risk (Optimal)"
        elif days <= 5.5: return "Moderate Risk (Monitor)"
        else: return "High Risk (Action Required)"
        
    new_data['Operational_Risk_Status'] = new_data['Predicted_Lead_Time_Days'].apply(assign_risk)
    
    print("\n" + "-"*80)
    print("                 LIVE INFERENCE ENGINE REAL-TIME OUTPUT (50 ROWS)        ")
    print("-"*80)
    print(new_data.to_string(index=False))
    print("-"*80)
    
    # Export full results to an Excel report for supply chain managers
    output_report = "final_predictions_report_50_rows.xlsx"
    new_data.to_excel(output_report, index=False)
    print(f"[SUCCESS] Operational auditing complete! Report generated: '{output_report}'")
    print("="*60)