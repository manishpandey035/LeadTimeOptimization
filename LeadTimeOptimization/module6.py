import os
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

print("="*60)
print("     STAGE 6: LAUNCHING INFERENCE ENGINE VERIFICATION     ")
print("="*60)

# Target the production spreadsheet
excel_filename = "historical_delivery_data_200_records.xlsx"

if not os.path.exists(excel_filename):
    print(f"[ERROR] Process halted. Cannot locate: '{excel_filename}'")
else:
    # 1. Core training pipeline execution
    print("[INFO] Simulating core training pipeline...")
    df = pd.read_excel(excel_filename)
    X = df[['Supplier_Rating', 'Warehouse_Time_Days', 'Transit_Distance_KM']]
    y_actual = df['Actual_Lead_Time_Days']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    production_model = LinearRegression()
    production_model.fit(X_scaled, y_actual)
    
    # Exporting model files to disk
    print("[PROCESSING] Serializing and saving trained model components...")
    with open("trained_scaler.pkl", "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)
    with open("trained_model.pkl", "wb") as model_file:
        pickle.dump(production_model, model_file)
    print("[SUCCESS] Core assets saved securely as 'trained_scaler.pkl' and 'trained_model.pkl'.\n")
    
    # -----------------------------------------------------------------
    # 2. INFERENCE ENGINE VERIFICATION (Live Application Processing)
    # -----------------------------------------------------------------
    print("[INFO] Initializing fresh Inference Bootloader...")
    
    # Reload model components back into memory
    with open("trained_scaler.pkl", "rb") as scaler_file:
        loaded_scaler = pickle.load(scaler_file)
    with open("trained_model.pkl", "rb") as model_file:
        loaded_model = pickle.load(model_file)
    print("[SUCCESS] Model intelligence reloaded into memory successfully.")
    
    # Process fresh incoming delivery logs directly
    print("\n[INFO] Loading fresh un-tracked logistical payload...")
    new_incoming_data = pd.DataFrame({
        'Supplier_Rating': [4.5, 3.2, 4.9],
        'Warehouse_Time_Days': [1.2, 3.8, 0.5],
        'Transit_Distance_KM': [320, 1450, 85]
    })
    
    print("\nNew Incoming Shipments Queue:")
    print(new_incoming_data)
    
    # Process features through the reloaded scaling rules
    X_new_scaled = loaded_scaler.transform(new_incoming_data)
    
    # Calculate delivery windows using the reloaded mathematical model
    predicted_lead_times = loaded_model.predict(X_new_scaled)
    
    # Apply operational classification logic
    def assign_risk_tier(days):
        if days <= 3.0: return "Low Risk (Optimal)"
        elif days <= 5.5: return "Moderate Risk (Monitor)"
        else: return "High Risk (Action Required)"
        
    new_incoming_data['Predicted_Lead_Time_Days'] = [round(p, 2) for p in predicted_lead_times]
    new_incoming_data['Operational_Risk_Status'] = new_incoming_data['Predicted_Lead_Time_Days'].apply(assign_risk_tier)
    
    print("\n" + "-"*75)
    print("                 LIVE INFERENCE ENGINE REAL-TIME OUTPUT                   ")
    print("-"*75)
    print(new_incoming_data.to_string(index=False))
    print("-"*75)
    print("="*60)