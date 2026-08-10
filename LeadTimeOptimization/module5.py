import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

print("="*60)
print("     STAGE 5: LAUNCHING LOGISTICAL KPI DASHBOARD ENGINE     ")
print("="*60)

# Target the production spreadsheet
excel_filename = "historical_delivery_data_200_records.xlsx"

if not os.path.exists(excel_filename):
    print(f"[ERROR] Process halted. Cannot locate: '{excel_filename}'")
else:
    # 1. Load active data stream
    print("[INFO] Ingesting active delivery logs from spreadsheet...")
    df = pd.read_excel(excel_filename)
    
    X = df[['Supplier_Rating', 'Warehouse_Time_Days', 'Transit_Distance_KM']]
    y_actual = df['Actual_Lead_Time_Days']
    
    # 2. Scale features for identical structural mapping
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Train the production deployment engine
    print("[PROCESSING] Syncing Linear Regression parameters...")
    production_model = LinearRegression()
    production_model.fit(X_scaled, y_actual)
    
    # 4. Generate system lead-time predictions
    predicted_lead_times = production_model.predict(X_scaled)
    
    # 5. Define Rule-Based Operational Risk Framework
    def classify_logistical_risk(predicted_days):
        if predicted_days <= 3.0:
            return "Tier 1: Low Risk (Optimal Route)"
        elif predicted_days <= 5.5:
            return "Tier 2: Moderate Risk (Monitor)"
        else:
            return "Tier 3: High Risk (Immediate Action)"
            
    # 6. Assemble the Performance Dashboard Matrix
    dashboard_df = pd.DataFrame({
        'Distance_KM': df['Transit_Distance_KM'],
        'Warehouse_Days': df['Warehouse_Time_Days'],
        'Supplier_Rating': df['Supplier_Rating'],
        'Actual_Days': df['Actual_Lead_Time_Days'],
        'Predicted_Days': [round(p, 2) for p in predicted_lead_times]
    })
    
    # Map predictions to the risk tiers
    dashboard_df['Risk_Classification'] = dashboard_df['Predicted_Days'].apply(classify_logistical_risk)
    
    # 7. Print Dashboard Summary Analytics
    print("\n" + "-"*75)
    print("                PRODUCTION LIVE PERFORMANCE DASHBOARD PREVIEW            ")
    print("-"*75)
    print(dashboard_df.head(10).to_string(index=False))
    print("-"*75)
    
    # Compute operational KPI summary metrics
    total_runs = len(dashboard_df)
    high_risk_count = (dashboard_df['Risk_Classification'] == "Tier 3: High Risk (Immediate Action)").sum()
    high_risk_pct = (high_risk_count / total_runs) * 100
    
    print(f"\n[KPI ALERT] Total active shipping routes audited  : {total_runs}")
    print(f"[KPI ALERT] Critical Bottlenecks Identified       : {high_risk_count} Shipments")
    print(f"[KPI ALERT] Risk Exposure Ratio                   : {high_risk_pct:.1f}% of entire pipeline")
    print("="*60)