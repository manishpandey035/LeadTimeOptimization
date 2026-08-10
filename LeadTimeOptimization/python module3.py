import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

print("="*60)
print("     STAGE 3: LAUNCHING MACHINE LEARNING MODEL TRAINING     ")
print("="*60)

# Target data file
excel_filename = "historical_delivery_data_200_records.xlsx"

if not os.path.exists(excel_filename):
    print(f"[ERROR] Process halted. Cannot locate: '{excel_filename}'")
else:
    # 1. Load and prepare features/targets
    df = pd.read_excel(excel_filename)
    X = df[['Supplier_Rating', 'Warehouse_Time_Days', 'Transit_Distance_KM']]
    y_reg = df['Actual_Lead_Time_Days']
    
    # Scale features exactly like Module 1
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Perform 80/20 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_reg, test_size=0.2, random_state=42)
    print(f"[INFO] Dataset split configured successfully.")
    print(f"       -> Training Records: {len(X_train)} rows")
    print(f"       -> Testing Records : {len(X_test)} rows\n")
    
    # 3. Initialize Algorithms
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(random_state=42),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42)
    }
    
    print("-"*65)
    print("             MODEL EVALUATION METRICS PERFORMANCE              ")
    print("-"*65)
    print(f"{'Algorithm Model Name':<25} | {'MAE (Days)':<12} | {'R-squared (R2)':<15}")
    print("-"*65)
    
    # 4. Train, Predict, and Evaluate each model
    for name, model in models.items():
        # Train the model (System learning happens here!)
        model.fit(X_train, y_train)
        
        # Predict on unseen test data
        predictions = model.predict(X_test)
        
        # Calculate accuracy metrics
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"{name:<25} | {mae:<12.4f} | {r2:<15.4f}")
        
    print("-"*65)
    print("="*60)