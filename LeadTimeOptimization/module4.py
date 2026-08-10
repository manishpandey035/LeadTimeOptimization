import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

print("="*60)
print("     STAGE 4: DEPLOYING 5-FOLD CROSS-VALIDATION ENGINE     ")
print("="*60)

# Target the main database file
excel_filename = "historical_delivery_data_200_records.xlsx"

if not os.path.exists(excel_filename):
    print(f"[ERROR] Process halted. Cannot locate: '{excel_filename}'")
else:
    # 1. Load data matrices
    df = pd.read_excel(excel_filename)
    X = df[['Supplier_Rating', 'Warehouse_Time_Days', 'Transit_Distance_KM']]
    y_reg = df['Actual_Lead_Time_Days']
    
    # 2. Re-apply uniform standard scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Configure the models for iterative evaluation
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(random_state=42),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42)
    }
    
    print("[INFO] Initializing K-Fold iterative evaluation (K=5 partitions)...")
    print("\n"+ "-"*65)
    print("             CROSS-VALIDATION GENERALIZATION SCORES             ")
    print("-"*65)
    
    # 4. Compute Cross-Validation Scores
    for name, model in models.items():
        # cross_val_score executes 5 independent train-test iterations automatically
        cv_scores = cross_val_score(model, X_scaled, y_reg, cv=5, scoring='r2')
        
        print(f"-> {name:<25}")
        print(f"   Individual Fold R2 Scores : {[round(score, 4) for score in cv_scores]}")
        print(f"   Average Cross-Validation R2: {cv_scores.mean():.4f}")
        print("-"*65)

    print("="*60)