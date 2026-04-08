import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score

# Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


# ==============================
# LOAD DATA
# ==============================
def load_data(file_path):
    df = pd.read_csv(file_path)
    print("✅ Data Loaded Successfully")
    return df


# ==============================
# SPLIT DATA
# ==============================
def split_data(df):
    X = df.drop('Price', axis=1)
    y = df['Price']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("✅ Data Split Completed")
    return X_train, X_test, y_train, y_test


# ==============================
# ENCODING FUNCTION
# ==============================
def encode_data(X_train, X_test):
    
    categorical_cols = ['Company', 'TypeName', 'OpSys', 'CPU_name', 'Gpu_brand']
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
        
        encoders[col] = le
    
    print("✅ Encoding Completed")
    return X_train, X_test, encoders


# ==============================
# SCALING FUNCTION
# ==============================
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ✅ Save column order
    feature_names = X_train.columns
    
    print("✅ Scaling Done")
    return X_train_scaled, X_test_scaled, scaler, feature_names


# ==============================
# TRAIN MODELS
# ==============================
def train_models(X_train, X_test, y_train, y_test):
    
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet(),
        "DecisionTree": DecisionTreeRegressor(),
        "BayesianRidge": BayesianRidge(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = []
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        score = r2_score(y_test, y_pred)
        
        results.append({
            "Model": name,
            "Score": score,
            "Model_Object": model
        })
        
        print(f"{name} → R2 Score: {score:.4f}")
    
    return results





# ==============================
# HYPERPARAMETER TUNING FUNCTION
# ==============================
def tune_models(X_train, X_test, y_train, y_test):
    
    model_params = {
        
        "LinearRegression": {
            "model": LinearRegression(),
            "params": {
                "fit_intercept": [True, False]
            }
        },
        
        "Ridge": {
            "model": Ridge(),
            "params": {
                "alpha": [0.01, 0.1, 1, 10, 100]
            }
        },
        
        "Lasso": {
            "model": Lasso(),
            "params": {
                "alpha": [0.01, 0.1, 1, 10]
            }
        },
        
        "ElasticNet": {
            "model": ElasticNet(),
            "params": {
                "alpha": [0.01, 0.1, 1],
                "l1_ratio": [0.2, 0.5, 0.8]
            }
        },
        
        "DecisionTree": {
            "model": DecisionTreeRegressor(),
            "params": {
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10]
            }
        },
        
        "BayesianRidge": {
            "model": BayesianRidge(),
            "params": {
                "alpha_1": [1e-6, 1e-5],
                "lambda_1": [1e-6, 1e-5]
            }
        },
        
        "RandomForest": {
            "model": RandomForestRegressor(random_state=42),
            "params": {
                "n_estimators": [50, 100],
                "max_depth": [None, 10, 20]
            }
        }
    }
    
    results = []
    
    for name, mp in model_params.items():
        
        print(f"\n🔍 Tuning {name}...")
        
        grid = GridSearchCV(
            mp["model"],
            mp["params"],
            cv=5,
            scoring='r2',
            n_jobs=-1
        )
        
        grid.fit(X_train, y_train)
        
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        score = r2_score(y_test, y_pred)
        
        results.append({
            "Model": name,
            "Best_Params": grid.best_params_,
            "Score": score,
            "Model_Object": best_model
        })
        
        print(f"✅ Best Params: {grid.best_params_}")
        print(f"🔥 R2 Score: {score:.4f}")
    
    return results


# ==============================
# COMPARE BEFORE & AFTER TUNING
# ==============================
def compare_results(before_results, after_results):
    
    print("\n📊 MODEL COMPARISON (Before vs After Tuning)\n")
    
    before_dict = {res['Model']: res['Score'] for res in before_results}
    after_dict = {res['Model']: res['Score'] for res in after_results}
    
    for model in before_dict:
        before_score = before_dict.get(model, 0)
        after_score = after_dict.get(model, 0)
        
        print(f"{model}:")
        print(f"   Before Tuning → {before_score:.4f}")
        print(f"   After Tuning  → {after_score:.4f}")
        print("-" * 40)



# ==============================
# RESULT FUNCTION
# ==============================
def get_best_model(results):
    
    results_sorted = sorted(results, key=lambda x: x['Score'], reverse=True)
    
    print("\n📊 Model Performance (Descending):\n")
    
    for res in results_sorted:
        print(f"{res['Model']} → {res['Score']:.4f}")
    
    best_model = results_sorted[0]
    
    print("\n🏆 Best Model:", best_model['Model'])
    print("🔥 Best Score:", round(best_model['Score'], 4))
    
    return best_model, results_sorted


# ==============================
# SAVE MODEL FUNCTION
# ==============================
def save_model(model, scaler, encoders, accuracy, path):
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    joblib.dump({
        "model": model,
        "scaler": scaler,
        "encoders": encoders,
        "Accuracy": accuracy   # ✅ ADD THIS
    }, path)
    
    print(f"✅ Model saved at: {path}")




# ==============================
# PREDICT FUNCTION (TEST DATA)
# ==============================
# def predict_price(model, scaler, encoders, feature_names):
    
#     print("\n🔮 Testing on Predefined Data...\n")
    
#     # =========================
#     # PREDEFINED DATA (UNCHANGED)
#     # =========================
#     sample_data = {
#         'Company': 'Dell',
#         'TypeName': 'Notebook',
#         'Ram': 8,
#         'Weight': 2.0,
#         'TouchScreen': 0,
#         'IPS': 1,
#         'PPI': 141,
#         'CPU_name': 'Intel Core i5',
#         'HDD': 1000,
#         'SSD': 0,
#         'Gpu_brand': 'Intel',
#         'OpSys': 'Windows'
#     }
    
#     sample_df = pd.DataFrame([sample_data])
    
#     # Encode predefined
#     for col in ['Company', 'TypeName', 'OpSys', 'CPU_name', 'Gpu_brand']:
#         le = encoders[col]
        
#         if sample_df[col][0] in le.classes_:
#             sample_df[col] = le.transform(sample_df[col])
#         else:
#             sample_df[col] = 0
    
#     # Align columns
#     sample_df = sample_df.reindex(columns=feature_names, fill_value=0)
    
#     # Scale & predict
#     sample_scaled = scaler.transform(sample_df)
#     prediction = model.predict(sample_scaled)
    
#     print(f"💰 Predefined Predicted Price: ₹ {round(prediction[0], 2)}")
    
    
#     # =========================
#     # USER INPUT SECTION
#     # =========================
#     print("\n🧑 Enter Your Own Laptop Details:\n")
    
#     user_data = {}
    
#     for col in feature_names:
        
#         # Skip encoded columns for manual handling
#         if col in ['Company', 'TypeName', 'OpSys', 'CPU_name', 'Gpu_brand']:
#             value = input(f"{col} (string): ")
#         else:
#             value = float(input(f"{col} (numeric): "))
        
#         user_data[col] = value
    
#     user_df = pd.DataFrame([user_data])
    
#     # Encode user input
#     for col in ['Company', 'TypeName', 'OpSys', 'CPU_name', 'Gpu_brand']:
#         le = encoders[col]
        
#         if user_df[col][0] in le.classes_:
#             user_df[col] = le.transform(user_df[col])
#         else:
#             print(f"⚠️ Unknown value '{user_df[col][0]}' in {col}, assigning 0")
#             user_df[col] = 0
    
#     # Align again (safety)
#     user_df = user_df.reindex(columns=feature_names, fill_value=0)
    
#     # Scale & predict
#     user_scaled = scaler.transform(user_df)
#     user_prediction = model.predict(user_scaled)
    
#     print(f"\n💰 Your Predicted Price: ₹ {round(user_prediction[0], 2)}")



# ==============================
# MAIN FUNCTION
# ==============================
def main():
    
    file_path = "C:\\Users\\vaishnavi potode\\OneDrive\\Desktop\\Projects\\laptop_price_prediction\\fourth_clean.csv"        # change if needed
    save_path = "C:\\Users\\vaishnavi potode\\OneDrive\\Desktop\\Projects\\laptop_price_prediction\\best_model.pkl"  # change if needed
    
    # Step 1: Load

    df = load_data(file_path)
    
    # Step 2: Split
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Step 3: Encode
    X_train, X_test, encoders = encode_data(X_train, X_test)
    
    # Step 4: Scale
    X_train_scaled, X_test_scaled, scaler, feature_names = scale_data(X_train, X_test)
    
    # Step 5: Train
    before_results = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
    after_results = tune_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Step 6: Compare
    compare_results(before_results, after_results)
    best_model, _ = get_best_model(after_results)
    
    # Step 7: Save best model
    save_model(
        best_model['Model_Object'],
        scaler,
        encoders,
        best_model['Score'], 
        save_path
    )
     
    # Step 8: Predict on test data
    # predict_price(best_model['Model_Object'], scaler, encoders, feature_names)


# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    main()