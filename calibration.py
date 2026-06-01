import os
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def train_industrial_calibration(csv_path, json_output_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到指定的 CSV 檔案：{csv_path}")
        
    # 1. 讀取包含您圖片中所有欄位的 CSV
    df = pd.read_excel('M006_result.xlsx')
    df_ = pd.read_excel('M006.xlsx')
    
    # 2. 定義特徵欄位 (輸入變數 X) 與 目標欄位 (標準真實值 y)
    feature_cols = [
        'Ø', 'X', 'X_difference', 
        '1&36_Y_difference', '1-36_Y_difference', '3&34_Y_difference', 
        '1&36 pitch error', '1-36 pitch error', '3&34 pitch error'
    ]
    
    X_data = df[feature_cols].values
    y_true = df_[feature_cols].values
    
    # 3. 建立多變量校正模型
    model = LinearRegression(fit_intercept=True)
    model.fit(X_data, y_true)
    
    # 4. 提取各個量測特徵的修正權重 (Coefficients) 與 系統總偏置 (Intercept)
    coefficients = model.coef_.tolist()
    intercept = float(model.intercept_)
    r2 = model.score(X_data, y_true)
    
    # 5. 將所有欄位對應的校正參數打包存成 JSON
    param_dict = {col: coef for col, coef in zip(feature_cols, coefficients)}
    
    config_data = {
        "model_info": "Multivariate Calibration Model",
        "r_squared": r2,
        "system_intercept": intercept,
        "feature_weights": param_dict
    }
    
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=4, ensure_ascii=False)
        
    print("========== 圖片格式 CSV 校正完成 ==========")
    print(f"成功解析特徵數: {len(feature_cols)} 個欄位")
    print(f"模型擬合度 (R²): {r2:.6f}")
    print(f"系統基礎總偏置  : {intercept:.6f} um")
    print(f"校正參數已導出至: {json_output_path}")
    print("==========================================")

# 執行校正
train_industrial_calibration("measurement_data.csv", "sensor_calibration.json")
