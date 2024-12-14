import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# 更新路徑
train_folder = r'C:\weather forecast\training'
input_folder = r'C:\weather forecast\input_folder'
output_folder = r'C:\weather forecast\output_folder'
os.makedirs(output_folder, exist_ok=True)
os.makedirs('model', exist_ok=True)

def train_models():
    print("已經開始訓練模型...")
    files = [os.path.join(train_folder, f) for f in os.listdir(train_folder) if f.endswith('.csv')]

    # 合併所有 CSV 檔案
    data_list = [pd.read_csv(file, header=0) for file in files]
    data = pd.concat(data_list, axis=0, ignore_index=True)

    # 資料欄位名稱
    data.columns = ['LocationCode', 'datetime', 'WindSpeed', 'Pressure', 'Temperature', 'Humidity', 'Sunlight', 'Power']

    # 特徵工程（將 datetime 分解成多個特徵）
    data['Year'] = data['datetime'].astype(str).str[:4].astype(int)
    data['Month'] = data['datetime'].astype(str).str[5:7].astype(int)
    data['Day'] = data['datetime'].astype(str).str[8:10].astype(int)
    data['Hour'] = data['datetime'].astype(str).str[11:13].astype(int)

    # 使用 LocationCode 作為 Station 特徵
    data['Station'] = data['LocationCode']

    # 標準化數據
    scaler = StandardScaler()

    # 模型 1：預測環境數據
    X_env = data[['Year', 'Month', 'Day', 'Hour', 'Station']]
    y_env = data[['WindSpeed', 'Pressure', 'Temperature', 'Humidity', 'Sunlight']]
    X_env_scaled = scaler.fit_transform(X_env)

    X_train_env, X_test_env, y_train_env, y_test_env = train_test_split(X_env_scaled, y_env, test_size=0.2, random_state=42)
    model_env = RandomForestRegressor(random_state=42, n_estimators=100, n_jobs=int(os.cpu_count() * 0.8))
    from tqdm import tqdm
    print("正在訓練環境模型...")
    for i in tqdm(range(1, 101), desc="環境模型訓練進度"):
        model_env.set_params(n_estimators=i, warm_start=True)
        model_env.fit(X_train_env, y_train_env)
    joblib.dump(model_env, 'model/env_model.pkl')
    joblib.dump(scaler, 'model/scaler_env.pkl')

    # 評估環境模型性能
    env_r2_score = model_env.score(X_test_env, y_test_env)
    env_absolute_error = np.sum(np.abs(y_test_env.values - model_env.predict(X_test_env)))
    print(f"環境模型 R^2 分數: {env_r2_score:.2f}")
    print(f"環境模型總絕對誤差: {env_absolute_error:.2f}")

    # 模型 2：預測發電量
    X_power = data[['Year', 'Month', 'Day', 'Hour', 'Station', 'WindSpeed', 'Pressure', 'Temperature', 'Humidity', 'Sunlight']]
    y_power = data['Power']
    X_power_scaled = scaler.fit_transform(X_power)

    X_train_power, X_test_power, y_train_power, y_test_power = train_test_split(X_power_scaled, y_power, test_size=0.2, random_state=42)
    model_power = RandomForestRegressor(random_state=42, n_estimators=100, n_jobs=int(os.cpu_count() * 0.8))
    from tqdm import tqdm
    print("正在訓練發電量模型...")
    for i in tqdm(range(1, 101), desc="發電量模型訓練進度"):
        model_power.set_params(n_estimators=i, warm_start=True)
        model_power.fit(X_train_power, y_train_power)
    joblib.dump(model_power, 'model/power_model.pkl')
    joblib.dump(scaler, 'model/scaler_power.pkl')

    # 評估發電量模型性能
    y_power_pred = model_power.predict(X_test_power)
    power_r2_score = model_power.score(X_test_power, y_test_power)
    power_absolute_error = np.sum(np.abs(y_test_power.values - y_power_pred))
    print(f"發電量模型 R^2 分數: {power_r2_score:.2f}")
    print(f"發電量模型總絕對誤差: {power_absolute_error:.2f}")

    print("模型訓練完成，已保存至 model 資料夾。")

def predict():
    files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.csv')]

    # 載入模型
    env_model = joblib.load('model/env_model.pkl')
    power_model = joblib.load('model/power_model.pkl')
    scaler_env = joblib.load('model/scaler_env.pkl')
    scaler_power = joblib.load('model/scaler_power.pkl')

    for file in files:
        data = pd.read_csv(file, header=0)
        data.columns = ['序號', '答案']

        # 從序號解析時間和站點特徵
        data['Year'] = data['序號'].astype(str).str[:4].astype(int)
        data['Month'] = data['序號'].astype(str).str[4:6].astype(int)
        data['Day'] = data['序號'].astype(str).str[6:8].astype(int)
        data['Hour'] = data['序號'].astype(str).str[8:10].astype(int)
        data['Station'] = data['序號'].astype(str).str[12:14].astype(int)

        X_env = data[['Year', 'Month', 'Day', 'Hour', 'Station']]
        X_env_scaled = scaler_env.transform(X_env)

        # 環境數據預測
        env_predictions = env_model.predict(X_env_scaled)
        data[['WindSpeed', 'Pressure', 'Temperature', 'Humidity', 'Sunlight']] = env_predictions

        # 發電量預測
        X_power = data[['Year', 'Month', 'Day', 'Hour', 'Station', 'WindSpeed', 'Pressure', 'Temperature', 'Humidity', 'Sunlight']]
        X_power_scaled = scaler_power.transform(X_power)
        power_predictions = power_model.predict(X_power_scaled)
        data['答案'] = power_predictions

        # 檢查並生成唯一輸出檔名
        base_filename = "upload"
        output_index = 1
        output_file = os.path.join(output_folder, f"{base_filename}{output_index}(answer).csv")
        while os.path.exists(output_file):
            output_index += 1
            output_file = os.path.join(output_folder, f"{base_filename}{output_index}(answer).csv")

        # 輸出結果
        data[['序號', '答案']].to_csv(output_file, index=False, header=True)

    print(f"預測完成，結果已保存至 {output_folder} 資料夾。")

if __name__ == '__main__':
    train_models()
    predict()
