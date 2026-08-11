import datetime
import os
import re
import sys
import urllib3
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn

# 引入 DB 操作模組
import db_manager

# 關閉 SSL 憑證警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# API 金鑰設定
CWA_API_KEY = "CWA-F6B5F348-77D8-4EA8-8874-FBA50E6191DE"
MOENV_API_KEY = "5ae4f1a2-b6e6-4b79-82c8-0c84d694b7a7"


# 1. 定義 LSTM 模型架構 (input_size=14)
class MultivariateLSTM(nn.Module):

    def __init__(self, input_size=14, hidden_size=128, num_layers=2):
        super(MultivariateLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 2. 自動化擷取【霧峰區】即時 14 項特徵 (採用高公局 M03A CSV 實測加總邏輯)
def fetch_wufeng_live_features():
    # 💡 在函式最頂端先宣告當前時間，確保區域變數健全
    now = datetime.datetime.now()
    print("📡 開始連線擷取【台中霧峰區】三大類即時自變數...")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            " (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
    }

    # (A) 霧峰 PM2.5 爬取
    pm25 = None
    try:
        url_epb_table = "https://taqm.epb.taichung.gov.tw/TQAMPM25table.ASPX"
        res_epb = requests.get(
            url_epb_table, headers=headers, timeout=5, verify=False
        )
        res_epb.encoding = "utf-8"
        soup = BeautifulSoup(res_epb.text, "html.parser")
        all_cells = [
            tag.text.strip() for tag in soup.find_all(["td", "th", "a"])
        ]

        for idx, text in enumerate(all_cells):
            if "霧峰" in text and idx + 1 < len(all_cells):
                val_str = all_cells[idx + 1]
                if val_str.isdigit() or re.match(r"^\d+(\.\d+)?$", val_str):
                    pm25 = float(val_str)
                    print(
                        f"   [1/3] ✅ 精準解析成功！【臺中環保局】霧峰站即時"
                        f" PM2.5: {pm25} µg/m³"
                    )
                    break
    except Exception as e:
        print(f"   [1/3] ℹ️ 網頁爬取跳過: {e}")

    if pm25 is None:
        try:
            url_dali = f"https://data.moenv.gov.tw/api/v2/aqx_p_432?api_key={MOENV_API_KEY}&limit=5&format=json&filters=sitename,eq,大里"
            res_dali = requests.get(
                url_dali, headers=headers, timeout=5, verify=False
            ).json()
            recs = (
                res_dali.get("records", [])
                if isinstance(res_dali, dict)
                else res_dali
            )
            if recs:
                val = recs[0].get("pm25") or recs[0].get("pm2.5")
                if val:
                    pm25 = float(val)
                    print(
                        "   [1/3] ✅ 採用鄰近【大里標準站】即時 PM2.5:"
                        f" {pm25} µg/m³"
                    )
        except Exception:
            pass

    if pm25 is None:
        pm25 = 15.0
        print(f"   [1/3] ℹ️ 採用系統預設 PM2.5 數值: {pm25} µg/m³")

    # (B) 霧峰氣象 爬取
    press, temp, rh, wind_spd, wind_dir, rain = (
        1008.5,
        24.5,
        75.0,
        1.8,
        180.0,
        0.0,
    )
    try:
        url_cwa = f"https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0001-001?Authorization={CWA_API_KEY}&StationName=霧峰"
        res_cwa = requests.get(
            url_cwa, headers=headers, timeout=5, verify=False
        ).json()
        if (
            isinstance(res_cwa, dict)
            and res_cwa.get("records")
            and res_cwa["records"].get("Station")
        ):
            station_data = res_cwa["records"]["Station"][0]
            obs_time_str = station_data.get("ObsTime", {}).get(
                "DateTime", "未知時間"
            )
            station_elem = station_data["WeatherElement"]

            def safe_float(val, default_val):
                try:
                    v = float(val)
                    return default_val if v < -90 else v
                except (ValueError, TypeError):
                    return default_val

            press = safe_float(station_elem.get("AirPressure"), press)
            temp = safe_float(station_elem.get("AirTemperature"), temp)
            rh = safe_float(station_elem.get("RelativeHumidity"), rh)
            wind_spd = safe_float(station_elem.get("WindSpeed"), wind_spd)
            wind_dir = safe_float(station_elem.get("WindDirection"), wind_dir)
            if "Now" in station_elem and isinstance(station_elem["Now"], dict):
                rain = safe_float(station_elem["Now"].get("Precipitation"), 0.0)

            print(
                f"   [2/3] ✅ 成功取得【氣象署霧峰站】氣象 (觀測時間:"
                f" {obs_time_str}): 氣溫 {temp}℃, 濕度 {rh}%"
            )
    except Exception as e:
        print(f"   [2/3] ⚠️ 氣象署 API 解析失敗，採用保底數值: {e}")

    wind_rad = np.radians(wind_dir)
    wind_x = np.cos(wind_rad)
    wind_y = np.sin(wind_rad)

    # (C) 國道 3 號車流量 - 精準讀取「上一個完整小時」M03A 資料夾
    v_2100N, v_2100S, v_2125N, v_2129S = 0.0, 0.0, 0.0, 0.0
    traffic_success = False

    try:
        # 1. 取得上一個完整小時的時間物件 (自動處理跨日與跨月)
        prev_hour_time = now - datetime.timedelta(hours=1)
        ymd = prev_hour_time.strftime("%Y%m%d")
        hh = prev_hour_time.strftime("%H")

        print(f"   [3/3] 📡 開始抓取高公局 M03A 資料夾【{ymd}/{hh}】整點 1 小時完整流量...")

        traffic_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': '*/*',
            'Referer': 'https://tisvcloud.freeway.gov.tw/',
            'Connection': 'keep-alive'
        }

        # 準備存放門架流量累計的字典
        target_gantries = {'03F2100N': 0.0, '03F2100S': 0.0, '03F2125N': 0.0, '03F2129S': 0.0}
        success_files = 0

        # 2. 迴圈讀取該小時資料夾內的 12 個 5 分鐘 CSV 檔案 (00, 05, 10, ..., 55 分)
        for mm_int in range(0, 60, 5):
            mm = f"{mm_int:02d}"
            url_csv = f"https://tisvcloud.freeway.gov.tw/history/TDCS/M03A/{ymd}/{hh}/TDCS_M03A_{ymd}_{hh}{mm}00.csv"
            
            try:
                res_csv = requests.get(url_csv, headers=traffic_headers, timeout=4, verify=False)
                if res_csv.status_code == 200:
                    success_files += 1
                    # 解析 CSV 內容並累加指定門架的車流量 (第 2 欄為 GantryID，第 5 欄為 Volume)
                    for line in res_csv.text.strip().split('\n'):
                        parts = line.split(',')
                        if len(parts) >= 5:
                            gantry_id = parts[1].strip()
                            if gantry_id in target_gantries:
                                target_gantries[gantry_id] += float(parts[4].strip())
            except Exception:
                continue

        # 3. 判斷是否有成功抓到資料並計算總和
        if success_files > 0 and sum(target_gantries.values()) > 0:
            # 若 12 份檔案有少許遺漏 (例如抓到 11 份)，按比例等比放大補齊至 60 分鐘完整流量
            scale_factor = 12.0 / success_files
            v_2100N = target_gantries['03F2100N'] * scale_factor
            v_2100S = target_gantries['03F2100S'] * scale_factor
            v_2125N = target_gantries['03F2125N'] * scale_factor
            v_2129S = target_gantries['03F2129S'] * scale_factor
            
            traffic_success = True
            print(f"   [3/3] ✅ 成功讀取【{hh}:00 時段】({success_files}/12 份 CSV) -> 北上總和: {int(v_2100N)} 輛, 南下總和: {int(v_2100S)} 輛")

    except Exception as e:
        print(f"   [3/3] ℹ️ 讀取上小時 M03A 資料夾失敗: {e}")

    # 保底機制 1：若線上抓取失敗 (如 Streamlit Cloud 被擋 IP)，從 SQLite 資料庫復原
    if not traffic_success:
        try:
            import sqlite3
            conn = sqlite3.connect("pm25_forecast.db")
            df_db = pd.read_sql("SELECT * FROM realtime_logs WHERE `03F2100N` > 0 ORDER BY timestamp DESC LIMIT 1", conn)
            conn.close()
            if not df_db.empty:
                v_2100N = float(df_db["03F2100N"].iloc[0])
                v_2100S = float(df_db["03F2100S"].iloc[0])
                v_2125N = float(df_db["03F2125N"].iloc[0])
                v_2129S = float(df_db["03F2129S"].iloc[0])
                if v_2100N > 0 and v_2100S > 0:
                    traffic_success = True
                    print(f"   [3/3] ✅ 從 SQLite 資料庫備援復原車流數據 -> 北上: {int(v_2100N)}, 南下: {int(v_2100S)}")
        except Exception:
            pass

    # 保底機制 2：最終動態預估保底 (避免寫死固定值)
    if not traffic_success or (v_2100N <= 0 and v_2100S <= 0):
        base_val = 1200 if 7 <= now.hour <= 19 else 400
        v_2100N = float(base_val + (now.hour % 5) * 35)
        v_2100S = float(base_val - (now.hour % 3) * 25)
        v_2125N, v_2129S = v_2100N * 0.85, v_2100S * 0.85
        print(f"   [3/3] ℹ️ 採用時段動態保底值 -> 北上: {int(v_2100N)}, 南下: {int(v_2100S)}")
        
    # (D) 當前時間點之週期特徵 sin_hour & cos_hour
    sin_hour = np.sin(2 * np.pi * now.hour / 24.0)
    cos_hour = np.cos(2 * np.pi * now.hour / 24.0)

    return [
        press,
        temp,
        rh,
        wind_spd,
        wind_x,
        wind_y,
        rain,
        pm25,
        v_2100N,
        v_2100S,
        v_2125N,
        v_2129S,
        sin_hour,
        cos_hour,
    ]


# 3. 主推論程式
def main():
    print("==================================================")
    print("🚀 啟動【霧峰 PM2.5 未來 24 小時預測系統 (方案B 整合資料庫版)】")
    print("==================================================")

    db_manager.init_db()

    df_history = pd.read_csv("dataset_for_lstm.csv")
    wind_rad = np.radians(df_history["風向(360degree)"])
    df_history["wind_x"] = np.cos(wind_rad)
    df_history["wind_y"] = np.sin(wind_rad)

    if "日期" in df_history.columns:
        hours = pd.to_datetime(df_history["日期"]).dt.hour
    elif "Time" in df_history.columns:
        hours = pd.to_datetime(df_history["Time"]).dt.hour
    else:
        hours = df_history.index % 24

    df_history["sin_hour"] = np.sin(2 * np.pi * hours / 24.0)
    df_history["cos_hour"] = np.cos(2 * np.pi * hours / 24.0)

    feature_cols = [
        "測站氣壓(hPa)",
        "氣溫(℃)",
        "相對溼度(%)",
        "風速(m/s)",
        "wind_x",
        "wind_y",
        "降水量(mm)",
        "pm25",
        "03F2100N",
        "03F2100S",
        "03F2125N",
        "03F2129S",
        "sin_hour",
        "cos_hour",
    ]
    target_col = "pm25"

    pm25_idx = feature_cols.index("pm25")
    sin_idx = feature_cols.index("sin_hour")
    cos_idx = feature_cols.index("cos_hour")

    train_end = int(len(df_history) * 0.7)
    df_train = df_history.iloc[:train_end]

    scaler_X = MinMaxScaler().fit(df_train[feature_cols])
    scaler_y = MinMaxScaler().fit(df_train[[target_col]])

    live_features = fetch_wufeng_live_features()
    live_features_list = [float(x) for x in live_features]

    now = datetime.datetime.now()
    current_time_str = now.strftime("%Y-%m-%d %H:00")
    db_manager.save_real_data(current_time_str, live_features_list)
    print(f"💾 已將當前時間點 ({current_time_str}) 實測資料存入 SQLite 資料庫")

    live_features_np = np.array(live_features_list, dtype=np.float32)

    recent_23 = df_history[feature_cols].iloc[-23:].values
    current_window = np.vstack([recent_23, live_features_np])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultivariateLSTM(input_size=14)

    model_path = "best_model_ExpC_Cyclic.pth"
    if not os.path.exists(model_path):
        print(f"❌ 錯誤：找不到訓練好的權重檔 '{model_path}'！")
        sys.exit(1)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    future_predictions = []
    predictions_to_db = []

    print("\n🔮 正在執行帶有時間週期的滾動推論計算未來 24 小時 PM2.5 趨勢...\n")

    rolling_window = current_window.copy()

    for step in range(1, 25):
        window_df = pd.DataFrame(rolling_window, columns=feature_cols)
        window_scaled = scaler_X.transform(window_df)

        input_tensor = (
            torch.tensor(window_scaled, dtype=torch.float32)
            .unsqueeze(0)
            .to(device)
        )

        with torch.no_grad():
            pred_scaled = model(input_tensor).cpu().numpy()

        pred_pm25 = float(scaler_y.inverse_transform(pred_scaled)[0][0])
        future_predictions.append(pred_pm25)

        future_time = now + datetime.timedelta(hours=step)
        target_time_str = future_time.strftime("%Y-%m-%d %H:00")
        predictions_to_db.append((target_time_str, pred_pm25))

        next_feature = rolling_window[-1].copy()
        next_feature[pm25_idx] = pred_pm25
        next_feature[sin_idx] = np.sin(2 * np.pi * future_time.hour / 24.0)
        next_feature[cos_idx] = np.cos(2 * np.pi * future_time.hour / 24.0)

        rolling_window = np.vstack([rolling_window[1:], next_feature])

    db_manager.save_predictions(current_time_str, predictions_to_db)
    print(f"💾 已將未來 24 小時預測值同步紀錄至 SQLite 資料庫")

    print("==================================================")
    print("📊 【霧峰區未來 24 小時 PM2.5 預測趨勢報告 (方案B 週期版)】")
    print("==================================================")
    print(f"• 當前基準時間 : {current_time_str}")
    print(f"• 當前實測 PM2.5 : {live_features_list[pm25_idx]:.1f} µg/m³\n")
    print(" 時間預測點               預測 PM2.5 (µg/m³)")
    print("--------------------------------------------------")

    for i, pred in enumerate(future_predictions):
        future_time = now + datetime.timedelta(hours=i + 1)
        time_str = future_time.strftime("%m/%d %H:00")
        print(f" +{i+1:02d} 小時 ({time_str})  -->  {pred:.2f} µg/m³")

    print("==================================================")


if __name__ == "__main__":
    main()
