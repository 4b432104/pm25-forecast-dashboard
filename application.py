import datetime
import io
import os
import re
import sys
import urllib3
from zoneinfo import ZoneInfo
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

# 引入 DB 操作模組
import db_manager

# 關閉 SSL 憑證警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# API 金鑰設定
CWA_API_KEY = "CWA-F6B5F348-77D8-4EA8-8874-FBA50E6191DE"
MOENV_API_KEY = "5ae4f1a2-b6e6-4b79-82c8-0c84d694b7a7"

# Cloudflare Proxy 轉接頭網址
CF_PROXY_URL = "https://steep-wood-cf94.4b432104.workers.dev"


# 1. 定義 Direct Multi-Step LSTM + Self-Attention 模型架構
class AttentionMultiStepLSTM(nn.Module):
    def __init__(self, input_size=17, hidden_size=64, num_layers=2, output_steps=24):
        super(AttentionMultiStepLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1,
        )
        self.attn = nn.Linear(hidden_size, 1)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_steps)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        out = self.fc1(context)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 2. 車流量專用擷取函式 (移植舊版 12 份 CSV 累加與解析邏輯，並產出 Debug Log)
def fetch_m03a_traffic_from_freeway():
    target_gantry = ["03F2100N", "03F2100S", "03F2125N", "03F2129S"]
    traffic_dict = {g: 0.0 for g in target_gantry}
    debug_logs = []

    taipei_tz = ZoneInfo("Asia/Taipei")
    now = datetime.datetime.now(taipei_tz)

    # 取得當前時間的「上一個整點」(例：11:38 -> 10:00)
    current_hour_start = now.replace(minute=0, second=0, microsecond=0)
    prev_hour_start = current_hour_start - datetime.timedelta(hours=1)

    debug_logs.append("🚗 [DEBUG] 開始經由 Cloudflare 抓取高公局 TDCS M03A 車流...")
    debug_logs.append(
        f"   📅 目標整點時段: {prev_hour_start.strftime('%Y-%m-%d %H:00')} ~ {prev_hour_start.strftime('%H:55')}"
    )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            " (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
    }

    success_count = 0
    session = requests.Session()
    session.headers.update(headers)

    # 抓取該整點小時內的 12 個 5 分鐘 CSV 檔案
    for minute_offset in range(0, 60, 5):
        target_dt = prev_hour_start + datetime.timedelta(minutes=minute_offset)
        ymd = target_dt.strftime("%Y%m%d")
        hh = target_dt.strftime("%H")
        mm = target_dt.strftime("%M")

        freeway_url = f"https://tisvcloud.freeway.gov.tw/history/TDCS/M03A/{ymd}/{hh}/TDCS_M03A_{ymd}_{hh}{mm}00.csv"
        proxy_request_url = f"{CF_PROXY_URL}/?url={freeway_url}"

        try:
            resp = session.get(proxy_request_url, timeout=8, verify=False)
            if resp.status_code == 200 and len(resp.text) > 100:
                csv_data = io.StringIO(resp.text)
                df_temp = pd.read_csv(csv_data, header=None)

                if len(df_temp.columns) >= 5:
                    df_temp.columns = [
                        "TimeInterval",
                        "GantryID",
                        "Direction",
                        "VehicleType",
                        "Volume",
                    ] + list(df_temp.columns[5:])
                    
                    df_wufeng = df_temp[df_temp["GantryID"].isin(target_gantry)]

                    matched_lines = len(df_wufeng)
                    for gantry in target_gantry:
                        vol = df_wufeng[df_wufeng["GantryID"] == gantry]["Volume"].sum()
                        traffic_dict[gantry] += float(vol)

                    success_count += 1
                    debug_logs.append(
                        f"   📄 [{hh}:{mm}] 成功讀取 {len(df_temp)} 行，命中霧峰門架 {matched_lines} 筆"
                    )
            else:
                debug_logs.append(f"   ⚠️ 抓取失敗 [{ymd} {hh}:{mm}] HTTP {resp.status_code}")
        except Exception as e:
            debug_logs.append(f"   ⚠️ 連線異常 [{ymd} {hh}:{mm}]: {e}")

    debug_logs.append(f"📊 下載結果: 成功取得 {success_count}/12 個時段 CSV 資料")

    if success_count > 0:
        if success_count < 12:
            scale_factor = 12.0 / success_count
            for gantry in target_gantry:
                traffic_dict[gantry] = round(traffic_dict[gantry] * scale_factor)

        debug_logs.append(
            f"   🎉 [SUCCESS] 上一整點小時 ({prev_hour_start.strftime('%H:00')}) 累計總車流量: {traffic_dict}"
        )
    else:
        debug_logs.append("   ⚠️ 無法取得 TDCS CSV 資料，將採用動態離尖峰預設值保底")

    return traffic_dict, debug_logs


# 3. 自動化擷取【霧峰區】即時 17 項特徵
def fetch_wufeng_live_features(df_history=None):
    debug_logs = []
    taipei_tz = ZoneInfo("Asia/Taipei")
    now = datetime.datetime.now(taipei_tz)
    debug_logs.append(f"🔍 [Debug] 開始執行即時數據擷取任務，系統時間: {now.strftime('%Y-%m-%d %H:%M:%S')}")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            " (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
    }

    # (A) 霧峰 PM2.5
    pm25 = None
    try:
        url_epb_table = "https://taqm.epb.taichung.gov.tw/TQAMPM25table.ASPX"
        res_epb = requests.get(url_epb_table, headers=headers, timeout=10, verify=False)
        res_epb.encoding = "utf-8"
        soup = BeautifulSoup(res_epb.text, "html.parser")
        all_cells = [tag.text.strip() for tag in soup.find_all(["td", "th", "a"])]

        for idx, text in enumerate(all_cells):
            if "霧峰" in text and idx + 1 < len(all_cells):
                val_str = all_cells[idx + 1]
                if val_str.isdigit() or re.match(r"^\d+(\.\d+)?$", val_str):
                    pm25 = float(val_str)
                    debug_logs.append(f"   [1/3] ✅ 【臺中環保局】霧峰站即時 PM2.5 成功解析: {pm25} µg/m³")
                    break
    except Exception as e:
        debug_logs.append(f"   [1/3] ℹ️ 臺中環保局網頁爬取跳過/失敗: {e}")

    if pm25 is None:
        try:
            url_dali = f"https://data.moenv.gov.tw/api/v2/aqx_p_432?api_key={MOENV_API_KEY}&limit=5&format=json&filters=sitename,eq,大里"
            res_dali = requests.get(url_dali, headers=headers, timeout=10, verify=False).json()
            recs = res_dali.get("records", []) if isinstance(res_dali, dict) else res_dali
            if recs:
                val = recs[0].get("pm25") or recs[0].get("pm2.5")
                if val:
                    pm25 = float(val)
                    debug_logs.append(f"   [1/3] ✅ 採用鄰近【大里標準站】即時 PM2.5: {pm25} µg/m³")
        except Exception as e:
            debug_logs.append(f"   [1/3] ℹ️ 大里站 API 跳過: {e}")

    if pm25 is None:
        pm25 = float(df_history["pm25"].iloc[-1]) if df_history is not None else 15.0
        debug_logs.append(f"   [1/3] ⚠️ PM2.5 採用歷史/保底數值: {pm25} µg/m³")

    # (B) 霧峰氣象
    press, temp, rh, wind_spd, wind_dir, rain = 1008.5, 24.5, 75.0, 1.8, 180.0, 0.0
    try:
        url_cwa = f"https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0001-001?Authorization={CWA_API_KEY}&StationName=霧峰"
        res_cwa = requests.get(url_cwa, headers=headers, timeout=10, verify=False).json()
        if isinstance(res_cwa, dict) and res_cwa.get("records") and res_cwa["records"].get("Station"):
            station_data = res_cwa["records"]["Station"][0]
            obs_time_str = station_data.get("ObsTime", {}).get("DateTime", "未知時間")
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

            debug_logs.append(
                f"   [2/3] ✅ 成功取得【氣象署霧峰站】 (時間: {obs_time_str}): 氣溫 {temp}℃, 濕度 {rh}%, 氣壓 {press}hPa"
            )
    except Exception as e:
        debug_logs.append(f"   [2/3] ⚠️ 氣象署 API 解析失敗，採用保底數值: {e}")

    # (C) 國道 3 號車流量 (採用舊版穩定邏輯)
    traffic_dict, traffic_logs = fetch_m03a_traffic_from_freeway()
    debug_logs.extend(traffic_logs)

    cur_h = now.hour
    if 7 <= cur_h <= 9 or 17 <= cur_h <= 19:
        default_v = (1200.0, 1300.0, 900.0, 850.0)
    elif 23 <= cur_h or cur_h <= 5:
        default_v = (200.0, 220.0, 150.0, 140.0)
    else:
        default_v = (650.0, 700.0, 500.0, 480.0)

    v_2100N = traffic_dict.get("03F2100N", 0.0) or default_v[0]
    v_2100S = traffic_dict.get("03F2100S", 0.0) or default_v[1]
    v_2125N = traffic_dict.get("03F2125N", 0.0) or default_v[2]
    v_2129S = traffic_dict.get("03F2129S", 0.0) or default_v[3]

    debug_logs.append(
        f"   [3/3] ✅ 車流計算完成 (上一小時總計): 2100N={v_2100N:.0f}, 2100S={v_2100S:.0f}, 2125N={v_2125N:.0f}, 2129S={v_2129S:.0f}"
    )

    # (D) 計算一階差分、二階加速度與週期特徵
    last_pm25 = float(df_history["pm25"].iloc[-1]) if df_history is not None else pm25
    last_pm25_diff = float(df_history["pm25_diff"].iloc[-1]) if df_history is not None else 0.0
    pm25_diff = pm25 - last_pm25
    pm25_accel = pm25_diff - last_pm25_diff

    current_traffic = v_2100N + v_2100S + v_2125N + v_2129S
    last_traffic = (
        float(df_history[["03F2100N", "03F2100S", "03F2125N", "03F2129S"]].iloc[-1].sum())
        if df_history is not None
        else current_traffic
    )
    last_traffic_diff = float(df_history["traffic_diff"].iloc[-1]) if df_history is not None else 0.0
    traffic_diff = current_traffic - last_traffic
    traffic_accel = traffic_diff - last_traffic_diff

    hour_sin = np.sin(2 * np.pi * cur_h / 24.0)
    hour_cos = np.cos(2 * np.pi * cur_h / 24.0)

    features = [
        press,
        temp,
        rh,
        wind_spd,
        wind_dir,
        rain,
        pm25,
        pm25_diff,
        pm25_accel,
        traffic_diff,
        traffic_accel,
        v_2100N,
        v_2100S,
        v_2125N,
        v_2129S,
        hour_sin,
        hour_cos,
    ]

    return features, debug_logs


# 4. 主推論程式
def main():
    print("==================================================")
    print("🚀 啟動【霧峰 PM2.5 未來 24 小時 Attention 多步預測系統】")
    print("==================================================")

    db_manager.init_db()
    df_history = pd.read_csv("dataset_for_lstm.csv")

    df_history["dt"] = pd.to_datetime(
        df_history["time"] if "time" in df_history.columns else df_history.iloc[:, 0]
    )
    df_history["hour"] = df_history["dt"].dt.hour
    df_history["hour_sin"] = np.sin(2 * np.pi * df_history["hour"] / 24.0)
    df_history["hour_cos"] = np.cos(2 * np.pi * df_history["hour"] / 24.0)

    df_history["pm25_diff"] = df_history["pm25"].diff().fillna(0)
    df_history["pm25_accel"] = df_history["pm25_diff"].diff().fillna(0)

    traffic_sum = df_history[["03F2100N", "03F2100S", "03F2125N", "03F2129S"]].sum(axis=1)
    df_history["traffic_diff"] = traffic_sum.diff().fillna(0)
    df_history["traffic_accel"] = df_history["traffic_diff"].diff().fillna(0)

    feature_cols = [
        "測站氣壓(hPa)",
        "氣溫(℃)",
        "相對溼度(%)",
        "風速(m/s)",
        "風向(360degree)",
        "降水量(mm)",
        "pm25",
        "pm25_diff",
        "pm25_accel",
        "traffic_diff",
        "traffic_accel",
        "03F2100N",
        "03F2100S",
        "03F2125N",
        "03F2129S",
        "hour_sin",
        "hour_cos",
    ]
    target_col = "pm25"

    scaler_X = StandardScaler().fit(df_history[feature_cols])
    scaler_y = StandardScaler().fit(df_history[[target_col]])

    live_features, logs = fetch_wufeng_live_features(df_history)
    for log in logs:
        print(log)

    live_features_list = [float(x) for x in live_features]
    
    taipei_tz = ZoneInfo("Asia/Taipei")
    now = datetime.datetime.now(taipei_tz)
    base_time = now.replace(minute=0, second=0, microsecond=0)
    current_time_str = base_time.strftime("%Y-%m-%d %H:00")

    db_manager.save_real_data(current_time_str, live_features_list)

    recent_23 = df_history[feature_cols].iloc[-23:].values
    current_window = np.vstack([recent_23, np.array(live_features_list, dtype=np.float32)])

    window_scaled = scaler_X.transform(pd.DataFrame(current_window, columns=feature_cols))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = torch.tensor(window_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    model = AttentionMultiStepLSTM(input_size=17, hidden_size=64, output_steps=24).to(device)

    model_path = "best_lstm_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        with torch.no_grad():
            preds_delta_scaled = model(input_tensor).cpu().numpy()[0]

        base_pm25_scaled = window_scaled[-1, 6]
        pred_y_scaled = base_pm25_scaled + preds_delta_scaled
        preds_pm25 = scaler_y.inverse_transform(pred_y_scaled.reshape(-1, 1)).flatten()

        predictions_to_db = []
        for i, pred in enumerate(preds_pm25):
            future_time = base_time + datetime.timedelta(hours=i + 1)
            target_time_str = future_time.strftime("%Y-%m-%d %H:00")
            pred_val = max(0.0, float(pred))
            predictions_to_db.append((target_time_str, pred_val, i + 1))

        db_manager.save_predictions(current_time_str, predictions_to_db)
        print("💾 已成功將預測數值寫入 SQLite 資料庫")


if __name__ == "__main__":
    main()
