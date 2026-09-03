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

# Cloudflare Proxy 轉接頭網址
CF_PROXY_URL = "https://steep-wood-cf94.4b432104.workers.dev"


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


def fetch_m03a_traffic_from_freeway():
    """透過 Cloudflare Proxy 從高公局 TDCS M03A 抓取上一個完整整點小時 4 個門架累計車流量"""
    target_gantry = ["03F2100N", "03F2100S", "03F2125N", "03F2129S"]
    traffic_dict = {g: 0.0 for g in target_gantry}

    taipei_tz = ZoneInfo("Asia/Taipei")
    now = datetime.datetime.now(taipei_tz)

    # 取得當前時間的「上一個整點」(例如 11:38 -> 10:00)
    current_hour_start = now.replace(minute=0, second=0, microsecond=0)
    prev_hour_start = current_hour_start - datetime.timedelta(hours=1)

    print(
        f"\n🔍 [DEBUG] 開始經由 Cloudflare 抓取 TDCS M03A 車流...",
        flush=True,
    )
    print(
        f"   📅 目標整點時段: {prev_hour_start.strftime('%Y-%m-%d %H:00')} ~ {prev_hour_start.strftime('%H:55')}",
        flush=True,
    )

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
        "Accept": "*/*",
    }

    success_count = 0
    session = requests.Session()
    session.headers.update(headers)

    # 抓取該小時內的 12 個 5 分鐘檔案
    for minute_offset in range(0, 60, 5):
        target_dt = prev_hour_start + datetime.timedelta(minutes=minute_offset)
        ymd = target_dt.strftime("%Y%m%d")
        hh = target_dt.strftime("%H")
        mm = target_dt.strftime("%M")

        # 原始高公局網址
        freeway_url = f"https://tisvcloud.freeway.gov.tw/history/TDCS/M03A/{ymd}/{hh}/TDCS_M03A_{ymd}_{hh}{mm}00.csv"

        # 改走 Cloudflare Worker Proxy 轉載
        proxy_request_url = f"{CF_PROXY_URL}/?url={freeway_url}"

        try:
            resp = session.get(proxy_request_url, timeout=10, verify=False)
            if resp.status_code == 200:
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

                    for gantry in target_gantry:
                        vol = df_wufeng[df_wufeng["GantryID"] == gantry][
                            "Volume"
                        ].sum()
                        traffic_dict[gantry] += float(vol)

                    success_count += 1
            else:
                print(
                    f"   ⚠️ 抓取失敗 [{ymd} {hh}:{mm}] HTTP {resp.status_code}",
                    flush=True,
                )
        except Exception as e:
            print(f"   ⚠️ 連線逾時/異常 [{ymd} {hh}:{mm}]: {e}", flush=True)

    print(
        f"   📊 下載結果: 成功取得 {success_count}/12 個時段資料", flush=True
    )

    if success_count > 0:
        if success_count < 12:
            scale_factor = 12.0 / success_count
            for gantry in target_gantry:
                traffic_dict[gantry] = round(
                    traffic_dict[gantry] * scale_factor
                )

        print(
            f"   🎉 [SUCCESS] 上一整點小時 ({prev_hour_start.strftime('%H:00')}) 累計車流量: {traffic_dict}",
            flush=True,
        )
    else:
        print(
            f"   ⚠️ [WARNING] 無法取得 TDCS CSV 資料，將採用動態離尖峰預設值保底",
            flush=True,
        )

    return traffic_dict


# 2. 自動化擷取【霧峰區】即時 14 項特徵
def fetch_wufeng_live_features():
    print(
        "\n📡 開始連線擷取【台中霧峰區】三大類即時自變數...", flush=True
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        ),
        "Accept": "*/*",
    }

    # (A) 霧峰 PM2.5
    pm25 = None
    try:
        url_epb_table = "https://taqm.epb.taichung.gov.tw/TQAMPM25table.ASPX"
        res_epb = requests.get(
            url_epb_table, headers=headers, timeout=10, verify=False
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
                        f"   [1/3] ✅ 精準解析成功！【臺中環保局】霧峰站即時 PM2.5: {pm25} µg/m³",
                        flush=True,
                    )
                    break
    except Exception as e:
        print(f"   [1/3] ℹ️ 網頁爬取跳過: {e}", flush=True)

    if pm25 is None:
        try:
            url_dali = f"https://data.moenv.gov.tw/api/v2/aqx_p_432?api_key={MOENV_API_KEY}&limit=5&format=json&filters=sitename,eq,大里"
            res_dali = requests.get(
                url_dali, headers=headers, timeout=10, verify=False
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
                        f"   [1/3] ✅ 採用鄰近【大里標準站】即時 PM2.5: {pm25} µg/m³",
                        flush=True,
                    )
        except Exception:
            pass

    if pm25 is None:
        pm25 = 15.0
        print(
            f"   [1/3] ℹ️ 採用系統預設 PM2.5 數值: {pm25} µg/m³", flush=True
        )

    # (B) 霧峰氣象
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
            url_cwa, headers=headers, timeout=10, verify=False
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
            if "Now" in station_elem and isinstance(
                station_elem["Now"], dict
            ):
                rain = safe_float(
                    station_elem["Now"].get("Precipitation"), 0.0
                )

            print(
                f"   [2/3] ✅ 成功取得【氣象署霧峰站】氣象 (觀測時間: {obs_time_str}): 氣溫 {temp}℃, 濕度 {rh}%",
                flush=True,
            )
    except Exception as e:
        print(
            f"   [2/3] ⚠️ 氣象署 API 解析失敗，採用保底數值: {e}", flush=True
        )

    wind_rad = np.radians(wind_dir)
    wind_x = np.cos(wind_rad)
    wind_y = np.sin(wind_rad)

    # (C) 國道 3 號車流量 (經由 Cloudflare 代理抓取)
    traffic_dict = fetch_m03a_traffic_from_freeway()

    # 動態時間預設值 (避免完全沒有資料時數值不合理)
    taipei_tz = ZoneInfo("Asia/Taipei")
    now_taipei = datetime.datetime.now(taipei_tz)
    cur_h = now_taipei.hour

    if 7 <= cur_h <= 9 or 17 <= cur_h <= 19:
        default_v = (1200.0, 1300.0, 900.0, 850.0)  # 尖峰
    elif 23 <= cur_h or cur_h <= 5:
        default_v = (200.0, 220.0, 150.0, 140.0)  # 離峰
    else:
        default_v = (650.0, 700.0, 500.0, 480.0)  # 一般

    v_2100N = traffic_dict.get("03F2100N", 0.0) or default_v[0]
    v_2100S = traffic_dict.get("03F2100S", 0.0) or default_v[1]
    v_2125N = traffic_dict.get("03F2125N", 0.0) or default_v[2]
    v_2129S = traffic_dict.get("03F2129S", 0.0) or default_v[3]

    print(f"   [3/3] ✅ 國道 3 號門架 1小時車流量總計:", flush=True)
    print(
        f"         └─ 03F2100N (北上): {int(v_2100N)} 輛 | 03F2100S (南下): {int(v_2100S)} 輛",
        flush=True,
    )
    print(
        f"         └─ 03F2125N: {int(v_2125N)} 輛 | 03F2129S: {int(v_2129S)} 輛\n",
        flush=True,
    )

    # (D) 當前時間點之週期特徵
    sin_hour = np.sin(2 * np.pi * cur_h / 24.0)
    cos_hour = np.cos(2 * np.pi * cur_h / 24.0)

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
    print("==================================================", flush=True)
    print("🚀 【NEW VERSION 2.0】啟動預測系統...", flush=True)
    print("==================================================", flush=True)
    print(
        "🚀 啟動【霧峰 PM2.5 未來 24 小時預測系統 (方案B 整合資料庫版)】",
        flush=True,
    )
    print("==================================================", flush=True)

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

    taipei_tz = ZoneInfo("Asia/Taipei")
    now = datetime.datetime.now(taipei_tz)
    current_time_str = now.strftime("%Y-%m-%d %H:00")
    db_manager.save_real_data(current_time_str, live_features_list)
    print(
        f"💾 已將當前時間點 ({current_time_str}) 實測資料存入 SQLite 資料庫",
        flush=True,
    )

    live_features_np = np.array(live_features_list, dtype=np.float32)

    recent_23 = df_history[feature_cols].iloc[-23:].values
    current_window = np.vstack([recent_23, live_features_np])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultivariateLSTM(input_size=14)

    model_path = "best_model_ExpC_Cyclic.pth"
    if not os.path.exists(model_path):
        print(f"❌ 錯誤：找不到訓練好的權重檔 '{model_path}'！", flush=True)
        print(
            "💡 請先執行 train_cyclic_model.py 完成訓練後再執行本推論腳本。",
            flush=True,
        )
        sys.exit(1)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    future_predictions = []
    predictions_to_db = []

    print(
        "\n🔮 正在執行帶有時間週期的滾動推論計算未來 24 小時 PM2.5 趨勢...\n",
        flush=True,
    )

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
    print("💾 已將未來 24 小時預測值同步紀錄至 SQLite 資料庫", flush=True)

    print("==================================================", flush=True)
    print("📊 【霧峰區未來 24 小時 PM2.5 預測趨勢報告 (方案B 週期版)】", flush=True)
    print("==================================================", flush=True)
    print(f"• 當前基準時間 : {current_time_str}", flush=True)
    print(
        f"• 當前實測 PM2.5 : {live_features_list[pm25_idx]:.1f} µg/m³\n",
        flush=True,
    )
    print(" 時間預測點                預測 PM2.5 (µg/m³)", flush=True)
    print("--------------------------------------------------", flush=True)

    for i, pred in enumerate(future_predictions):
        future_time = now + datetime.timedelta(hours=i + 1)
        time_str = future_time.strftime("%m/%d %H:00")
        print(f" +{i+1:02d} 小時 ({time_str})  -->  {pred:.2f} µg/m³", flush=True)

    print("==================================================", flush=True)


if __name__ == "__main__":
    main()
