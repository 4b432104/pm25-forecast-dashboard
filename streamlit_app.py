import datetime
import os
import sqlite3
import traceback
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from sklearn.preprocessing import StandardScaler

# 強制鎖定工作目錄
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

import application
import db_manager

st.set_page_config(
    page_title="台中霧峰 PM2.5 未來 24 小時預測系統",
    layout="wide",
    page_icon="🌬️",
)


@st.cache_data(ttl=3600)  # 快取 1 小時 (3600 秒)
def dynamic_predict_24h(current_hour, live_features_list):
    """根據當前動態基準時間與即時特徵，使用 Attention LSTM 模型直接推論未來 24 小時數值"""
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

    pm25_idx = feature_cols.index("pm25")

    # 讀取歷史數據檔建立 Scaler 與歷史滾動視窗
    csv_path = "dataset_for_lstm.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError("找不到 dataset_for_lstm.csv")

    df_history = pd.read_csv(csv_path)

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

    # 計算 24 小時歷史平均 Profile
    hourly_profile = df_history.groupby("hour")[feature_cols].mean()

    scaler_X = StandardScaler().fit(df_history[feature_cols])
    scaler_y = StandardScaler().fit(df_history[[target_col]])

    # 準備模型輸入視窗 (過去 23 小時 + 當前第 24 小時即時值)
    live_features_np = np.array(live_features_list, dtype=np.float32)
    recent_23 = df_history[feature_cols].iloc[-23:].values
    rolling_window = np.vstack([recent_23, live_features_np])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = application.AttentionMultiStepLSTM(
        input_size=17, hidden_size=64, output_steps=24
    ).to(device)

    model_path = "best_lstm_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型權重 `{model_path}`")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    current_hour_naive = current_hour.replace(tzinfo=None)

    # 模型直接推論 24 步
    window_df = pd.DataFrame(rolling_window, columns=feature_cols)
    window_scaled = scaler_X.transform(window_df)

    input_tensor = (
        torch.tensor(window_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    )

    with torch.no_grad():
        preds_delta_scaled = model(input_tensor).cpu().numpy()[0]

    base_pm25_scaled = window_scaled[-1, pm25_idx]
    pred_y_scaled = base_pm25_scaled + preds_delta_scaled
    preds_pm25 = scaler_y.inverse_transform(pred_y_scaled.reshape(-1, 1)).flatten()

    future_predictions = []
    future_times = []

    for step in range(1, 25):
        pred_val = max(0.0, float(preds_pm25[step - 1]))
        future_predictions.append(round(pred_val, 2))

        future_time = current_hour_naive + datetime.timedelta(hours=step)
        future_times.append(future_time)

    df_result = pd.DataFrame(
        {"target_datetime": future_times, "predicted_pm25": future_predictions}
    )
    return df_result


def get_fallback_features(prev_hour, df_history=None):
    """取得前一小時 (prev_hour) 的車流與氣象備援數據 (SQLite 或 靜態動態預設值)"""
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

    # 1. 嘗試從 SQLite 資料庫讀取最近一次真實紀錄
    try:
        conn = sqlite3.connect("pm25_forecast.db")
        df_db = pd.read_sql(
            "SELECT * FROM realtime_logs ORDER BY timestamp DESC LIMIT 1", conn
        )
        conn.close()
        if not df_db.empty and all(col in df_db.columns for col in feature_cols):
            print("   ℹ️ 成功使用 SQLite 資料庫內的最後紀錄做為備援數據", flush=True)
            return df_db[feature_cols].iloc[0].tolist()
    except Exception as err:
        print(f"   ⚠️ 讀取 SQLite 備援失敗: {err}", flush=True)

    # 2. 若 SQLite 沒資料，回傳安全的動態預設值
    print("   ℹ️ 使用靜態動態預設值做為備援數據", flush=True)
    h = prev_hour.hour
    sin_h = float(np.sin(2 * np.pi * h / 24.0))
    cos_h = float(np.cos(2 * np.pi * h / 24.0))

    return [
        1008.5,  # 測站氣壓
        24.5,    # 氣溫
        75.0,    # 相對溼度
        1.8,     # 風速
        180.0,   # 風向
        0.0,     # 降水量
        15.0,    # pm25
        0.0,     # pm25_diff
        0.0,     # pm25_accel
        0.0,     # traffic_diff
        0.0,     # traffic_accel
        620.0,   # 03F2100N
        580.0,   # 03F2100S
        510.0,   # 03F2125N
        490.0,   # 03F2129S
        sin_h,   # hour_sin
        cos_h,   # hour_cos
    ]


def main():
    st.title("🌬️ 台中市霧峰區 PM2.5 未來 24 小時預測系統")
    st.caption(
        "結合大氣氣象、即時環測與國道 3 號車流量之 Attention LSTM 深度學習趨勢預測儀表板"
    )

    st.sidebar.header("⚙️ 系統狀態與設定")
    if st.sidebar.button("🔄 刷新即時監測數據"):
        st.cache_data.clear()
        st.rerun()

    # 1. 精準計算【基準時間】與【前一小時車流區間】
    taipei_tz = ZoneInfo("Asia/Taipei")
    now = datetime.datetime.now(taipei_tz)

    current_hour = now.replace(minute=0, second=0, microsecond=0)
    prev_hour = current_hour - datetime.timedelta(hours=1)

    current_time_str = current_hour.strftime("%Y-%m-%d %H:00")
    prev_time_str = prev_hour.strftime("%H:00")
    traffic_range_str = f"{prev_time_str} ~ {current_hour.strftime('%H:00')}"

    st.sidebar.write(f"🕒 **當前基準時間**: {current_time_str}")
    st.sidebar.write(f"🚗 **車流統計區間**: {traffic_range_str}")

    # 2. 擷取即時監測與前一小時數據
    live_features_list = [0.0] * 17
    with st.spinner(
        f"📡 正在擷取霧峰即時監測與車流數據 ({traffic_range_str})..."
    ):
        try:
            print("\n" + "=" * 50, flush=True)
            print("🚀 【NEW VERSION 2.0】Streamlit 儀表板啟動即時預測...", flush=True)
            print("=" * 50 + "\n", flush=True)

            csv_path = "dataset_for_lstm.csv"
            df_history = pd.read_csv(csv_path) if os.path.exists(csv_path) else None

            live_features, logs = application.fetch_wufeng_live_features(df_history)
            live_features_list = [float(x) for x in live_features]
        except Exception as e:
            print("\n❌ [ERROR] fetch_wufeng_live_features 執行失敗:", flush=True)
            print(traceback.format_exc(), flush=True)

            st.warning(
                f"⚠️ 即時 API 擷取異常 ({e})，已切換至 [{traffic_range_str}] 動態備援數據。"
            )
            live_features_list = get_fallback_features(prev_hour)

    # 💡 提取 4 個門架流量並計算「北上總和」、「南下總和」與「4門架全區總車流量」
    # 特徵欄位順序: 
    # [11]: 03F2100N, [12]: 03F2100S, [13]: 03F2125N, [14]: 03F2129S
    gantry_2100N = live_features_list[11]
    gantry_2100S = live_features_list[12]
    gantry_2125N = live_features_list[13]
    gantry_2129S = live_features_list[14]

    traffic_north = gantry_2100N + gantry_2125N  # 北上門架加總
    traffic_south = gantry_2100S + gantry_2129S  # 南下門架加總
    traffic_total = traffic_north + traffic_south  # 4 個門架雙向總流量

    # 3. 即時數據 Summary
    st.subheader(f"📊 即時監測與車流 Summary ({traffic_range_str} 累積值)")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("即時 PM2.5", f"{live_features_list[6]:.1f} µg/m³")
    col2.metric("氣溫", f"{live_features_list[1]:.1f} ℃")
    col3.metric("相對濕度", f"{live_features_list[2]:.0f} %")
    col4.metric("風速", f"{live_features_list[3]:.1f} m/s")

    st.markdown("##### 🚗 國道 3 號霧峰段總車流量統計 (4門架加總)")
    col_t1, col_t2, col_t3 = st.columns(3)
    col_t1.metric("國道 3 號 (北上車流總和)", f"{int(traffic_north):,} 輛")
    col_t2.metric("國道 3 號 (南下車流總和)", f"{int(traffic_south):,} 輛")
    col_t3.metric("國道 3 號 (雙向全區總車流量)", f"{int(traffic_total):,} 輛")

    st.markdown("---")
    st.subheader("🔮 未來 24 小時 PM2.5 預測趨勢圖")

    # 4. 推論未來 24 小時
    with st.spinner("🔮 正在根據最新基準時間即時計算未來 24 小時趨勢..."):
        try:
            df_pred = dynamic_predict_24h(current_hour, live_features_list)
        except Exception as e:
            st.error(f"❌ 模型推論發生錯誤: {e}")
            return

    # 5. 繪製圖表
    current_hour_naive = current_hour.replace(tzinfo=None)
    end_time_24h = current_hour_naive + datetime.timedelta(hours=24)

    fig = go.Figure()

    # 基準時間實測點
    fig.add_trace(
        go.Scatter(
            x=[current_hour_naive],
            y=[live_features_list[6]],
            mode="markers",
            name="當前實測值 (基準時間)",
            marker=dict(color="red", size=12),
        )
    )

    # 未來 24 小時預測折線
    fig.add_trace(
        go.Scatter(
            x=df_pred["target_datetime"],
            y=df_pred["predicted_pm25"],
            mode="lines+markers",
            name="LSTM 預測 PM2.5 (µg/m³)",
            line=dict(color="#0083B0", width=3),
            marker=dict(size=6),
        )
    )

    # 標準參考線
    fig.add_hline(
        y=15,
        line_dash="dash",
        line_color="orange",
        annotation_text="WHO 24小時建議值 (15 µg/m³)",
    )
    fig.add_hline(
        y=35.5,
        line_dash="dash",
        line_color="red",
        annotation_text="環境部橘色提醒臨界點 (35.5 µg/m³)",
    )

    fig.update_layout(
        xaxis=dict(
            title="預測時間點",
            type="date",
            tickformat="%m/%d %H:00",
            range=[current_hour_naive, end_time_24h],
        ),
        yaxis_title="PM2.5 濃度 (µg/m³)",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=30, b=20),
        height=450,
    )

    st.plotly_chart(fig, use_container_width=True)

    # 6. 未來 24 小時明細表格
    st.subheader("📋 未來 24 小時預測數值明細")
    df_display = pd.DataFrame(
        {
            "預測時間點": df_pred["target_datetime"].dt.strftime(
                "%m/%d %H:00"
            ),
            "預測 PM2.5 (µg/m³)": df_pred["predicted_pm25"].round(2),
        }
    )
    df_display = df_display.astype(str)
    st.dataframe(df_display, use_container_width=True)


if __name__ == "__main__":
    main()
