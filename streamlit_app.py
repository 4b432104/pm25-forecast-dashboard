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


def get_fallback_features(prev_hour):
    """取得前一小時 (prev_hour) 的備援數據"""
    feature_cols = [
        "測站氣壓(hPa)", "氣溫(℃)", "相對溼度(%)", "風速(m/s)", "風向(360degree)",
        "降水量(mm)", "pm25", "pm25_diff", "pm25_accel", "traffic_diff",
        "traffic_accel", "03F2100N", "03F2100S", "03F2125N", "03F2129S",
        "hour_sin", "hour_cos",
    ]

    try:
        conn = sqlite3.connect("pm25_forecast.db")
        df_db = pd.read_sql(
            "SELECT * FROM realtime_logs ORDER BY timestamp DESC LIMIT 1", conn
        )
        conn.close()
        if not df_db.empty and all(col in df_db.columns for col in feature_cols):
            return df_db[feature_cols].iloc[0].tolist(), ["ℹ️ 成功使用 SQLite 資料庫紀錄作為車流與氣象備援數據"]
    except Exception as err:
        pass

    h = prev_hour.hour
    sin_h = float(np.sin(2 * np.pi * h / 24.0))
    cos_h = float(np.cos(2 * np.pi * h / 24.0))

    fallback_data = [
        1008.5, 24.5, 75.0, 1.8, 180.0, 0.0, 15.0,
        0.0, 0.0, 0.0, 0.0,
        620.0, 580.0, 510.0, 490.0,
        sin_h, cos_h
    ]
    return fallback_data, ["ℹ️ API 連線失敗，已載入動態預設車流與氣象備援數值"]


def main():
    st.title("🌬️ 台中市霧峰區 PM2.5 未來 24 小時預測系統")
    st.caption(
        "結合大氣氣象、即時環測與國道 3 號車流量之 Attention LSTM 深度學習趨勢預測儀表板"
    )

    st.sidebar.header("⚙️ 系統狀態與設定")
    if st.sidebar.button("🔄 刷新即時監測數據"):
        st.cache_data.clear()
        st.rerun()

    # 1. 計算【基準時間】與【前一小時車流區間】
    taipei_tz = ZoneInfo("Asia/Taipei")
    now = datetime.datetime.now(taipei_tz)

    current_hour = now.replace(minute=0, second=0, microsecond=0)
    prev_hour = current_hour - datetime.timedelta(hours=1)

    current_time_str = current_hour.strftime("%Y-%m-%d %H:00")
    prev_time_str = prev_hour.strftime("%H:00")
    traffic_range_str = f"{prev_time_str} ~ {current_hour.strftime('%H:00')}"

    st.sidebar.write(f"🕒 **當前基準時間**: {current_time_str}")
    st.sidebar.write(f"🚗 **車流統計區間**: {traffic_range_str}")

    # 2. 擷取即時數據與工作日誌
    live_features_list = []
    fetch_logs = []
    with st.spinner(f"📡 正在擷取霧峰即時監測與車流數據 ({traffic_range_str})..."):
        try:
            csv_path = "dataset_for_lstm.csv"
            df_history = pd.read_csv(csv_path) if os.path.exists(csv_path) else None

            live_features, fetch_logs = application.fetch_wufeng_live_features(df_history)
            live_features_list = [float(x) for x in live_features]
        except Exception as e:
            st.warning(f"⚠️ 即時 API 擷取異常 ({e})，已切換至 [{traffic_range_str}] 備援數據。")
            live_features_list, fetch_logs = get_fallback_features(prev_hour)

    # 💡 解析 4 個門架獨立車流量與計算總流量
    # 索引位置: [11]: 03F2100N, [12]: 03F2100S, [13]: 03F2125N, [14]: 03F2129S
    g_2100N = live_features_list[11]
    g_2100S = live_features_list[12]
    g_2125N = live_features_list[13]
    g_2129S = live_features_list[14]

    traffic_north = g_2100N + g_2125N
    traffic_south = g_2100S + g_2129S
    traffic_total = traffic_north + traffic_south

    # 3. 即時監測與車流 Summary 顯示
    st.subheader(f"📊 即時監測與車流 Summary ({traffic_range_str} 累積值)")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("即時 PM2.5", f"{live_features_list[6]:.1f} µg/m³")
    col2.metric("氣溫", f"{live_features_list[1]:.1f} ℃")
    col3.metric("相對濕度", f"{live_features_list[2]:.0f} %")
    col4.metric("風速", f"{live_features_list[3]:.1f} m/s")

    st.markdown("##### 🚗 國道 3 號霧峰段車流量數據")
    col_t1, col_t2, col_t3 = st.columns(3)
    col_t1.metric("北上總車流量 (2100N + 2125N)", f"{int(traffic_north):,} 輛")
    col_t2.metric("南下總車流量 (2100S + 2129S)", f"{int(traffic_south):,} 輛")
    col_t3.metric("🔥 雙向全區總車流量", f"{int(traffic_total):,} 輛")

    # ------------------------------------------------------------------
    # 📝 核心新增：工作日記與每筆門架總車流量細節
    # ------------------------------------------------------------------
    with st.expander("📝 查看車流量抓取明細與系統工作日記 (Work Log)", expanded=False):
        st.markdown(f"**⏰ 紀錄時間**：`{current_time_str}` | **統計時段**：`{traffic_range_str}`")
        
        # 門架流量明細表
        df_traffic_log = pd.DataFrame({
            "門架代碼": ["03F2100N (北上)", "03F2100S (南下)", "03F2125N (北上)", "03F2129S (南下)", "合計總流量"],
            "抓取車種範疇": ["全車種 (31,32,41,42,51)", "全車種 (31,32,41,42,51)", "全車種 (31,32,41,42,51)", "全車種 (31,32,41,42,51)", "全區雙向彙整"],
            "累積流量 (輛)": [f"{int(g_2100N):,}", f"{int(g_2100S):,}", f"{int(g_2125N):,}", f"{int(g_2129S):,}", f"👉 {int(traffic_total):,} 輛"]
        })
        st.table(df_traffic_log)

        # 系統日誌輸出
        st.markdown("**🔍 即時爬蟲與資料處理執行日誌：**")
        log_text = "\n".join(fetch_logs) if fetch_logs else "尚無日誌訊息"
        st.code(log_text, language="text")

    st.markdown("---")
    st.subheader("🔮 未來 24 小時 PM2.5 預測趨勢圖")

    # 4. 推論未來 24 小時
    with st.spinner("🔮 正在根據最新基準時間即時計算未來 24 小時趨勢..."):
        try:
            df_pred = dynamic_predict_24h(current_hour, live_features_list)
        except Exception as e:
            st.error(f"❌ 模型推論發生錯誤: {e}")
            return

    # 5. 繪製預測折線圖
    current_hour_naive = current_hour.replace(tzinfo=None)
    end_time_24h = current_hour_naive + datetime.timedelta(hours=24)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=[current_hour_naive],
            y=[live_features_list[6]],
            mode="markers",
            name="當前實測值 (基準時間)",
            marker=dict(color="red", size=12),
        )
    )

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

    # 6. 明細表格
    st.subheader("📋 未來 24 小時預測數值明細")
    df_display = pd.DataFrame(
        {
            "預測時間點": df_pred["target_datetime"].dt.strftime("%m/%d %H:00"),
            "預測 PM2.5 (µg/m³)": df_pred["predicted_pm25"].round(2),
        }
    )
    df_display = df_display.astype(str)
    st.dataframe(df_display, use_container_width=True)


if __name__ == "__main__":
    main()
