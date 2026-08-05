import datetime
import os
import sqlite3
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# 引用你原本完全未修改的 application.py 與 db_manager.py
import application
import db_manager

# 設定 Streamlit 網頁標題與寬度
st.set_page_config(
    page_title="台中霧峰 PM2.5 未來 24 小時預測系統",
    layout="wide",
    page_icon="🌬️",
)


def main():
    st.title("🌬️ 台中市霧峰區 PM2.5 未來 24 小時預測系統")
    st.caption(
        "結合大氣氣象、即時環測與國道 3 號車流量之 LSTM 深度學習趨勢預測儀表板"
    )

    # 側邊欄控制與資訊
    st.sidebar.header("⚙️ 系統狀態與設定")
    if st.sidebar.button("🔄 刷新即時監測數據"):
        st.rerun()

    now = datetime.datetime.now()
    current_time_str = now.strftime("%Y-%m-%d %H:00")
    st.sidebar.write(f"🕒 **當前基準時間**: {current_time_str}")

    # 執行與呼叫原本 application.py 裡面的即時資料抓取
    with st.spinner("📡 正在呼叫爬蟲與模型推論..."):
        try:
            live_features = application.fetch_wufeng_live_features()
            live_features_list = [float(x) for x in live_features]
        except Exception as e:
            st.error(f"擷取即時特徵時發生錯誤: {e}")
            return

    # **頂部即時指標卡片** (對應原本 14 項特徵順序)
    st.subheader("📊 即時監測數據 Summary")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("即時 PM2.5", f"{live_features_list[7]:.1f} µg/m³")
    col2.metric("氣溫", f"{live_features_list[1]:.1f} ℃")
    col3.metric("相對濕度", f"{live_features_list[2]:.0f} %")
    col4.metric("風速", f"{live_features_list[3]:.1f} m/s")
    col5.metric("國道3號車流量 (北上)", f"{int(live_features_list[8])} 輛/時")

    st.markdown("---")

    # **觸發背景預測與 SQLite 寫入 (直接執行你原本寫好的 application.main)**
    st.subheader("🔮 未來 24 小時 PM2.5 預測趨勢圖")

    # 檢查模型權重是否存在
    model_path = "best_model_ExpC_Cyclic.pth"
    if not os.path.exists(model_path):
        st.error(
            f"❌ 找不到模型權重檔 `{model_path}`，請確認已將權重檔上傳至"
            " GitHub！"
        )
        return

    # 執行原本 application.py 的流程來計算預測值並更新資料庫
    with st.spinner("🔮 正在進行 LSTM 滾動推論..."):
        application.main()

    # **修正處：改用標準 sqlite3 連線讀取 SQLite 資料庫 (pm25_data.db)**
    df_pred = pd.DataFrame()
    db_file = "pm25_data.db"

    if os.path.exists(db_file):
        try:
            conn = sqlite3.connect(db_file)
            query = """
                SELECT target_time, predicted_pm25 
                FROM predictions 
                WHERE base_time = ? 
                ORDER BY target_time ASC
            """
            df_pred = pd.read_sql_query(query, conn, params=(current_time_str,))
            conn.close()
        except Exception as e:
            st.warning(f"讀取 SQLite 數據庫時發生提示: {e}")
    else:
        st.warning(f"⚠️ 找不到 `{db_file}` 資料庫檔案")

    if not df_pred.empty:
        # 整理時間格式
        df_pred["display_time"] = pd.to_datetime(
            df_pred["target_time"]
        ).dt.strftime("%m/%d %H:00")

        # **繪製互動式 Plotly 折線圖**
        fig = go.Figure()

        # 現況實測點
        fig.add_trace(
            go.Scatter(
                x=[now.strftime("%m/%d %H:00")],
                y=[live_features_list[7]],
                mode="markers",
                name="當前實測值",
                marker=dict(color="red", size=12),
            )
        )

        # 預測折線
        fig.add_trace(
            go.Scatter(
                x=df_pred["display_time"],
                y=df_pred["predicted_pm25"],
                mode="lines+markers",
                name="LSTM 預測 PM2.5 (µg/m³)",
                line=dict(color="#0083B0", width=3),
                marker=dict(size=6),
            )
        )

        # WHO 與環境部標準線
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
            xaxis_title="預測時間點",
            yaxis_title="PM2.5 濃度 (µg/m³)",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=30, b=20),
            height=450,
        )

        st.plotly_chart(fig, use_container_width=True)

        # **數據表格**
        st.subheader("📋 未來 24 小時預測數值明細")
        df_display = pd.DataFrame(
            {
                "預測時間點": df_pred["display_time"],
                "預測 PM2.5 (µg/m³)": df_pred["predicted_pm25"].round(2),
            }
        )
        st.dataframe(df_display.T, use_container_width=True)
    else:
        st.info("💡 預測完成，請重新整理頁面讀取 SQLite 最新數據。")


if __name__ == "__main__":
    main()