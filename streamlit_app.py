import datetime
import os
import sqlite3
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# **強制鎖定工作目錄**，確保 Streamlit 與 application.py 使用同一個路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# 引用原本完全未修改的 application.py 與 db_manager.py
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
        "結合大氣氣象、即時環測與國道 3 號車流量之 LSTM"
        " 深度學習趨勢預測儀表板"
    )

    # 側邊欄控制與資訊
    st.sidebar.header("⚙️ 系統狀態與設定")
    if st.sidebar.button("🔄 刷新即時監測數據"):
        st.rerun()

    now = datetime.datetime.now()
    current_time_str = now.strftime("%Y-%m-%d %H:00")
    st.sidebar.write(f"🕒 **當前基準時間**: {current_time_str}")

    # 1. 抓取即時資料
    with st.spinner("📡 正在擷取霧峰即時監測數據..."):
        try:
            live_features = application.fetch_wufeng_live_features()
            live_features_list = [float(x) for x in live_features]
        except Exception as e:
            st.error(f"擷取即時特徵時發生錯誤: {e}")
            return

    # 2. 頂部即時指標卡片
    st.subheader("📊 即時監測數據 Summary")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("即時 PM2.5", f"{live_features_list[7]:.1f} µg/m³")
    col2.metric("氣溫", f"{live_features_list[1]:.1f} ℃")
    col3.metric("相對濕度", f"{live_features_list[2]:.0f} %")
    col4.metric("風速", f"{live_features_list[3]:.1f} m/s")
    col5.metric(
        "國道3號車流量 (北上)", f"{int(live_features_list[8])} 輛/時"
    )

    st.markdown("---")

    # 3. 檢查模型權重
    st.subheader("🔮 未來 24 小時 PM2.5 預測趨勢圖")

    model_path = "best_model_ExpC_Cyclic.pth"
    if not os.path.exists(model_path):
        st.error(
            f"❌ 找不到模型權重檔 `{model_path}`，請確認檔案已上傳至 GitHub！"
        )
        return

    # 4. 執行應用主程式與讀取 SQLite 數據
    df_pred = pd.DataFrame()

    with st.spinner("🔮 正在執行 application.main() 進行滾動推論..."):
        try:
            # 呼叫你原本寫好的 main 流程 (會自動執行推論並寫入 SQLite)
            application.main()
        except Exception as e:
            st.warning(f"背景主程式執行提示: {e}")

        # ✅ 修正一：優先讀取你建立的 pm25_forecast.db
        db_file = os.path.join(current_dir, "pm25_forecast.db")
        if not os.path.exists(db_file):
            db_file = os.path.join(current_dir, "pm25_data.db")

        if os.path.exists(db_file):
            try:
                conn = sqlite3.connect(db_file)

                # ✅ 修正二：動態支援 pred_pm25 與 predicted_pm25 欄位名
                try:
                    query = """
                        SELECT target_time, pred_pm25 AS predicted_pm25 
                        FROM predictions 
                        ORDER BY id DESC LIMIT 24
                    """
                    df_pred = pd.read_sql_query(query, conn)
                except Exception:
                    query_backup = """
                        SELECT target_time, predicted_pm25 
                        FROM predictions 
                        ORDER BY id DESC LIMIT 24
                    """
                    df_pred = pd.read_sql_query(query_backup, conn)

                if not df_pred.empty:
                    df_pred = df_pred.iloc[::-1].reset_index(drop=True)
                conn.close()
            except Exception as e:
                st.error(f"讀取 SQLite 預測數據失敗: {e}")

    # 5. 繪製 Plotly 圖表與表格
    if not df_pred.empty:
        df_pred["display_time"] = pd.to_datetime(
            df_pred["target_time"]
        ).dt.strftime("%m/%d %H:00")

        fig = go.Figure()

        # 當前實測點
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

        # 參考標準線
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

        st.subheader("📋 未來 24 小時預測數值明細")
        df_display = pd.DataFrame(
            {
                "預測時間點": df_pred["display_time"],
                "預測 PM2.5 (µg/m³)": df_pred["predicted_pm25"].round(2),
            }
        )
        st.dataframe(df_display.T, use_container_width=True)
    else:
        st.error(
            "⚠️ 無法取得預測數據，請確認 application.main() 是否順利將結果寫入"
            " SQLite 資料庫。"
        )


if __name__ == "__main__":
    main()