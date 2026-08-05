import sqlite3
import datetime

DB_NAME = "pm25_forecast.db"

def init_db():
    """初始化 SQLite 資料庫與資料表"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # 1. 建立實測資料表 (real_data)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS real_data (
            timestamp TEXT PRIMARY KEY,
            press REAL,
            temp REAL,
            rh REAL,
            wind_spd REAL,
            wind_x REAL,
            wind_y REAL,
            rain REAL,
            pm25 REAL,
            v_2100N REAL,
            v_2100S REAL,
            v_2125N REAL,
            v_2129S REAL,
            sin_hour REAL,
            cos_hour REAL
        )
    ''')

    # 2. 建立預測紀錄表 (predictions)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            base_time TEXT,
            target_time TEXT,
            step INTEGER,
            pred_pm25 REAL,
            created_at TEXT
        )
    ''')

    conn.commit()
    conn.close()

def save_real_data(timestamp, features):
    """寫入或更新當前實測資料"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    query = '''
        INSERT OR REPLACE INTO real_data (
            timestamp, press, temp, rh, wind_spd, wind_x, wind_y,
            rain, pm25, v_2100N, v_2100S, v_2125N, v_2129S, sin_hour, cos_hour
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    '''
    cursor.execute(query, [timestamp] + list(features))
    conn.commit()
    conn.close()

def save_predictions(base_time, predictions_list):
    """寫入未來 24 小時的預測值"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for step, (target_time_str, pred_val) in enumerate(predictions_list, 1):
        cursor.execute('''
            INSERT INTO predictions (base_time, target_time, step, pred_pm25, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (base_time, target_time_str, step, pred_val, created_at))

    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("✅ SQLite 資料庫 `pm25_forecast.db` 初始化成功！")