import sqlite3
import os
DB_PATH = os.getenv("PCOS_DB_PATH", "pcos_agent_logs.db")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        thread_id TEXT,
        user_input TEXT,
        prediction TEXT,
        probability REAL,
        explanation TEXT,
        recommendation TEXT,
        model_version TEXT DEFAULT 'v1.0',
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()


def save_to_db(user_input, prediction, probability, explanation, recommendation, user_id="anonymous", thread_id="unknown", model_version="v1.0"):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (user_id, thread_id, user_input, prediction, probability, explanation, recommendation, model_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, thread_id, str(user_input), prediction, probability, explanation, recommendation, model_version))
    conn.commit()
    conn.close()
