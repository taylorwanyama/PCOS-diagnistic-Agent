#This file helps you view the database
import sqlite3
import json

def get_pcos_logs(db_path='pcos_agent_logs.db'):
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get table names to confirm 'predictions' exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Tables in {db_path}: {tables}")

        # Check if 'predictions' table exists
        if ('predictions',) not in tables:
            print(f"Error: Table 'predictions' not found in {db_path}.")
            return

        # Get column names from the 'predictions' table
        cursor.execute("PRAGMA table_info(predictions);")
        column_info = cursor.fetchall()
        column_names = [col[1] for col in column_info]
        print(f"Columns in 'predictions' table: {column_names}\n")

        # Select all data from the 'predictions' table
        cursor.execute("SELECT * FROM predictions;")
        rows = cursor.fetchall()

        if not rows:
            print("No data found in the 'predictions' table.")
            return

        print("--- Stored PCOS Prediction Logs ---")
        for row in rows:
            # Create a dictionary for each row for easier access
            row_data = dict(zip(column_names, row))

            print(f"User ID: {row_data.get('user_id', 'N/A')}")
            print(f"Thread ID: {row_data.get('thread_id', 'N/A')}")
            print(f"Prediction: {row_data.get('prediction', 'N/A')}")
            print(f"Probability: {row_data.get('probability', 'N/A')}")
            print(f"Explanation: {row_data.get('explanation', 'N/A')}")
            print(f"Recommendation: {row_data.get('recommendation', 'N/A')}")

            # 'user_input' might be stored as a JSON string, try to parse it
            user_input_str = row_data.get('user_input')
            if user_input_str:
                try:
                    user_input_dict = json.loads(user_input_str)
                    print("User Input Features:")
                    for k, v in user_input_dict.items():
                        print(f"  {k.replace('_', ' ').title()}: {v}")
                except json.JSONDecodeError:
                    print(f"User Input (raw): {user_input_str}")
            else:
                print("User Input: N/A")
            print("-" * 40)

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    get_pcos_logs()