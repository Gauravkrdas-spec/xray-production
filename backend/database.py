import sqlite3
import os
import json
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), 'xray_database.db')


# ── Initialize database ───────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Analyses table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analyses (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id      TEXT NOT NULL,
            patient_name  TEXT,
            patient_age   TEXT,
            patient_gender TEXT,
            referred_by   TEXT,
            findings      TEXT,
            report        TEXT,
            urgency       TEXT,
            timestamp     TEXT,
            created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Doctor feedback table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS doctor_feedback (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id   TEXT,
            doctor_name   TEXT NOT NULL,
            hospital      TEXT,
            rating        TEXT NOT NULL,
            comments      TEXT,
            recommend     TEXT,
            timestamp     TEXT,
            created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Validation stats table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS validation_stats (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            total_reviews   INTEGER DEFAULT 0,
            correct_count   INTEGER DEFAULT 0,
            partial_count   INTEGER DEFAULT 0,
            wrong_count     INTEGER DEFAULT 0,
            recommend_yes   INTEGER DEFAULT 0,
            recommend_no    INTEGER DEFAULT 0,
            recommend_maybe INTEGER DEFAULT 0,
            last_updated    DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Insert initial stats row if empty
    cursor.execute(
        'SELECT COUNT(*) FROM validation_stats'
    )
    if cursor.fetchone()[0] == 0:
        cursor.execute(
            'INSERT INTO validation_stats DEFAULT VALUES'
        )

    conn.commit()
    conn.close()
    print("Database initialized successfully")


# ── Save analysis ─────────────────────────────────────
def save_analysis(
    image_id, patient_name, patient_age,
    patient_gender, referred_by, findings, report
):
    try:
        conn   = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO analyses (
                image_id, patient_name, patient_age,
                patient_gender, referred_by,
                findings, report, urgency, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            image_id,
            patient_name,
            patient_age,
            patient_gender,
            referred_by,
            json.dumps(findings),
            json.dumps(report),
            report.get('urgency', 'routine'),
            datetime.now().isoformat()
        ))
        analysis_db_id = cursor.lastrowid
        conn.commit()
        conn.close()
        print(f"Analysis saved: ID {analysis_db_id}")
        return analysis_db_id
    except Exception as e:
        print(f"Error saving analysis: {e}")
        return None


# ── Save doctor feedback ──────────────────────────────
def save_feedback(
    analysis_id, doctor_name, hospital,
    rating, comments, recommend
):
    try:
        conn   = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Save feedback
        cursor.execute('''
            INSERT INTO doctor_feedback (
                analysis_id, doctor_name, hospital,
                rating, comments, recommend, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            analysis_id,
            doctor_name,
            hospital,
            rating,
            comments,
            recommend,
            datetime.now().isoformat()
        ))

        # Update validation stats
        cursor.execute('''
            UPDATE validation_stats SET
                total_reviews   = total_reviews + 1,
                correct_count   = correct_count + ?,
                partial_count   = partial_count + ?,
                wrong_count     = wrong_count + ?,
                recommend_yes   = recommend_yes + ?,
                recommend_no    = recommend_no + ?,
                recommend_maybe = recommend_maybe + ?,
                last_updated    = CURRENT_TIMESTAMP
        ''', (
            1 if rating    == 'correct' else 0,
            1 if rating    == 'partial' else 0,
            1 if rating    == 'wrong'   else 0,
            1 if recommend == 'yes'     else 0,
            1 if recommend == 'no'      else 0,
            1 if recommend == 'maybe'   else 0,
        ))

        conn.commit()
        conn.close()
        print(f"Feedback saved from Dr. {doctor_name}")
        return True
    except Exception as e:
        print(f"Error saving feedback: {e}")
        return False


# ── Get validation stats ──────────────────────────────
def get_stats():
    try:
        conn   = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM validation_stats LIMIT 1')
        row = cursor.fetchone()
        conn.close()

        if not row:
            return {}

        total = row[1]
        correct = row[2]
        accuracy = round((correct / total * 100), 1) if total > 0 else 0
        deployment_ready = total >= 100 and accuracy >= 80

        return {
            'total_reviews':    total,
            'correct_count':    correct,
            'partial_count':    row[3],
            'wrong_count':      row[4],
            'recommend_yes':    row[5],
            'recommend_no':     row[6],
            'recommend_maybe':  row[7],
            'accuracy_rate':    accuracy,
            'deployment_ready': deployment_ready,
            'reviews_needed':   max(0, 100 - total),
            'last_updated':     row[8]
        }
    except Exception as e:
        print(f"Error getting stats: {e}")
        return {}


# ── Get recent analyses ───────────────────────────────
def get_recent_analyses(limit=10):
    try:
        conn   = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT
                id, image_id, patient_name,
                patient_age, patient_gender,
                urgency, timestamp
            FROM analyses
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [{
            'id':             r[0],
            'image_id':       r[1],
            'patient_name':   r[2],
            'patient_age':    r[3],
            'patient_gender': r[4],
            'urgency':        r[5],
            'timestamp':      r[6]
        } for r in rows]
    except Exception as e:
        print(f"Error getting analyses: {e}")
        return []


# ── Get recent feedback ───────────────────────────────
def get_recent_feedback(limit=10):
    try:
        conn   = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT
                doctor_name, hospital,
                rating, recommend, timestamp
            FROM doctor_feedback
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [{
            'doctor_name': r[0],
            'hospital':    r[1],
            'rating':      r[2],
            'recommend':   r[3],
            'timestamp':   r[4]
        } for r in rows]
    except Exception as e:
        print(f"Error getting feedback: {e}")
        return []