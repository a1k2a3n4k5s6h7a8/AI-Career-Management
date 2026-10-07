# database.py
# This file handles the SQLite database initialization and operations.
# It manages user authentication (registration, login) and placement prediction history.
# It uses Flask's built-in werkzeug.security library to securely hash passwords.

import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash

# Path to the SQLite database file
DB_PATH = "career_mentor.db"

def get_db_connection():
    """
    Establishes and returns a connection to the SQLite database.
    Enables row factory so results can be accessed like dictionaries (row['column_name']).
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # Enable Foreign Key support in SQLite
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    """
    Initializes the SQLite database by creating 'users' and 'predictions' tables
    if they do not already exist.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Create the Users Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 2. Create the Predictions Table
    # Stores all student inputs, placement outcomes, and recommended salary packages
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            
            -- Academic & Demographics
            age INTEGER NOT NULL,
            gender TEXT NOT NULL,
            cgpa REAL NOT NULL,
            branch TEXT NOT NULL,
            college_tier TEXT NOT NULL,
            attendance_percentage REAL NOT NULL,
            backlogs INTEGER NOT NULL,
            
            -- Technical & Non-Technical Scores
            coding_skill_score REAL NOT NULL,
            aptitude_score REAL NOT NULL,
            communication_skill_score REAL NOT NULL,
            logical_reasoning_score REAL NOT NULL,
            mock_interview_score REAL NOT NULL,
            extracurricular_score REAL NOT NULL,
            leadership_score REAL NOT NULL,
            
            -- Experience & Activity Counts
            internships_count INTEGER NOT NULL,
            projects_count INTEGER NOT NULL,
            certifications_count INTEGER NOT NULL,
            hackathons_participated INTEGER NOT NULL,
            github_repos INTEGER NOT NULL,
            linkedin_connections INTEGER NOT NULL,
            volunteer_experience TEXT NOT NULL,
            
            -- Habits
            sleep_hours REAL NOT NULL,
            study_hours_per_day REAL NOT NULL,
            
            -- Model Predictions
            prediction_class TEXT NOT NULL,      -- "Placed" or "Not Placed"
            prediction_prob REAL NOT NULL,       -- Probability (0% to 100%)
            salary_package REAL NOT NULL,        -- Salary package in LPA (0.0 if not placed)
            
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Database initialized successfully.")

# -------------------------------------------------------------
# USER AUTHENTICATION FUNCTIONS
# -------------------------------------------------------------

def register_user(username, email, password):
    """
    Registers a new user in the database.
    Hashes the password using werkzeug's PBKDF2 algorithm for security.
    Returns:
        True: If registration is successful.
        False: If username or email already exists.
    """
    hashed_password = generate_password_hash(password)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
            (username, email, hashed_password)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # Username or Email already exists (violates UNIQUE constraint)
        return False
    finally:
        conn.close()

def authenticate_user(username, password):
    """
    Authenticates a user for login.
    Checks the username and verifies the hashed password.
    Returns:
        Row: The user record dict-like object if login is successful.
        None: If credentials are invalid.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    
    if user and check_password_hash(user['password'], password):
        return user
    return None

# -------------------------------------------------------------
# PREDICTION HISTORY FUNCTIONS
# -------------------------------------------------------------

def save_prediction(user_id, data, pred_class, pred_prob, salary):
    """
    Saves a student placement prediction record linked to a user.
    Parameters:
        user_id (int): ID of the logged-in user.
        data (dict): The dictionary containing all student input parameters.
        pred_class (str): "Placed" or "Not Placed".
        pred_prob (float): Probability score (0.0 to 100.0).
        salary (float): Predicted salary package in LPA.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (
            user_id, age, gender, cgpa, branch, college_tier, attendance_percentage, backlogs,
            coding_skill_score, aptitude_score, communication_skill_score, logical_reasoning_score,
            mock_interview_score, extracurricular_score, leadership_score,
            internships_count, projects_count, certifications_count, hackathons_participated,
            github_repos, linkedin_connections, volunteer_experience, sleep_hours, study_hours_per_day,
            prediction_class, prediction_prob, salary_package
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
    ''', (
        user_id, data['age'], data['gender'], data['cgpa'], data['branch'], data['college_tier'],
        data['attendance_percentage'], data['backlogs'], data['coding_skill_score'], data['aptitude_score'],
        data['communication_skill_score'], data['logical_reasoning_score'], data['mock_interview_score'],
        data['extracurricular_score'], data['leadership_score'], data['internships_count'],
        data['projects_count'], data['certifications_count'], data['hackathons_participated'],
        data['github_repos'], data['linkedin_connections'], data['volunteer_experience'],
        data['sleep_hours'], data['study_hours_per_day'], pred_class, pred_prob, salary
    ))
    conn.commit()
    conn.close()

def get_user_history(user_id):
    """
    Fetches all historical predictions for a specific user, sorted from newest to oldest.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,)
    )
    history = cursor.fetchall()
    conn.close()
    return history

def delete_prediction(prediction_id, user_id):
    """
    Deletes a specific prediction from the history of a user.
    Ensures that a user can only delete their own predictions (via user_id verification).
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM predictions WHERE id = ? AND user_id = ?",
        (prediction_id, user_id)
    )
    conn.commit()
    conn.close()

# If run directly, initialize the database
if __name__ == "__main__":
    init_db()
