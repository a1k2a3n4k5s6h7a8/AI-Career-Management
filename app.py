# app.py
# The main Flask application that handles routes, session management,
# SQLite database calls, and ML model inference.

import os
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request, redirect, url_for, session, flash
from database import init_db, register_user, authenticate_user, save_prediction, get_user_history, delete_prediction

app = Flask(__name__)
# Secret key for secure session cookie signing. In production, use an environment variable.
app.secret_key = "career_mentor_secret_key_12345"

# Initialize database on startup
init_db()

# Define paths for models
CLASSIFIER_PATH = os.path.join("model", "placement_classifier.joblib")
REGRESSOR_PATH = os.path.join("model", "salary_regressor.joblib")

# Global variables for models (loaded on first request or startup)
clf_model = None
reg_model = None

def load_models():
    """
    Helper function to load the trained models from disk if they aren't loaded yet.
    """
    global clf_model, reg_model
    if clf_model is None or reg_model is None:
        if os.path.exists(CLASSIFIER_PATH) and os.path.exists(REGRESSOR_PATH):
            clf_model = joblib.load(CLASSIFIER_PATH)
            reg_model = joblib.load(REGRESSOR_PATH)
            print("Models loaded successfully.")
        else:
            print("WARNING: Models not found on disk. Please run train_model.py first.")

# -------------------------------------------------------------
# MIDDLEWARE/DECORATOR FOR SESSION PROTECTION
# -------------------------------------------------------------
def login_required(f):
    """
    Custom decorator to protect routes that require user authentication.
    """
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please log in to access this page.", "danger")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# -------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------

@app.route('/')
def home():
    """
    Landing page route. Displays app overview and hero section.
    """
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """
    User registration route.
    """
    if 'user_id' in session:
        return redirect(url_for('predict'))
        
    if request.method == 'POST':
        username = request.form['username'].strip()
        email = request.form['email'].strip()
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        # Validation checks
        if not username or not email or not password:
            flash("All fields are required.", "danger")
            return render_template('register.html')
            
        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template('register.html')
            
        # Try to register user
        success = register_user(username, email, password)
        if success:
            flash("Registration successful! Please login.", "success")
            return redirect(url_for('login'))
        else:
            flash("Username or Email already exists.", "danger")
            
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    User login route.
    """
    if 'user_id' in session:
        return redirect(url_for('predict'))
        
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        
        if not username or not password:
            flash("Both fields are required.", "danger")
            return render_template('login.html')
            
        user = authenticate_user(username, password)
        if user:
            # Save user details to session
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash(f"Welcome back, {user['username']}!", "success")
            return redirect(url_for('predict'))
        else:
            flash("Invalid username or password.", "danger")
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    """
    Clears the session and logs the user out.
    """
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for('home'))

@app.route('/predict', methods=['GET'])
@login_required
def predict():
    """
    Renders the predictive form.
    """
    # Try loading models to ensure they're available
    load_models()
    return render_template('predict.html', models_available=(clf_model is not None))

@app.route('/result', methods=['POST'])
@login_required
def result():
    """
    Processes form input, runs the ML models, saves prediction, and returns analysis.
    """
    load_models()
    if clf_model is None or reg_model is None:
        flash("Models are currently training or unavailable. Please try again in a few minutes.", "warning")
        return redirect(url_for('predict'))
        
    try:
        # Collect all form fields and convert to appropriate types
        data = {
            'age': int(request.form['age']),
            'gender': request.form['gender'],
            'cgpa': float(request.form['cgpa']),
            'branch': request.form['branch'],
            'college_tier': request.form['college_tier'],
            'attendance_percentage': float(request.form['attendance_percentage']),
            'backlogs': int(request.form['backlogs']),
            'coding_skill_score': float(request.form['coding_skill_score']),
            'aptitude_score': float(request.form['aptitude_score']),
            'communication_skill_score': float(request.form['communication_skill_score']),
            'logical_reasoning_score': float(request.form['logical_reasoning_score']),
            'mock_interview_score': float(request.form['mock_interview_score']),
            'extracurricular_score': float(request.form['extracurricular_score']),
            'leadership_score': float(request.form['leadership_score']),
            'internships_count': int(request.form['internships_count']),
            'projects_count': int(request.form['projects_count']),
            'certifications_count': int(request.form['certifications_count']),
            'hackathons_participated': int(request.form['hackathons_participated']),
            'github_repos': int(request.form['github_repos']),
            'linkedin_connections': int(request.form['linkedin_connections']),
            'volunteer_experience': request.form['volunteer_experience'],
            'sleep_hours': float(request.form['sleep_hours']),
            'study_hours_per_day': float(request.form['study_hours_per_day'])
        }
        
        # Convert dictionary to a pandas DataFrame (1 row) with matching column names
        input_df = pd.DataFrame([data])
        
        # 1. Predict Placement Status and Probability
        pred_class_encoded = clf_model.predict(input_df)[0]  # returns 1 or 0
        pred_prob = clf_model.predict_proba(input_df)[0][pred_class_encoded] * 100
        
        pred_class = "Placed" if pred_class_encoded == 1 else "Not Placed"
        
        # 2. Predict Salary Package
        if pred_class == "Placed":
            # If placed, predict estimated salary package (LPA)
            pred_salary = float(reg_model.predict(input_df)[0])
            pred_salary = round(pred_salary, 2)
        else:
            # If not placed, salary is 0
            pred_salary = 0.0
            
        # 3. Save prediction to DB linked to user
        save_prediction(session['user_id'], data, pred_class, pred_prob, pred_salary)
        
        # 4. Generate Career Mentoring Recommendations
        recommendations = generate_career_advice(data, pred_class)
        
        return render_template(
            'result.html',
            prediction=pred_class,
            probability=round(pred_prob, 1),
            salary=pred_salary,
            student_data=data,
            recommendations=recommendations
        )
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        flash("An error occurred during prediction. Please verify your inputs.", "danger")
        return redirect(url_for('predict'))

@app.route('/history')
@login_required
def history():
    """
    Retrieves and displays prediction history of the logged-in user.
    """
    user_id = session['user_id']
    predictions = get_user_history(user_id)
    return render_template('history.html', predictions=predictions)

@app.route('/delete_history/<int:id>')
@login_required
def delete_history(id):
    """
    Deletes a specific prediction entry and redirects back to history.
    """
    user_id = session['user_id']
    delete_prediction(id, user_id)
    flash("Prediction record deleted.", "success")
    return redirect(url_for('history'))

# -------------------------------------------------------------
# MENTORING RECOMMENDATION GENERATOR
# -------------------------------------------------------------
def generate_career_advice(data, prediction):
    """
    Generates personalized recommendations based on student scores.
    """
    advice = {
        'critical': [],   # Urgent things to fix (e.g. low CGPA, high backlogs)
        'skills': [],     # Technical/soft skill suggestions
        'networking': [], # LinkedIn, GitHub, hackathons
        'lifestyle': [],  # Sleep/Study habits
        'verdict': ""     # Overall mentorship summary
    }
    
    # Academics & Backlogs
    if data['backlogs'] > 0:
        advice['critical'].append(
            f"You have {data['backlogs']} active backlog(s). Clearing them is your absolute priority. Most companies have a 'Zero Backlogs' policy during recruitment."
        )
    if data['cgpa'] < 7.0:
        advice['critical'].append(
            f"Your CGPA is {data['cgpa']:.2f}. Many premium tier companies filter out resumes below 7.0 or 7.5. Focus on improving exam performance to lift your GPA."
        )
    if data['attendance_percentage'] < 75.0:
        advice['critical'].append(
            f"Your attendance is {data['attendance_percentage']:.1f}%. Falling below 75% can lead to academic penalties and suggests lack of discipline to recruiters."
        )
        
    # Technical & Aptitude Skills
    if data['coding_skill_score'] < 65.0:
        advice['skills'].append(
            f"Coding Score ({data['coding_skill_score']:.1f}%) is low. Solve 2-3 coding problems daily on platforms like LeetCode, GeeksforGeeks, or HackerRank."
        )
    if data['aptitude_score'] < 65.0:
        advice['skills'].append(
            f"Aptitude Score ({data['aptitude_score']:.1f}%) needs work. Quantitative/logic rounds are the first elimination filters. Practice puzzles and time-bound math quizzes."
        )
    if data['projects_count'] < 2:
        advice['skills'].append(
            "You only have completed 0 or 1 project. Build at least 2 robust, end-to-end projects demonstrating your skills in Web Dev, App Dev, or Data Science."
        )
    if data['certifications_count'] == 0:
        advice['skills'].append(
            "Gain industry credibility by earning certifications in cloud services (AWS/GCP), database systems, or software engineering methodologies."
        )

    # Soft Skills & Interviews
    if data['communication_skill_score'] < 70.0:
        advice['skills'].append(
            f"Communication score is {data['communication_skill_score']:.1f}%. Practice mock presentations, speak in front of a mirror, or participate in group discussions."
        )
    if data['mock_interview_score'] < 70.0:
        advice['skills'].append(
            f"Mock interview performance ({data['mock_interview_score']:.1f}%) is weak. Work on technical articulation, resume walkthroughs, and behavior responses (STAR method)."
        )

    # Networking & Practical exposure
    if data['linkedin_connections'] < 250:
        advice['networking'].append(
            f"Networking is low ({data['linkedin_connections']} connections). Connect with college alumni, industry professionals, and recruiters. Share your learning journey."
        )
    if data['github_repos'] < 3:
        advice['networking'].append(
            f"GitHub presence is minimal ({data['github_repos']} repos). Push all your project code, write professional READMEs, and contribute to open-source."
        )
    if data['hackathons_participated'] == 0:
        advice['networking'].append(
            "Participate in hackathons! Even if you don't win, hackathons simulate real pressure, build teamwork, and add great value to your resume."
        )
        
    # Habits
    if data['sleep_hours'] < 6.0:
        advice['lifestyle'].append(
            f"You get only {data['sleep_hours']:.1f} hours of sleep. Insufficient sleep severely affects cognitive skills, coding focus, and logical reasoning."
        )
    if data['study_hours_per_day'] < 2.0:
        advice['lifestyle'].append(
            "Allocate at least 2-3 structured hours per day for upskilling, system design concepts, and mock interview preparations."
        )

    # Overall Verdict
    if prediction == "Placed":
        advice['verdict'] = "Excellent job! Your profile shows solid academics and skills. Focus on polishing advanced system design, coding optimizations, and behavioral communication to target top-tier dream offers."
    else:
        advice['verdict'] = "Don't lose heart! Your profile has potential, but key weaknesses (like backlogs, GPA, or coding scores) are dragging you down. Focus on the critical alerts and skill improvements highlighted above."
        
    return advice

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False, host="127.0.0.1", port=5000)