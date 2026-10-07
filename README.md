# AI Placement Predictor & Career Mentor

An institutional-grade Machine Learning web application that predicts college student placement probabilities and estimates salary packages (LPA), paired with an automated AI career mentor to provide personalized upskilling roadmaps.

## 🚀 Features

*   **Secure Authentication**: User sign-up and login with passwords hashed using industrial-standard PBKDF2 cryptography.
*   **Double ML Pipeline**: 
    1.  **Placement Classifier**: Compares Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting, automatically selecting and deploying the highest accuracy model.
    2.  **Salary Regressor**: Estimates salary package ranges in Lakhs Per Annum (LPA) for students using a Random Forest Regressor trained on historical placed profiles.
*   **Personalized Career Roadmaps**: Analyzes user scores (CGPA, coding skill, attendance, backlogs, LinkedIn, and mock interview performance) to generate immediate upskilling roadmaps.
*   **Assessment History Dashboard**: View, track, and manage past predictions to see performance trends.
*   **Modern Responsive UI**: Built with Bootstrap 5 and a custom glassmorphic slate dark theme for premium styling.

---

## 📁 Project Structure

```text
AI-Career-Mentor/
│
├── dataset/
│   └── student_placement_prediction_dataset_2026.csv   # 100k student records dataset
│
├── model/
│   ├── placement_classifier.joblib                     # Saved best classifier model
│   ├── salary_regressor.joblib                         # Saved salary regressor model
│   └── feature_info.joblib                             # Saved column options and lists
│
├── templates/
│   ├── index.html                                      # Landing/Hero page
│   ├── login.html                                      # User Login Card
│   ├── register.html                                   # User Registration Card
│   ├── predict.html                                    # Multi-category assessment inputs
│   ├── result.html                                     # Premium predictive dashboard
│   └── history.html                                    # Tabular prediction history
│
├── static/
│   ├── css/
│   │   └── style.css                                   # Glassmorphism, animations & styles
│   └── js/
│       └── script.js                                   # Custom validation & match checking
│
├── app.py                                              # Flask main application & endpoints
├── train_model.py                                      # Dataset processing & ML training
├── database.py                                         # SQLite setup and user queries
├── requirements.txt                                    # Project dependencies
├── README.md                                           # Project documentation
└── .gitignore                                          # Untracked files list
```

---

## 🛠️ Installation & Setup

Follow these steps to run the application on your local machine:

### 1. Prerequisite: Python
Make sure you have Python 3.8 or above installed on your system.

### 2. Set Up a Virtual Environment (Recommended)
Open your terminal/command prompt in the project root directory and run:

```bash
# Create a virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment
# On Windows (Command Prompt):
venv\Scripts\activate
# On Windows (PowerShell):
venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
Install all required libraries specified in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Train the ML Models
To train the classifiers, compare accuracies, and generate the model joblib files, run:

```bash
python train_model.py
```
This script will output the comparison tables and confirm the files are saved in the `model/` folder.

### 5. Launch the Web Application
Run the Flask server:

```bash
python app.py
```

Open your browser and navigate to:
```text
http://127.0.0.1:5000/
```

---

## 📊 Dataset Insights & Columns
The models analyze **23 parameters** across 4 categories:
1.  **Academic Standing**: `cgpa`, `backlogs`, `attendance_percentage`, `college_tier`, `branch`, `age`
2.  **Assessment Scores**: `coding_skill_score`, `aptitude_score`, `logical_reasoning_score`, `communication_skill_score`, `mock_interview_score`
3.  **Experience & Profile**: `internships_count`, `projects_count`, `certifications_count`, `hackathons_participated`, `github_repos`, `linkedin_connections`, `volunteer_experience`
4.  **Habits**: `sleep_hours`, `study_hours_per_day`
