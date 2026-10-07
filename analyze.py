import pandas as pd
import numpy as np

# Load dataset
dest_dataset = r"c:\Users\HP\OneDrive\Documents\AI-Career-Mentor\AI-Career-Mentor\dataset\student_placement_prediction_dataset_2026.csv"
df = pd.read_csv(dest_dataset)

# Categorical columns distribution
categorical_cols = ['gender', 'branch', 'college_tier', 'volunteer_experience', 'placement_status']
print("--- Categorical Columns Distributions ---")
for col in categorical_cols:
    print(f"\nValue counts for '{col}':")
    print(df[col].value_counts(dropna=False))
    print(df[col].value_counts(normalize=True) * 100)

# Summary statistics of numerical columns
print("\n--- Summary Statistics of Numerical Columns ---")
print(df.describe().transpose())

# Relationship between placement_status and salary_package_lpa
print("\n--- Salary package when Placed vs Not Placed ---")
print(df.groupby('placement_status')['salary_package_lpa'].agg(['count', 'min', 'max', 'mean', 'median']))

# Correlation of numerical columns with a binary target (Placed = 1, Not Placed = 0)
df['placed_binary'] = df['placement_status'].apply(lambda x: 1 if x == 'Placed' else 0)
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numerical_cols.remove('student_id')  # ID column

print("\n--- Correlation with Placement Status (binary) ---")
correlations = df[numerical_cols].corr()['placed_binary'].sort_values(ascending=False)
print(correlations)

print("\n--- Correlation with Salary Package (LPA) ---")
# Only for placed students, or overall? Let's check both
print("Overall correlation with Salary:")
print(df[numerical_cols + ['salary_package_lpa']].corr()['salary_package_lpa'].sort_values(ascending=False))
print("\nCorrelation with Salary (only Placed students):")
placed_df = df[df['placement_status'] == 'Placed']
numerical_cols_placed = placed_df.select_dtypes(include=[np.number]).columns.tolist()
if 'student_id' in numerical_cols_placed:
    numerical_cols_placed.remove('student_id')
if 'placed_binary' in numerical_cols_placed:
    numerical_cols_placed.remove('placed_binary')
print(placed_df[numerical_cols_placed].corr()['salary_package_lpa'].sort_values(ascending=False))
