# Titanic EDA - Jupyter Notebook
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ========================
# 2. Load Dataset
# ========================
# Titanic dataset (commonly available in seaborn as well)
# If you have "titanic.csv", replace the below line with:
df = pd.read_csv(r"C:\Users\jivit\OneDrive\Documents\Matplotib & Seaborn\titanic.csv")
df = sns.load_dataset("titanic")

# ========================
# 3. Initial Exploration
# ========================
print("Dataset Info:")
print(df.info())

print("\nDataset Description:")
print(df.describe(include='all'))

print("\nMissing Values:")
print(df.isnull().sum())

# Value counts for categorical columns
print("\nSurvival Count:")
print(df['survived'].value_counts())

print("\nClass Distribution:")
print(df['class'].value_counts())

# ========================
# 4. Univariate Analysis
# ========================

# Histogram of Age
plt.figure(figsize=(8,5))
sns.histplot(df['age'].dropna(), bins=30, kde=True)
plt.title("Age Distribution")
plt.show()
# Observation: Most passengers are between 20-40 years.

# Countplot for Survived
plt.figure(figsize=(6,4))
sns.countplot(x='survived', data=df)
plt.title("Survival Distribution")
plt.show()
# Observation: More passengers did not survive.

# Boxplot of Age by Class
plt.figure(figsize=(8,5))
sns.boxplot(x='class', y='age', data=df)
plt.title("Age Distribution by Class")
plt.show()
# Observation: Higher class passengers are generally older.

# ========================
# 5. Bivariate Analysis
# ========================

# Survival rate by gender
plt.figure(figsize=(6,4))
sns.countplot(x='sex', hue='survived', data=df)
plt.title("Survival by Gender")
plt.show()
# Observation: Females had higher survival rate.

# Survival rate by class
plt.figure(figsize=(6,4))
sns.countplot(x='class', hue='survived', data=df)
plt.title("Survival by Class")
plt.show()
# Observation: First class passengers had higher survival chances.

# Scatterplot Age vs Fare
plt.figure(figsize=(8,5))
sns.scatterplot(x='age', y='fare', hue='survived', data=df)
plt.title("Age vs Fare (Survival Highlighted)")
plt.show()
# Observation: High fare passengers were more likely to survive.

# ========================
# 6. Multivariate Analysis
# ========================

# Correlation heatmap (numerical features only)
plt.figure(figsize=(8,6))
# FIX APPLIED HERE: Explicitly select the key numerical/ordinal columns for a focused analysis
key_cols = ['survived', 'age', 'fare', 'pclass', 'sibsp', 'parch']
sns.heatmap(df[key_cols].corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap of Key Features") # Updated title
plt.show()
# Observation: Fare and pclass (class) show strong correlation with survival; age is not strongly correlated.

# Pairplot
sns.pairplot(df[['age', 'fare', 'survived']], hue='survived')
plt.suptitle("Pairplot of Age, Fare & Survival", y=1.02)
plt.show()
# Observation: Survived passengers tend to pay higher fares.

# ========================
# 7. Summary of Findings
# ========================
print("\n--- Summary of Findings ---")
print("""
1. Majority of passengers were in the 20–40 age range.
2. Survival rate was low (~38%).
3. Females had much higher survival rate compared to males.
4. Higher-class passengers (1st class) survived more often than lower-class.
5. Higher fare is associated with better survival chances.
6. Correlation heatmap shows Fare and Pclass are moderately/strongly correlated with Survival.
7. Age was not a strong determinant of survival.
""")