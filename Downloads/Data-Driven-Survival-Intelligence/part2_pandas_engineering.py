import pandas as pd
import numpy as np

# STEP 1: Load CSV using Pandas

df = pd.read_csv("train.csv")

print("=" * 55)
print("  STEP 1: Dataset Loaded")
print("=" * 55)
print(f"  Rows    : {df.shape[0]}")   # 891 passengers
print(f"  Columns : {df.shape[1]}")   # 12 features
print(f"\n  Column names:\n  {list(df.columns)}")


# STEP 2: Check Missing Values BEFORE filling

print("\n" + "=" * 55)
print("  STEP 2: Missing Values (Before Filling)")
print("=" * 55)
missing_before = df.isnull().sum()
print(missing_before[missing_before > 0])   # only show columns with missing values


# STEP 3: Fill Missing Values

# ── 3A: Fill Age with MEDIAN per Passenger Class 
# Why per class? Because:
#   Class 1 passengers → mostly wealthy adults → older median
#   Class 3 passengers → mostly immigrants    → younger median
# Using one overall median would be less accurate.
#
# How it works:
#   df.groupby("Pclass")["Age"]  → splits Age into 3 groups (class 1, 2, 3)
#   .transform("median")         → replaces each value with its GROUP's median
#   .fillna(...)                 → only fills the NaN (missing) spots

df["Age"] = df["Age"].fillna(
    df.groupby("Pclass")["Age"].transform("median")
)

# ── 3B: Fill Embarked with MODE
# Mode = most frequently occurring value.
# Only 2 values are missing. We fill them with the most common port.
#
# df["Embarked"].mode()    → returns a Series, e.g. ['S']
# [0]                      → gets the first (most common) value → 'S'

most_common_port = df["Embarked"].mode()[0]
df["Embarked"] = df["Embarked"].fillna(most_common_port)

# ── Verify: no more missing values in Age and Embarked ─────────
print("\n" + "=" * 55)
print("  STEP 3: Missing Values (After Filling)")
print("=" * 55)
missing_after = df.isnull().sum()
print(missing_after[missing_after > 0])
# Only Cabin should remain (we intentionally don't fill it)
print(f"\n  Age missing now    : {df['Age'].isnull().sum()}")      # should be 0
print(f"  Embarked missing now: {df['Embarked'].isnull().sum()}")  # should be 0


# STEP 4: Create New Features (Feature Engineering)
# Feature engineering = creating new columns from existing ones.
# These new columns may carry more useful information.

# ── 4A: FamilySize
# SibSp = number of siblings + spouse onboard
# Parch = number of parents + children onboard
# +1    = the passenger themselves
# So FamilySize = total number of family members including self

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# ── 4B: IsAlone
# If FamilySize == 1, the passenger is travelling alone
# (df["FamilySize"] == 1) gives True/False
# .astype(int) converts True → 1, False → 0

df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# ── 4C: FarePerPerson 
# The Fare column is the TOTAL fare paid for the whole group/family.
# Dividing by FamilySize gives the fare per individual.
# This is a fairer measure of wealth per person.

df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

print("\n" + "=" * 55)
print("  STEP 4: New Features Created")
print("=" * 55)
print(df[["Name", "SibSp", "Parch", "FamilySize", "IsAlone", "Fare", "FarePerPerson"]].head(8).to_string())


# ─────────────────────────────────────────────────────────────
# STEP 5: Create Categorical Bins
# ─────────────────────────────────────────────────────────────
# pd.cut() divides a continuous number column into labeled groups.
# bins  = the boundary values
# labels = the name for each group between boundaries

# ── 5A: AgeGroup ──────────────────────────────────────────────
# Boundaries: 0 → 15 → 60 → 100
# Groups:        Child  Adult  Senior

df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 15, 60, 100],
    labels=["Child", "Adult", "Senior"]
)

# ── 5B: FareGroup ─────────────────────────────────────────────
# Boundaries: 0 → 50 → 200 → 600
# Groups:       Low   Medium  High
# These boundaries are chosen based on the fare distribution.

df["FareGroup"] = pd.cut(
    df["Fare"],
    bins=[0, 50, 200, 600],
    labels=["Low", "Medium", "High"]
)

print("\n" + "=" * 55)
print("  STEP 5: Categorical Bins Created")
print("=" * 55)
print("\n  AgeGroup distribution:")
print(df["AgeGroup"].value_counts().to_string())

print("\n  FareGroup distribution:")
print(df["FareGroup"].value_counts().to_string())


# ─────────────────────────────────────────────────────────────
# STEP 6: Pivot Tables
# ─────────────────────────────────────────────────────────────
# A pivot table summarizes data by two categories.
# aggfunc="mean" on Survived gives survival RATE (0 to 1).
# Multiply by 100 to get percentage.

# ── 6A: Survival by Gender & Passenger Class ──────────────────
pivot_gender_class = df.pivot_table(
    values="Survived",       # column to summarize
    index="Sex",             # rows  (female / male)
    columns="Pclass",        # columns (1 / 2 / 3)
    aggfunc="mean"           # calculate mean of Survived
) * 100                      # convert to percentage

print("\n" + "=" * 55)
print("  STEP 6A: Survival Rate (%) by Gender & Class")
print("=" * 55)
print(pivot_gender_class.round(1).to_string())
# Expected insight:
#   Female Class 1 → ~96.8%    Male Class 1 → ~36.9%
#   Female Class 3 → ~50.0%    Male Class 3 → ~13.5%

# ── 6B: Survival by FareGroup & Embarked Port ─────────────────
pivot_fare_embarked = df.pivot_table(
    values="Survived",
    index="FareGroup",       # rows  (Low / Medium / High)
    columns="Embarked",      # columns (C / Q / S)
    aggfunc="mean"
) * 100

print("\n" + "=" * 55)
print("  STEP 6B: Survival Rate (%) by FareGroup & Embarked")
print("=" * 55)
print(pivot_fare_embarked.round(1).to_string())


# ─────────────────────────────────────────────────────────────
# STEP 7: Correlation Matrix
# ─────────────────────────────────────────────────────────────
# Correlation only works on numerical columns.
# We select the most relevant ones.
# Values close to +1 or -1 = strong relationship with Survived.
# Values close to 0 = weak relationship.

numerical_cols = ["Survived", "Pclass", "Age", "Fare",
                  "FamilySize", "IsAlone", "FarePerPerson"]

corr_matrix = df[numerical_cols].corr()

print("\n" + "=" * 55)
print("  STEP 7: Correlation Matrix")
print("=" * 55)
print(corr_matrix.round(3).to_string())

# Pull only the correlation WITH Survived (most useful column)
print("\n  Correlation with Survived (sorted):")
corr_with_survived = corr_matrix["Survived"].drop("Survived").sort_values(ascending=False)
print(corr_with_survived.round(3).to_string())


# ─────────────────────────────────────────────────────────────
# STEP 8: Rank Top 5 Features Affecting Survival
# ─────────────────────────────────────────────────────────────
# We use the absolute value of correlation — because both strong
# positive AND strong negative correlations mean the feature matters.
# abs(-0.34) = 0.34 → Pclass is important even though negative.

print("\n" + "=" * 55)
print("  STEP 8: Top 5 Features by Correlation with Survival")
print("=" * 55)

top5 = corr_with_survived.abs().sort_values(ascending=False).head(5)

for rank, (feature, value) in enumerate(top5.items(), start=1):
    direction = "positive" if corr_with_survived[feature] > 0 else "negative"
    print(f"  #{rank}  {feature:<15} r = {corr_with_survived[feature]:+.3f}  ({direction})")

# Expected ranking (approximate):
#   #1  Fare            +0.257  (higher fare → more likely to survive)
#   #2  Pclass          -0.338  (higher class number → less likely)
#   #3  FarePerPerson   +0.26
#   #4  IsAlone         -0.20   (alone passengers survived less)
#   #5  Age             -0.07


df.to_csv("titanic_cleaned.csv", index=False)


