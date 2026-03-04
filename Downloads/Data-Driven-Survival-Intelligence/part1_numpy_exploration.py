import csv
import numpy as np

# STEP 1: Load CSV manually using Python's csv module
# We read the file row-by-row and store each row as a dict.
# This is what Pandas does internally — we're doing it by hand.

rows = []
with open("train.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

print(f"Total passengers loaded: {len(rows)}")
print(f"Columns available: {list(rows[0].keys())}\n")


# STEP 2: Extract Age column into a Python List
# Age values can be empty strings ("") for missing entries.
# We first collect ALL age values (including empty) as a raw list.

age_raw = [row["Age"] for row in rows]
print(f"Total Age entries (including missing): {len(age_raw)}")
print(f"Sample (first 10): {age_raw[:10]}\n")


# STEP 3: Remove missing ages manually
# We keep only rows where Age is NOT an empty string.
# We also track which row indices have valid ages — needed later
# when we compare survival rates.

valid_indices = []
age_clean = []

for i, val in enumerate(age_raw):
    if val != "":                       # only keep non-missing
        age_clean.append(float(val))    # convert string → float
        valid_indices.append(i)

print(f"Ages with valid values : {len(age_clean)}")
print(f"Ages missing (dropped) : {len(age_raw) - len(age_clean)}\n")


# STEP 4: Convert to NumPy array & compute statistics

age_array = np.array(age_clean)

mean_age   = np.mean(age_array)
median_age = np.median(age_array)
std_age    = np.std(age_array)

print("=" * 45)
print("  AGE STATISTICS")
print("=" * 45)
print(f"  Mean Age      : {mean_age:.2f} years")
print(f"  Median Age    : {median_age:.2f} years")
print(f"  Std Deviation : {std_age:.2f} years")
print("=" * 45)

# Insight note:
# Mean > Median → distribution is slightly right-skewed
# (a few older passengers pull the mean up)


# STEP 5: Create a NumPy array of Fare
# Fare has no missing values in train.csv, but we guard anyway.

fare_list = []
for row in rows:
    if row["Fare"] != "":
        fare_list.append(float(row["Fare"]))

fare_array = np.array(fare_list)
print(f"\nTotal Fare entries loaded: {len(fare_array)}")


# STEP 6: Identify Top 10% and Bottom 10% Fare passengers
# np.percentile gives the threshold values.
# We then filter the array to find passengers above/below those thresholds.

top_10_threshold    = np.percentile(fare_array, 90)   # 90th percentile
bottom_10_threshold = np.percentile(fare_array, 10)   # 10th percentile

top_10_fares    = fare_array[fare_array >= top_10_threshold]
bottom_10_fares = fare_array[fare_array <= bottom_10_threshold]

print("\n" + "=" * 45)
print("  FARE ANALYSIS")
print("=" * 45)
print(f"  Top 10% threshold    : £{top_10_threshold:.2f}")
print(f"  # passengers in top 10%  : {len(top_10_fares)}")
print(f"  Avg fare (top 10%)   : £{np.mean(top_10_fares):.2f}")
print()
print(f"  Bottom 10% threshold : £{bottom_10_threshold:.2f}")
print(f"  # passengers in bottom 10% : {len(bottom_10_fares)}")
print(f"  Avg fare (bottom 10%): £{np.mean(bottom_10_fares):.2f}")
print("=" * 45)


# STEP 7: Compare Survival Rate by Age Group
# We use ONLY the rows that had a valid (non-missing) age.
# For each of those rows, we pair (age, survived).
# Then we filter by group and compute survival rate.

# Build parallel arrays: age and survived (only for valid-age passengers)
survived_valid = np.array([int(rows[i]["Survived"]) for i in valid_indices])

# Group 1: Children  → Age < 15
# Group 2: Adults    → 15 ≤ Age ≤ 60
# Group 3: Seniors   → Age > 60

mask_child  = age_array < 15
mask_adult  = (age_array >= 15) & (age_array <= 60)
mask_senior = age_array > 60

def survival_rate(survived_arr, mask):
    """Returns survival rate (0–1) for a boolean mask."""
    group = survived_arr[mask]
    if len(group) == 0:
        return 0.0, 0
    return np.mean(group), len(group)

rate_child,  n_child  = survival_rate(survived_valid, mask_child)
rate_adult,  n_adult  = survival_rate(survived_valid, mask_adult)
rate_senior, n_senior = survival_rate(survived_valid, mask_senior)

print("\n" + "=" * 45)
print("  SURVIVAL RATE BY AGE GROUP")
print("=" * 45)
print(f"  Children (Age < 15)   : {rate_child*100:.1f}%  (n={n_child})")
print(f"  Adults   (15–60)      : {rate_adult*100:.1f}%  (n={n_adult})")
print(f"  Seniors  (Age > 60)   : {rate_senior*100:.1f}%  (n={n_senior})")
print("=" * 45)


