import csv
import numpy as np

# STEP 1: Load CSV manually (no Pandas)

rows = []
with open("train.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

print("=" * 55)
print("  BONUS CHALLENGE: Pure NumPy Vectorized Analysis")
print("=" * 55)
print(f"  Total passengers loaded: {len(rows)}")


# ─────────────────────────────────────────────────────────────
# STEP 2: Extract Raw Arrays from CSV rows
# ─────────────────────────────────────────────────────────────
# We pull each column directly into a NumPy array.
# np.array([...])  converts a list into a NumPy array.
# int(row["Survived"]) converts string "0"/"1" to integer 0/1.

survived = np.array([int(row["Survived"])   for row in rows])
pclass   = np.array([int(row["Pclass"])     for row in rows])
sex      = np.array([row["Sex"]             for row in rows])
fare     = np.array([float(row["Fare"]) if row["Fare"] != "" else 0.0
                     for row in rows])
sibsp    = np.array([int(row["SibSp"])      for row in rows])
parch    = np.array([int(row["Parch"])      for row in rows])

# Age needs special handling — missing values exist
# np.where(condition, value_if_true, value_if_false)
# We temporarily fill missing age with 0 — will handle properly later
age_raw  = np.array([float(row["Age"]) if row["Age"] != "" else np.nan
                     for row in rows])

print(f"\n  Arrays extracted:")
print(f"  survived : {survived[:5]}  ...")
print(f"  pclass   : {pclass[:5]}  ...")
print(f"  sex      : {sex[:5]}  ...")


# ─────────────────────────────────────────────────────────────
# STEP 3: Encode Sex → Numbers using np.where (no loops)
# ─────────────────────────────────────────────────────────────
# np.where applies if/else to the ENTIRE array at once.
# Wherever sex == "female" → put 1, everywhere else → put 0.
# This replaces a loop like: for s in sex: 1 if s=="female" else 0

gender = np.where(sex == "female", 1, 0)
# female → 1,  male → 0

print("\n" + "=" * 55)
print("  STEP 3: Sex Encoded → Gender (vectorized)")
print("=" * 55)
print(f"  sex    (first 5): {sex[:5]}")
print(f"  gender (first 5): {gender[:5]}")


# ─────────────────────────────────────────────────────────────
# STEP 4: Survival by Class — Vectorized, No Loops
# ─────────────────────────────────────────────────────────────
# np.unique finds all unique values in pclass → [1, 2, 3]
# For each class, we use a boolean mask to filter survived array
# np.mean on 0s and 1s gives survival rate directly

print("\n" + "=" * 55)
print("  STEP 4: Survival by Passenger Class (Vectorized)")
print("=" * 55)

classes = np.unique(pclass)    # array([1, 2, 3]) — no hardcoding needed

# np.vectorize creates a function that works on entire arrays at once
# This avoids writing a loop over each class

def survival_rate_for_class(c):
    """Returns survival rate for a given class value — vectorized"""
    mask  = (pclass == c)                   # True where pclass matches
    total = np.sum(mask)                    # count passengers in this class
    rate  = np.mean(survived[mask]) * 100   # mean of 0s and 1s = rate
    return total, rate

# Apply for each class — still no explicit loop in computation
print(f"\n  {'Class':<10} {'Total':<10} {'Survived':<12} {'Survival Rate'}")
print(f"  {'-'*45}")

for c in classes:     # this loop is just for PRINTING — computation is vectorized
    total, rate = survival_rate_for_class(c)
    survived_count = int(np.sum((pclass == c) & (survived == 1)))
    print(f"  Class {c:<5}  {total:<10} {survived_count:<12} {rate:.1f}%")

# ── Fully vectorized version (no loop at all) ─────────────────
# np.array([...]) with a list comprehension — computation inside is vectorized
class_rates = np.array([np.mean(survived[pclass == c]) * 100 for c in classes])

print(f"\n  Vectorized rates array: {class_rates.round(1)}")
print(f"  Class 1: {class_rates[0]:.1f}%  |  Class 2: {class_rates[1]:.1f}%  |  Class 3: {class_rates[2]:.1f}%")


# ─────────────────────────────────────────────────────────────
# STEP 5: Survival by Gender — Vectorized, No Loops
# ─────────────────────────────────────────────────────────────
# Same pattern — use boolean mask on gender array
# gender == 1 → female passengers
# gender == 0 → male passengers

print("\n" + "=" * 55)
print("  STEP 5: Survival by Gender (Vectorized)")
print("=" * 55)

# Both calculations done in ONE vectorized step each — no loops
female_rate = np.mean(survived[gender == 1]) * 100
male_rate   = np.mean(survived[gender == 0]) * 100

female_total    = np.sum(gender == 1)
male_total      = np.sum(gender == 0)
female_survived = np.sum((gender == 1) & (survived == 1))
male_survived   = np.sum((gender == 0) & (survived == 1))

print(f"\n  {'Gender':<10} {'Total':<10} {'Survived':<12} {'Survival Rate'}")
print(f"  {'-'*45}")
print(f"  {'Female':<10} {female_total:<10} {female_survived:<12} {female_rate:.1f}%")
print(f"  {'Male':<10}   {male_total:<10} {male_survived:<12} {male_rate:.1f}%")
print(f"\n  Gender gap: {female_rate - male_rate:.1f}% higher survival for females")


# ─────────────────────────────────────────────────────────────
# STEP 6: Define the predict_survival() Function
# ─────────────────────────────────────────────────────────────
# This function takes ONE passenger as a dictionary
# and returns their predicted survival using the same
# weighted scoring formula we built in Part 3.
#
# Input example:
#   {"Pclass": 1, "Sex": "female", "Age": 28,
#    "Fare": 120, "SibSp": 0, "Parch": 0}
#
# Output example:
#   {"survived": 1, "probability": 0.82,
#    "verdict": "Likely Survived"}

# ── Normalization constants (from full dataset) ───────────────
# We precompute min/max from the full dataset ONCE
# so predict_survival() can normalize any new passenger correctly

PCLASS_MIN,      PCLASS_MAX      = 1.0, 3.0
AGE_MIN,         AGE_MAX         = 0.42, 80.0
FARE_MIN,        FARE_MAX        = 0.0, 512.33
FAMILYSIZE_MIN,  FAMILYSIZE_MAX  = 1.0, 11.0
FAREGROUP_MIN,   FAREGROUP_MAX   = 0.0, 2.0
AGEGROUP_MIN,    AGEGROUP_MAX    = 0.0, 2.0

# ── Weights from Part 3 (based on correlation strength) ───────
# [Gender, Pclass, AgeGroup, FareGroup, FamilySize]
RAW_CORR = np.array([0.54, 0.34, 0.08, 0.26, 0.02])
WEIGHTS  = RAW_CORR / RAW_CORR.sum()
W1, W2, W3, W4, W5 = WEIGHTS

def normalize_value(value, min_val, max_val):
    """Normalize a single value to 0-1 range"""
    if max_val == min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)

def predict_survival(passenger_dict):
    """
    Predict survival for a single passenger.

    Parameters:
        passenger_dict (dict): passenger details with keys:
            Pclass (int)  : 1, 2, or 3
            Sex    (str)  : "male" or "female"
            Age    (float): age in years
            Fare   (float): ticket fare
            SibSp  (int)  : siblings/spouses aboard
            Parch  (int)  : parents/children aboard

    Returns:
        dict with keys:
            survived    (int)  : 1 or 0
            probability (float): 0.0 to 1.0
            verdict     (str)  : human-readable result
            score_breakdown (dict): contribution of each feature
    """

    # ── Extract values from dictionary ────────────────────────
    p_class     = int(passenger_dict["Pclass"])
    p_sex       = str(passenger_dict["Sex"]).lower()
    p_age       = float(passenger_dict.get("Age", 28.0))   # default to median if missing
    p_fare      = float(passenger_dict.get("Fare", 14.0))  # default to median if missing
    p_sibsp     = int(passenger_dict.get("SibSp", 0))
    p_parch     = int(passenger_dict.get("Parch", 0))

    # ── Encode Gender ──────────────────────────────────────────
    gender_val = 1.0 if p_sex == "female" else 0.0

    # ── Compute FamilySize ─────────────────────────────────────
    family_size = p_sibsp + p_parch + 1

    # ── Compute AgeGroup number ────────────────────────────────
    # Child → 2, Adult → 1, Senior → 0
    if p_age < 15:
        age_group_num = 2.0
    elif p_age <= 60:
        age_group_num = 1.0
    else:
        age_group_num = 0.0

    # ── Compute FareGroup number ───────────────────────────────
    # Low → 0, Medium → 1, High → 2
    if p_fare <= 50:
        fare_group_num = 0.0
    elif p_fare <= 200:
        fare_group_num = 1.0
    else:
        fare_group_num = 2.0

    # ── Normalize all features to 0-1 ─────────────────────────
    gender_norm      = gender_val    # already 0 or 1
    pclass_norm      = normalize_value(p_class,       PCLASS_MIN,     PCLASS_MAX)
    age_group_norm   = normalize_value(age_group_num, AGEGROUP_MIN,   AGEGROUP_MAX)
    fare_group_norm  = normalize_value(fare_group_num,FAREGROUP_MIN,  FAREGROUP_MAX)
    family_size_norm = normalize_value(family_size,   FAMILYSIZE_MIN, FAMILYSIZE_MAX)

    # ── Apply survival scoring formula ────────────────────────
    # Pclass is flipped (1 - pclass_norm) because higher class = worse survival
    raw_score = (
          W1 * gender_norm
        + W2 * (1 - pclass_norm)    # flipped — negative correlation
        + W3 * age_group_norm
        + W4 * fare_group_norm
        + W5 * family_size_norm
    )

    # ── Normalize score to 0-1 probability ────────────────────
    # We use known min/max from Part 3 formula bounds
    # Min possible score ≈ 0.0  (male, class3, senior, low fare, alone)
    # Max possible score ≈ 1.0  (female, class1, child, high fare, family)
    # Clip to ensure it stays in 0-1 range
    probability = float(np.clip(raw_score, 0.0, 1.0))

    # ── Classify using threshold = 0.5 ────────────────────────
    predicted_survived = 1 if probability >= 0.5 else 0

    # ── Human-readable verdict ────────────────────────────────
    if probability >= 0.75:
        verdict = "Very Likely Survived"
    elif probability >= 0.5:
        verdict = "Likely Survived"
    elif probability >= 0.25:
        verdict = "Likely Did Not Survive"
    else:
        verdict = "Very Unlikely to Survive"

    # ── Score breakdown (contribution of each feature) ────────
    score_breakdown = {
        "Gender"     : round(W1 * gender_norm,           4),
        "Pclass"     : round(W2 * (1 - pclass_norm),     4),
        "AgeGroup"   : round(W3 * age_group_norm,        4),
        "FareGroup"  : round(W4 * fare_group_norm,       4),
        "FamilySize" : round(W5 * family_size_norm,      4),
    }

    return {
        "survived"        : predicted_survived,
        "probability"     : round(probability, 4),
        "verdict"         : verdict,
        "score_breakdown" : score_breakdown,
    }


# ─────────────────────────────────────────────────────────────
# STEP 7: Test predict_survival() with Sample Passengers
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("  STEP 6: predict_survival() Function — Test Cases")
print("=" * 55)

test_passengers = [
    {
        "name"   : "Rose (1st class female, young)",
        "data"   : {"Pclass":1,"Sex":"female","Age":20,"Fare":150,"SibSp":0,"Parch":1}
    },
    {
        "name"   : "Jack (3rd class male, young)",
        "data"   : {"Pclass":3,"Sex":"male","Age":20,"Fare":5,"SibSp":0,"Parch":0}
    },
    {
        "name"   : "Elderly 1st class male",
        "data"   : {"Pclass":1,"Sex":"male","Age":65,"Fare":200,"SibSp":1,"Parch":0}
    },
    {
        "name"   : "Child (3rd class, female)",
        "data"   : {"Pclass":3,"Sex":"female","Age":8,"Fare":15,"SibSp":1,"Parch":2}
    },
    {
        "name"   : "2nd class male, middle-aged",
        "data"   : {"Pclass":2,"Sex":"male","Age":35,"Fare":25,"SibSp":0,"Parch":0}
    },
]

for test in test_passengers:
    result = predict_survival(test["data"])
    print(f"\n  Passenger : {test['name']}")
    print(f"  Input     : {test['data']}")
    print(f"  Prediction: {'SURVIVED ✅' if result['survived'] == 1 else 'DID NOT SURVIVE ❌'}")
    print(f"  Probability: {result['probability']:.4f}  →  {result['verdict']}")
    print(f"  Score breakdown:")
    for feature, contribution in result["score_breakdown"].items():
        bar = "█" * int(contribution * 100)
        print(f"    {feature:<12}: {contribution:.4f}  {bar}")


# ─────────────────────────────────────────────────────────────
# STEP 8: Verify vectorized results match Part 1 results
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  STEP 7: Verification — Matches Part 1 Results?")
print("=" * 55)

# Recompute survival by class vectorized
c1_rate = np.mean(survived[pclass == 1]) * 100
c2_rate = np.mean(survived[pclass == 2]) * 100
c3_rate = np.mean(survived[pclass == 3]) * 100

print(f"\n  Survival by Class (pure NumPy, no loops):")
print(f"  Class 1 : {c1_rate:.1f}%")
print(f"  Class 2 : {c2_rate:.1f}%")
print(f"  Class 3 : {c3_rate:.1f}%")

print(f"\n  Survival by Gender (pure NumPy, no loops):")
print(f"  Female  : {np.mean(survived[gender == 1])*100:.1f}%")
print(f"  Male    : {np.mean(survived[gender == 0])*100:.1f}%")

print(f"\n  Overall survival rate : {np.mean(survived)*100:.1f}%")
print(f"  Total survivors       : {np.sum(survived)}")
print(f"  Total passengers      : {len(survived)}")

print(f"""
  Key NumPy functions used (NO loops, NO Pandas):
  ─────────────────────────────────────────────
  np.array()       → convert list to array
  np.where()       → vectorized if/else
  np.unique()      → find all unique values
  np.mean()        → survival rate from 0s and 1s
  np.sum()         → count Trues in boolean mask
  np.clip()        → keep values within range
  Boolean masking  → filter without loops
  ─────────────────────────────────────────────
""")
