import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# STEP 1: Load the Cleaned Dataset from Part 2
# ─────────────────────────────────────────────────────────────
# We use titanic_cleaned.csv saved at the end of Part 2.
# This already has:
#   → Missing values filled
#   → FamilySize, IsAlone, FarePerPerson columns
#   → AgeGroup, FareGroup bin columns

df = pd.read_csv("titanic_cleaned.csv")

print("=" * 55)
print("  STEP 1: Cleaned Dataset Loaded")
print("=" * 55)
print(f"  Rows    : {df.shape[0]}")
print(f"  Columns : {list(df.columns)}")


# ─────────────────────────────────────────────────────────────
# STEP 2: Encode Text Columns into Numbers
# ─────────────────────────────────────────────────────────────
# NumPy can only do math on numbers.
# Text columns like Sex, AgeGroup, FareGroup must be converted.

# ── 2A: Gender ────────────────────────────────────────────────
# female → 1   (higher survival rate in data)
# male   → 0
df["Gender"] = (df["Sex"] == "female").astype(int)

# ── 2B: AgeGroup → number ─────────────────────────────────────
# Child  → 2  (highest survival rate ~59%)
# Adult  → 1  (medium survival rate  ~37%)
# Senior → 0  (lowest survival rate  ~22%)
# Higher number = historically better survival chance
age_map = {"Child": 2, "Adult": 1, "Senior": 0}
df["AgeGroupNum"] = df["AgeGroup"].map(age_map)

# ── 2C: FareGroup → number ────────────────────────────────────
# Low    → 0
# Medium → 1
# High   → 2
fare_map = {"Low": 0, "Medium": 1, "High": 2}
df["FareGroupNum"] = df["FareGroup"].map(fare_map)

print("\n" + "=" * 55)
print("  STEP 2: Text Columns Encoded to Numbers")
print("=" * 55)
print(df[["Sex", "Gender", "AgeGroup", "AgeGroupNum",
          "FareGroup", "FareGroupNum"]].head(8).to_string())


# ─────────────────────────────────────────────────────────────
# STEP 3: Extract Feature Arrays using NumPy
# ─────────────────────────────────────────────────────────────
# Convert each feature column into a NumPy array.
# This allows us to do fast math operations on them.

gender      = df["Gender"].to_numpy()        # 0 or 1
pclass      = df["Pclass"].to_numpy()        # 1, 2, or 3
age_group   = df["AgeGroupNum"].to_numpy()   # 0, 1, or 2
fare_group  = df["FareGroupNum"].to_numpy()  # 0, 1, or 2
family_size = df["FamilySize"].to_numpy()    # 1 to 11
actual      = df["Survived"].to_numpy()      # 0 or 1 (ground truth)

print("\n" + "=" * 55)
print("  STEP 3: Feature Arrays Extracted")
print("=" * 55)
print(f"  gender      (first 5): {gender[:5]}")
print(f"  pclass      (first 5): {pclass[:5]}")
print(f"  age_group   (first 5): {age_group[:5]}")
print(f"  fare_group  (first 5): {fare_group[:5]}")
print(f"  family_size (first 5): {family_size[:5]}")
print(f"  actual      (first 5): {actual[:5]}")


# ─────────────────────────────────────────────────────────────
# STEP 4: Normalize All Features to 0–1 Scale
# ─────────────────────────────────────────────────────────────
# Formula: normalized = (value - min) / (max - min)
#
# Why normalize?
# Without normalization:
#   FamilySize ranges 1–11 → large numbers
#   Pclass ranges 1–3      → small numbers
# FamilySize would dominate the score just because of bigger numbers,
# not because it is more important.
# After normalization everything is between 0 and 1 — fair comparison.

def normalize(arr):
    """Normalize a NumPy array to range 0–1"""
    min_val = arr.min()
    max_val = arr.max()
    # avoid division by zero (if all values are same)
    if max_val == min_val:
        return np.zeros_like(arr, dtype=float)
    return (arr - min_val) / (max_val - min_val)

gender_norm      = normalize(gender.astype(float))
pclass_norm      = normalize(pclass.astype(float))
age_group_norm   = normalize(age_group.astype(float))
fare_group_norm  = normalize(fare_group.astype(float))
family_size_norm = normalize(family_size.astype(float))

print("\n" + "=" * 55)
print("  STEP 4: Features Normalized to 0–1")
print("=" * 55)
print(f"  gender_norm      (first 5): {gender_norm[:5].round(3)}")
print(f"  pclass_norm      (first 5): {pclass_norm[:5].round(3)}")
print(f"  age_group_norm   (first 5): {age_group_norm[:5].round(3)}")
print(f"  fare_group_norm  (first 5): {fare_group_norm[:5].round(3)}")
print(f"  family_size_norm (first 5): {family_size_norm[:5].round(3)}")


# ─────────────────────────────────────────────────────────────
# STEP 5: Assign Weights Based on Correlation Strength
# ─────────────────────────────────────────────────────────────
# From Part 2 correlation matrix, we found approximate correlations
# between each feature and Survived.
#
# We use ABSOLUTE values because:
#   Pclass has r = -0.34 (negative) but is still very important.
#   abs(-0.34) = 0.34 → correctly ranked as important.
#
# Then we divide each by the total sum → weights add up to exactly 1.0
# This is called "proportional weighting".

# Approximate correlations with Survived from Part 2:
raw_corr = np.array([
    0.54,   # Gender      (female → strong positive)
    0.34,   # Pclass      (abs value, negative in reality)
    0.08,   # AgeGroup    (weak)
    0.26,   # FareGroup   (moderate positive)
    0.02    # FamilySize  (very weak)
])

# Divide by sum so all weights add up to 1.0
weights = raw_corr / raw_corr.sum()

print("\n" + "=" * 55)
print("  STEP 5: Weights Assigned")
print("=" * 55)
features = ["Gender", "Pclass", "AgeGroup", "FareGroup", "FamilySize"]
for f, r, w in zip(features, raw_corr, weights):
    print(f"  {f:<15} corr={r:.2f}  weight={w:.3f}")
print(f"\n  Total weight sum: {weights.sum():.3f}  (should be 1.000)")


# ─────────────────────────────────────────────────────────────
# STEP 6: Compute Survival Score
# ─────────────────────────────────────────────────────────────
# SurvivalScore = w1*Gender + w2*Pclass + w3*AgeGroup
#               + w4*FareGroup + w5*FamilySize
#
# BUT — Pclass is NEGATIVELY correlated with survival.
# Pclass 1 = best,  Pclass 3 = worst.
# After normalization: Pclass 1 → 0.0,  Pclass 3 → 1.0
# So higher pclass_norm = WORSE chance.
# We must FLIP it: use (1 - pclass_norm) so that:
#   Pclass 1 → 1.0 (good)
#   Pclass 3 → 0.0 (bad)

w1, w2, w3, w4, w5 = weights

survival_score = (
      w1 * gender_norm
    + w2 * (1 - pclass_norm)    # flipped because negative correlation
    + w3 * age_group_norm
    + w4 * fare_group_norm
    + w5 * family_size_norm
)

print("\n" + "=" * 55)
print("  STEP 6: Raw Survival Scores Computed")
print("=" * 55)
print(f"  Score range  : {survival_score.min():.3f}  to  {survival_score.max():.3f}")
print(f"  First 5 scores: {survival_score[:5].round(3)}")


# ─────────────────────────────────────────────────────────────
# STEP 7: Normalize Score → Survival Probability (0 to 1)
# ─────────────────────────────────────────────────────────────
# The raw score is already between 0 and 1 roughly,
# but we normalize again to make sure it is exactly 0–1.
# This makes it a proper "probability" value.

survival_prob = normalize(survival_score)

print("\n" + "=" * 55)
print("  STEP 7: Survival Probability (0 to 1)")
print("=" * 55)
print(f"  Probability range  : {survival_prob.min():.3f}  to  {survival_prob.max():.3f}")
print(f"  First 5 probs: {survival_prob[:5].round(3)}")
# Values close to 1.0 → high chance of survival
# Values close to 0.0 → low chance of survival


# ─────────────────────────────────────────────────────────────
# STEP 8: Classify Using Threshold = 0.5
# ─────────────────────────────────────────────────────────────
# If probability >= 0.5 → predict Survived = 1
# If probability <  0.5 → predict Survived = 0

threshold = 0.5
predicted = (survival_prob >= threshold).astype(int)

print("\n" + "=" * 55)
print("  STEP 8: Predictions Made (threshold = 0.5)")
print("=" * 55)
print(f"  Predicted survived (1): {predicted.sum()}")
print(f"  Predicted died    (0): {(predicted == 0).sum()}")
print(f"  Actual survived   (1): {actual.sum()}")
print(f"  Actual died       (0): {(actual == 0).sum()}")


# ─────────────────────────────────────────────────────────────
# STEP 9: Confusion Matrix (manually using NumPy)
# ─────────────────────────────────────────────────────────────
# We use boolean masks to count each of the 4 outcomes.
#
#                   Predicted 0    Predicted 1
# Actual 0 (died)       TN              FP
# Actual 1 (survived)   FN              TP

TP = np.sum((predicted == 1) & (actual == 1))  # correctly predicted survived
TN = np.sum((predicted == 0) & (actual == 0))  # correctly predicted died
FP = np.sum((predicted == 1) & (actual == 0))  # said survived, actually died
FN = np.sum((predicted == 0) & (actual == 1))  # said died, actually survived

print("\n" + "=" * 55)
print("  STEP 9: Confusion Matrix")
print("=" * 55)
print(f"\n               Predicted 0    Predicted 1")
print(f"  Actual 0  |     TN={TN:<6}     FP={FP}")
print(f"  Actual 1  |     FN={FN:<6}     TP={TP}")
print()
print(f"  TP (True Positive)  = {TP}  → predicted survived, actually survived ✅")
print(f"  TN (True Negative)  = {TN}  → predicted died,     actually died     ✅")
print(f"  FP (False Positive) = {FP}   → predicted survived, actually died     ❌")
print(f"  FN (False Negative) = {FN}   → predicted died,     actually survived ❌")


# ─────────────────────────────────────────────────────────────
# STEP 10: Compute Accuracy, Precision, Recall
# ─────────────────────────────────────────────────────────────
# All computed manually using NumPy values — no sklearn allowed.

# Accuracy = how often were we right overall?
accuracy  = (TP + TN) / (TP + TN + FP + FN)

# Precision = of all we predicted survived, how many actually did?
# High precision → when we say survived, we are usually right
precision = TP / (TP + FP) if (TP + FP) > 0 else 0

# Recall = of all who actually survived, how many did we correctly catch?
# High recall → we don't miss many actual survivors
recall    = TP / (TP + FN) if (TP + FN) > 0 else 0

# Random guessing baseline = 50% accuracy
random_accuracy = 0.50

print("\n" + "=" * 55)
print("  STEP 10: Performance Metrics")
print("=" * 55)
print(f"  Accuracy  : {accuracy*100:.2f}%")
print(f"  Precision : {precision*100:.2f}%")
print(f"  Recall    : {recall*100:.2f}%")
print(f"\n  Random Guessing Baseline : 50.00%")
print(f"  Our Model Accuracy       : {accuracy*100:.2f}%")
improvement = (accuracy - random_accuracy) * 100
print(f"  Improvement over random  : +{improvement:.2f}%")
