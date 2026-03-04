# Titanic — Data-Driven Survival Intelligence
### Insight Report | IIT Gandhinagar | Data Analytics Assignment
**Dataset:** Kaggle Titanic (train.csv, 891 passengers) | **Tools:** Python, NumPy, Pandas

---

## Part 1 — Raw Data Exploration

### Insight Question
> Is age linearly related to survival? Justify statistically.

#### Age Statistics

| Metric | Value | Interpretation |
|---|---|---|
| Mean Age | 29.7 yrs | Average passenger age |
| Median Age | 28.0 yrs | Half passengers younger, half older |
| Std Deviation | 14.5 yrs | Ages spread widely across all groups |
| Pearson r (Age vs Survived) | −0.07 | Near-zero = no linear relationship |

#### Survival Rate by Age Group

| Age Group | Condition | Count | Survival Rate |
|---|---|---|---|
| Children | Age < 15 | 83 | ~58% |
| Adults | Age 15–60 | 588 | ~37% |
| Seniors | Age > 60 | 43 | ~22% |

#### Answer

The Pearson correlation coefficient between Age and Survived is **r = −0.07**, which is extremely close to zero. This statistically confirms there is **no meaningful linear relationship** between age and survival.

However, survival rates across age groups (58% children, 37% adults, 22% seniors) clearly show that age does influence survival — just not in a straight line. The relationship is **non-linear and categorical**. A simple linear model would fail to capture this pattern. Age must be treated as grouped bins (Child / Adult / Senior) rather than a continuous variable to extract predictive value.

---

## Part 2 — Advanced Pandas Engineering

### Insight Question
> Does wealth dominate gender in predicting survival? Support with grouped statistics.

#### Survival by Gender and Class

| Group | Survival Rate | Conclusion |
|---|---|---|
| Female (all classes) | ~74% | Gender = strongest predictor |
| Male (all classes) | ~18% | 56% gap driven by gender |
| 1st Class (all genders) | ~63% | Wealth amplifies survival |
| 3rd Class (all genders) | ~24% | Wealth gap = 39% |
| Female + 1st Class | ~97% | Best combined outcome |
| Male + 3rd Class | ~13% | Worst combined outcome |

#### Answer

Gender is the stronger predictor of survival. Even within the lowest fare group, females survived at significantly higher rates than males. However, wealth (Pclass / FareGroup) substantially **amplifies** the gender advantage — a 1st class female had ~97% survival while a 3rd class male had only ~13%.

Neither factor alone fully explains survival. **Gender is the primary driver; wealth is a powerful amplifier.** This interaction effect means both features must be included together for accurate prediction.

---

## Part 3 — NumPy Survival Scoring

### Insight Question
> Does your handcrafted score outperform random guessing significantly?

#### Model Performance vs Random Guessing

| Metric | Our Model | Random Guessing |
|---|---|---|
| Accuracy | ~78% | ~50% |
| Precision | ~74% | ~50% |
| Recall | ~68% | ~50% |
| Improvement | **+28%** | Baseline |

#### Answer

Yes — the handcrafted NumPy model **significantly outperforms random guessing**. Random coin-flipping gives ~50% accuracy. Our weighted scoring formula achieves approximately **78% accuracy — a +28% improvement**.

The model works because it correctly captures the two strongest signals: gender (weight = 0.435) and passenger class (weight = 0.274). Limitations include no interaction terms between features and weights based only on linear correlation. A logistic regression model would perform better, but this manual approach is fully transparent and explainable without any black-box ML library.

---

## Part 4 — Executive Challenge

### Q1. If Titanic happened today, who should be prioritized for rescue?

**Priority 1 — Children (Age < 15)**
Children have limited physical ability to self-rescue. They cannot swim long distances or hold on in rough water. Survival rate was ~58% but required active rescue assistance. Universally recognized as the highest moral and legal priority.

**Priority 2 — Elderly and Disabled Passengers**
Limited mobility reduces ability to reach lifeboats independently. Need earliest evacuation assistance regardless of class or gender.

**Priority 3 — Passengers in Lower Decks (Economy Class)**
In 1912, Class 3 passengers had only ~24% survival vs ~63% for Class 1. The structural disadvantage of lower deck positioning means rescue crews today should **actively reach lower decks first** to compensate — reversing the 1912 pattern.

**Modern Context:**
SOLAS (International Convention for Safety of Life at Sea) mandates equal lifeboat access for ALL passengers regardless of class. A modern rescue system must not replicate the class-based discrimination of 1912. The correct modern priority order is:
1. Children (any class, any gender)
2. Elderly and disabled passengers
3. Remaining passengers — **no class distinction**

---

### Q2. Identify 3 ethical concerns in automated survival prediction.

**Concern 1 — Encoding Historical Discrimination**

Our model learned that Class 3 passengers had very low survival (~24%). But this was because crew physically blocked them from lifeboats — not because they were less deserving. Deploying this model would automate and permanently encode the original discrimination into future rescue decisions.

Real-world parallel: Amazon's AI hiring tool (2018) was scrapped after it learned to reject women because historical hiring data was male-dominated. The model repeated the bias it was trained on.

**Concern 2 — Gender as a Binary Feature**

The model treats gender as only 0 (male) or 1 (female), completely excluding non-binary and transgender individuals. Assigning life-or-death priority based on binary gender violates modern equality laws and human rights frameworks. Any use of protected attributes (gender, race, religion) in automated life-or-death decisions carries serious legal and ethical risk.

**Concern 3 — Lack of Transparency and Accountability**

The model produces a score without explaining why a specific passenger scored low. If an automated system causes a wrong rescue decision, who is accountable? Affected individuals or families cannot challenge an unexplainable decision. This violates the "right to explanation" under GDPR and similar data protection laws. Any life-or-death automated system must be explainable, auditable, and always overridable by a human decision-maker.

---

### Q3. If this were an insurance underwriting dataset, how would your logic change?

**Change 1 — Target Variable**

Titanic predicts Survived (0/1). Insurance predicts Claim Filed (0/1) or Claim Amount (in pounds). The model structure stays the same but what we are predicting changes completely — from survival to financial risk.

**Change 2 — Remove Protected Attributes**

The EU Gender Directive (2012) legally prohibits using gender to set insurance premiums. Direct use of socioeconomic class is also restricted in many jurisdictions. Features like Gender and Pclass must be removed and replaced with legally permitted variables such as driving history, property type, health records, or geographic location.

**Change 3 — Redefine Weights**

| Feature | Titanic Weight | Insurance Weight |
|---|---|---|
| Gender | 0.435 | 0.000 (removed — legally protected) |
| Pclass | 0.274 | 0.000 (removed — discriminatory) |
| FareGroup | 0.210 | 0.000 (replaced) |
| Age | 0.065 | Higher — older = more claims |
| Pre-existing conditions | Not present | Highest weight |
| Geographic risk zone | Not present | Moderate weight |
| FamilySize | 0.016 | Moderate — more dependents = more risk |

**Change 4 — Output Changes**

Titanic output is binary: Survived or Died. Insurance output is a **premium price** (in pounds), an approval/rejection decision, and a risk tier (Low / Medium / High). The scoring formula structure — weighted sum, normalize, threshold — remains identical but all inputs and outputs are redefined for the business context.

---

