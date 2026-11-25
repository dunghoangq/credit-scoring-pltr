# credit-scoring-pltr
Apply ML with Logistic Regression in Credit Scoring, based on Dumitrescu's et al.

## 1. The problem with LR

Logistic Regression looks at each feature separately, then gives it points and adds them up.

e.g., each missed payment = +0.2 points applies to everyone, whatever else is true about them.

## 2. How PLTR fixes the problem

PLTR doesn't change the LR. It changes **what you feed it**. It invents new yes/no columns that describe a combination, and the same LR then scores those columns like any other.

A loan officer has a printed points sheet, one row per fact. The experienced officer knows things the sheet can't say:

- "A maxed-out card is normal for a salaried customer who pays it off monthly. A maxed-out card plus a late payment means they're already slipping."
- "Below roughly 8 million VND a month there's no cushion at all."

PLTR adds new rows for those situations, such as "card ≥70% used AND at least one late payment: +1.45 points". It's still a points sheet. The sheet just now contains rows that look at two facts at once.

### 2.1 Example

100,000 invented applicants, 60% for training and 40% for testing, with a 10.3% default rate. It isn't real bank data and it isn't from the paper.

The hidden truth:

- Missed payments only matter seriously in one situation: utilisation ≥ 0.70 and at least one missed payment. That group gets +1.6 points. Everywhere else, a missed payment adds a trivial 0.1.
- Income has a cliff: below 8 million, +1.2 points. Above that, extra income does almost nothing.
- Tenure and enquiries are ordinary and smooth (−0.1 per year, +0.15 per enquiry). Utilisation also has a mild smooth effect.

**What plain LR does**

| Group                 | Applicants | Observed default | LR predicts | PLTR predicts |
| --------------------- | ---------- | ---------------- | ----------- | ------------- |
| High util, missed ≥ 1 | 2,178      | 38.0%            | 20.2%       | 36.0%         |
| High util, no missed  | 1,137      | 9.1%             | 15.5%       | 10.7%         |
| Low util, no missed   | 19,151     | 7.9%             | 7.9%        | 8.0%          |
| Low util, missed ≥ 1  | 17,534     | 9.5%             | 11.3%       | 9.5%          |
| Income < 8m	4,173     | 23.7%      | 15.1%            | 22.5%       |
| Income ≥ 8m	35,827    | 8.7%       | 9.7%             | 8.9%        |

LR fitted +0.20 points per missed payment and +1.81 per 100% of utilisation. Look at what that forces:

- **Missed payments**: the true effect is about 0.1 for most people and about 1.7 inside the danger zone. LR has one number, so it compromises. It under-predicts the danger zone (20% vs 38%) and over-penalises people with only a minor late payment (11.3% vs 9.5%).
- **Utilisation**: much of the danger from high utilisation comes from that one combination. LR can't isolate it, so it spreads the penalty over everyone with high utilisation. A heavy-but-punctual card user gets wrongly flagged (15.5% predicted, 9.1% actual).
- **Income**: LR draws a smooth ramp, but reality is a cliff. It under-predicts the poorest group (15.1% vs 23.7%).

**How PLTR fixes it**

**Step 1: the scouts**. PLTR grows a tiny tree for each feature (5 trees, one cut each) and for each pair (10 trees, two cuts each). A tree simply tries every possible cut and keeps the one that makes the groups' default rates most different. Two examples:

- On income alone, the scout found a cut at income < 7.99. Nobody told it 8. It recovered the cliff from the data.
- On the pair (utilisation, missed payments), the scout cut first at utilisation 0.70. Then, within the high-utilisation branch, it cut at missed < 0.5, which for a whole-number count means "zero". That gives three groups: low utilisation; high utilisation with no misses; and high utilisation with at least one miss, the danger zone.

**Step 2: leaves become columns**. Each group becomes a 0/1 column ("is this applicant in this group?"). After removing duplicate rules, there were 15 such columns. The table now has the 5 original features plus 15 rule columns.

**Step 3: ordinary LR on 20 columns**. An adaptive lasso is a penalty that shrinks or deletes weak columns. The biggest terms it kept:

| Points | Term                               |
| ------ | ---------------------------------- |
| −1.45  | util < 0.70                        |
| −1.45  | util ≥ 0.70 AND no missed payments |
| +1.10  | income < 7.99                      |

The first two are mirror pieces of the same idea. Together they say: everyone outside the danger zone gets −1.45 relative to someone inside it. That is the interaction, recovered at about +1.45 against my hidden +1.6. The income cliff came out at +1.10 against the true +1.2. The other terms were all small (|points| ≤ 0.3).

**Step 4: the result**. The scoring is still addition. Take two applicants who share utilisation 0.85, income 14m, tenure 3 years and 1 enquiry, but one has 2 missed payments and the other has none. Under my hidden rule, their true default risks are about 37% vs 9%. A single "+0.2 per miss" number can't produce that gap, but a +1.45 row that switches on for the danger zone can. This is why PLTR's group predictions in the table sit so close to reality.

### 2.2 Why not use more than 2 features?

#### 2.2.1 Explainability (the paper's reason)

A pair rule fits in one sentence to a customer, a regulator or a model validator: "Your card is over 70% used and you've had a late payment." With four conditions, the rule becomes a mouthful: "…and your income is under 8m and you've been with us under two years."

The paper measures this directly. PLTR's rules never exceed 2 conditions, whereas the random forest's longest rules average 32.2 conditions on the Kaggle data. The depth dial moves you along that same axis.

#### 2.2.2 Data thinness

Each extra condition shrinks the group of applicants who meet it. Suppose each condition is met by about 20% of applicants, independently (a simplification). With 100,000 applicants and a 10% default rate:

| Conditions in the rule | Share of applicants | Applicants | Expected defaulters |
| ---------------------- | ------------------- | ---------- | ------------------- |
| 1                      | 20%                 | 20,000     | 2,000               |
| 2                      | 4%                  | 4,000      | 400                 |
| 3                      | 0.8%                | 800        | 80                  |
| 4                      | 0.16%               | 160        | 16                  |

A coefficient estimated from 16 defaulters is mostly noise. In the analogy, a scout who describes a tiny village of 160 people can't give you a reliable rule about it.

#### 2.2.3 The search explodes, and so do lucky accidents

With p = 20 features, the number of candidate combinations is:

- Pairs: 190
- Triples: 1,140
- Quadruples: 4,845

The more candidate rules you test against the labels, the more likely some look predictive by pure chance. The lasso helps remove them, but remember that the rules are built using the training labels, so the penalty-tuning step is a little optimistic. More candidates make that worse.

**Can pairs approximate a three-way effect?**

Partly, but it overstates. Say the danger is +1.6 only when all three of A, B and C hold. Three pair rules (A&B, A&C, B&C), each worth about +0.6, give an applicant with all three +1.8, which is close. But an applicant with only A and B also gets +0.6, though the truth for them is 0. Pairs blur the edges. Whether that blur matters is an empirical question.

### 3. Dataset

UCI Machine Learning Repository, "Default of Credit Card Clients" (Yeh & Lien, 2009). UCI lists 30,000 rows, 23 features, no missing values, and a CC BY 4.0 licence.

https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset

## 4. Replication on real data: what actually happened

**Question.** Does PLTR (Dumitrescu et al. 2022) beat logistic regression on the UCI Taiwan
credit-card default data, and if so, why?

### 4.1 Setup
- 30,000 clients, 22.1% default. SEX dropped (protected attribute); EDUCATION {0,5,6} merged into 4.
- PAY_* clipped at 3 (default rates above 3 are tiny, noisy groups).
- Final protocol: **5×2 cross-validation** on all rows, with paired AUC differences per fold
  (Alpaydin F-test). An earlier 70/30 single split was used for debugging only.
- Models:
  - LR-A: naive logistic regression, all columns scaled.
  - LR-B: LR with sensible encoding (signed-log money, one-hot categories, PAY as categories).
  - ALasso-B: LR-B with PLTR's adaptive-lasso penalty and no rules (control).
  - PLTR: rules from one-variable trees and pair trees (3 leaves, both variables used) + adaptive lasso.
  - WoE-LR: binned scorecard. RF/HGB: tree references.
  - HGB-add (additive boosting) and HGB-d2 (depth-2 boosting): controls separating curvature from interactions.

### 4.2 Results (mean AUC over 10 folds)
| Model | AUC | Note |
|---|---|---|
| LR-A (naive) | 0.7216 | |
| WoE-LR | 0.7697 | simple scorecard |
| HGB additive | 0.7729 | flexible but no interactions |
| LR-B | 0.7738 | encoded properly |
| PLTR singles-only | 0.7710 | |
| **PLTR full** | **0.7759** | |
| HGB depth-2 | 0.7784 | |
| HGB | 0.7801 | |
| RF | 0.7816 | |

| Comparison | ΔAUC | Folds won | p |
|---|---|---|---|
| PLTR − LR-A | +0.054 | 10/10 | <0.001 |
| PLTR − LR-B | +0.002 | 8/10 | 0.33 |
| ALasso-B − LR-B | −0.0003 | 1/10 | 0.55 |
| PLTR full − singles-only | +0.005 | 10/10 | 0.04 (fails Bonferroni) |
| HGB-add − LR-B | −0.001 | 4/10 | 0.70 |
| HGB − HGB-add | +0.007 | 10/10 | 0.002 |
| HGB-d2 − HGB-add | +0.006 | 10/10 | 0.003 |
| HGB − PLTR | +0.004 | 10/10 | <0.001 |

### 4.3 Findings
1. **Most of PLTR's gain over naive LR is an encoding effect.** The +0.054 AUC gap shrinks to
   +0.002 (not significant) once LR gets categorical PAY and log-money.
2. **The adaptive-lasso penalty alone adds nothing.**
3. **Pair rules help a little:** +0.005 over singles-only PLTR, consistent in 10/10 folds but not
   significant after multiple-comparison correction.
4. **Interactions exist, but are small.** A flexible additive model matches LR-B, while depth-2
   boosting beats it by ~0.006. The whole non-additive headroom is under 0.01 AUC, and no model
   passes ~0.78.
5. **PLTR captures well under half of that signal.** Depth-2 boosting captures ~75%.
   Candidate causes (untested): the one-greedy-tree-per-pair extractor, lasso shrinkage of
   correlated rules, and the lighter inner search used in CV.
6. **A crude WoE scorecard is 0.004 below LR-B.** Binning discards information on skewed money columns.

### 4.4 Interpretation
The toy example in section 2.1 works because the interactions were planted in the data. On
the Taiwan data, the interaction signal is small, so there is little for PLTR to find.
PLTR is a reasonable method on data with strong interactions. This dataset is not one of them.
The paper's reported LR baseline (AUC 0.631, Brier worse than a constant guess) looks weak,
so its headline gap is probably inflated by the baseline.

### 4.5 Limits
One dataset; untuned baselines; the paper's protocol only approximated; the first test split
was reused during debugging; about 11 comparisons were run, so single p-values near 0.05
should not be over-read; solver noise is about 0.001 AUC.

### 4.6 Process lessons
- A `-- 1` typo (parsed as `+ 1`) silently produced zero pair rules. Counters such as `stats_` expose this.
- A `globals()` swap meant a "singles-only" run was identical to the full model. Pass components explicitly.
- A single split overstated PLTR's gain by about 5× (+0.011 vs +0.002