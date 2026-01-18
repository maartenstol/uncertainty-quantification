# Hackathon - Uncertainty Quantification in Predictive Maintenance

## 1. Context and Learning Objectives

### Context

This hackathon is designed to make the Deep Learning (DL) team more familiar with the challenges of **Uncertainty Quantification (UQ)**. Currently, the team works primarily in computer vision for pavement management with a focus on image classification and object detection. Besides the basic softmax confidence scores provided by the DL models, there is no real form of UQ in the product stack.

One family of new product features will concern the estimation of future pavement conditions based on observations in the present, and with very little access to longitudinal data. The challenge is two-fold. First, a reliable estimate of future asset condition is to be made. Second, a more formal approach towards quantifying the uncertainties of these estimates is desired. In other words, we want to believe our estimates, but also know how certain we really are. This is the domain of UQ.

For a single-day event, we need to simplify. While our product’s visual data is gradually being extended by relational metadata, making the product stack multi-modal, *we don’t need the visual data for our introduction to UQ*. The relational data is sufficient to demonstrate relevant notions of uncertainty and some of the standard methods and approaches to take.

Additionally, since real-world relational data on pavement asset management is often messy, we have chosen to work with a well-behaved synthetic dataset: **The NASA Turbofan dataset**. We then transform the dataset and the prediction task to make the analogy with our product goals as tight as possible. By doing this, the hackathon utilizes tabular data to isolate the statistical concepts without the engineering overhead of Bayesian CNNs or real-world data integration challenges.

### Learning Objectives

* **Distinguish Uncertainties:** Participants must distinguish between *Aleatoric Uncertainty* (inherent sensor noise or ambiguity) and *Epistemic Uncertainty* (model ignorance due to lack of data).
* **Risk-Aware Decision Making:** Move beyond minimizing MSE/Accuracy. The goal is to minimize *financial risk* by using uncertainty to inform maintenance decisions.
* **Methodological Comparison:** Contrast theoretical Bayesian purity against the advantages of more pragmatic or scalable methods, for example, Gaussian Processes versus Ensemble-based methods.

---

## 2. The Dataset: Modified NASA Turbofan Data

As a close analogy to the product’s pavement condition use case, we use the [**NASA Turbofan Engine Degradation Simulation data**](https://www.kaggle.com/datasets/bishals098/nasa-turbofan-engine-degradation-simulation).

### NASA Turbofan Engine Degradation

***From the website:***

* *The Turbofan Engine Degradation Simulation Dataset represents one of the most widely used benchmark datasets in the prognostics and health management (PHM) community.*
* *This dataset contains comprehensive prognostic data for turbofan engine degradation simulation, generated using NASA's Commercial Modular Aero-Propulsion System Simulation (C-MAPSS).*
* *The dataset is part of NASA's Prognostics Data Repository, which focuses exclusively on prognostic datasets that can be used for developing predictive maintenance algorithms.*
* *It provides run-to-failure time series data that captures the degradation patterns of aircraft engines under various operational conditions.*

### Analogies with Pavement Condition

Analogies between pavement and engine condition estimation are:

* Pavement Deterioration = Engine Degradation
* Absence of longitudinal data = Work with snapshots of engine data
* Inaccurate data records = Handling noisy sensor inputs
* Out-of-distribution predictions = Hold out set of specific operating conditions
* Task: Predict time to repair = Task: Predict Remaining Useful Life (RUL)
* UQ: Where to spend budget? = UQ: Decide ACTION (repair) or PASS (wait).

### Dataset Modifications & Scenarios

To enforce a focus on "snapshot" inspection (analogous to current pavement inspection limitations) rather than time-series monitoring, we have modified the standard benchmark.

**The "Snapshot" Transformation:**
* **No Asset IDs:** The `Unit_ID` and `Cycle` columns are removed from the training features. This prevents the model from memorizing specific asset histories ("Clever Hans" effect).
* **Shuffled Snapshots:** Rows are randomly shuffled. The model must learn $f(Sensors) \rightarrow RUL$ purely from the current asset state, not the temporal sequence.

**The Scenarios:**
We provide four distinct scenarios (FD001–FD004) that represent increasing levels of difficulty. Teams will train primarily on **Scenario 1 (FD001)** but will be evaluated against hidden test sets from the other scenarios to test generalization.

1.  **Scenario FD001 (Baseline):** Engines operating at Sea Level with a single fault mode (High Pressure Compressor degradation).
2.  **Scenario FD002 (Epistemic Challenge):** Engines operating under **6 different conditions** (Altitude/Speed). The model must generalize across regimes.
3.  **Scenario FD003 (Aleatoric Challenge):** Engines with **2 distinct fault modes** (HPC or Fan degradation). Identical sensor readings may map to different RULs depending on which invisible component is breaking, creating inherent ambiguity.
4.  **Scenario FD004 (Hard Mode):** Multiple operating conditions AND multiple fault modes.

**Data Format:**
* **Features (X):** Operational Settings (op1-op3) + 21 Sensor Readings (s1-s21).
* **Target (y):** Remaining Useful Life (RUL).

---

## 3. Team Tracks - The Showdown

The group will be split into two tracks to facilitate a debate on "Exactness vs. Expressivity". Methods and tooling are suggestions that have gone through a mild vetting process.

On our particular problem format, this split is motivated by what we hope to learn from the hackathon:

***What advantage does a Purist approach have over a Pragmatist approach, and vice versa?***

### Track A: The “Purists” or Exact Bayesians

* **Method:** Gaussian Processes (GPs), Sparse Gaussian Processes (SGPR), or Variational GPs.
* **Hypothesis:** GPs provide the "gold standard" for uncertainty but may struggle with data dimensionality or non-linearities if the kernel is not well-chosen.
* **Tooling:** [GPyTorch](https://gpytorch.ai/) - A highly efficient and modular implementation of GPs, with GPU acceleration.

### Track B: The “Pragmatists” or Approximate Bayesians

* **Method:** Deep Ensembles (training 5+ independent models) or Monte Carlo Dropout.
* **Hypothesis:** Neural Networks offer superior expressivity for complex functions, but their uncertainty estimates are often uncalibrated (overconfident).
* **Tooling:** Standard `torch.nn` modules.

---

## 4. The Afternoon Challenge - The Maintenance Portfolio Game

Decision-making requires more than accuracy! The winners are not decided merely by highest accuracy or F1-score of the prediction itself. Instead, we simulate the need to act and make decisions under uncertainty. And this needs a focus on UQ in addition to model accuracies. We’ll do this in the last hour of the day.

### The Scenario

We simulate an Asset Management scenario with “credits” and an asset “portfolio”.

* All models act as Asset Managers with a limited budget in credits.
* Each model is presented with the same portfolio of asset snapshots that require a decision.
* For each asset, the model sees the sensor snapshot and gives a probabilistic prediction.

### The Decision

The decision is analogous to that of pavement repair based on a single observation.

* Based on predicted probabilities, a decision is made for each asset.
* The decision is a binary choice between **ACT** (Repair) or **PASS** (Ignore).
* Depending on how much RUL the asset has left, a cost is incurred according to the Cost Matrix.

### The Cost Matrix

The scoring simulates the economic reality where failure is catastrophic, but early repair is wasteful.

* **T (Threshold):** The "danger zone". Engines with RUL < 30 must be repaired.
* **F (Failure Cost):** **1000 credits.** The penalty for PASSING on a failing asset.
* **R (Repair Cost):** **50 credits.** The fixed cost for every decision to ACT.
* **W (Waste Factor):** **0.5.** A penalty multiplier for repairing healthy assets too early.

### The Winning Condition

Teams start with a hypothetical budget and pay all costs incurred on the test set. The team with the highest remaining budget wins.

### Why UQ Wins

* A deterministic model predicting RUL=35 (Safe) might miss a 10% risk of failure. The expected cost of passing (0.10 * 1000 = 100) exceeds the repair cost (50), but the deterministic model will pass and eventually incur catastrophe.
* A calibrated Bayesian model will detect the tail risk and choose to repair, paying 50 to save an expected 100.

### Evaluation Strategy: The Stress Test Matrix

Since the goal is to spark debate on model robustness, we will not rely on a single leaderboard. Instead, we perform a **Cross-Scenario Evaluation** to see how models behave when specific assumptions are violated.

We evaluate models in three tiers.

#### Tier 1: The Baseline (Sanity Check)
* **Task:** Train on FD00X $\rightarrow$ Test on FD00X.
* **Goal:** Establish that the model actually learned the task. 
* **Metric:** Cost Matrix Score (Remaining Budget).

#### Tier 2: The Ambiguity Test (Aleatoric Uncertainty)
* **Task:** Train on Single-Fault (FD001) $\rightarrow$ Test on Multi-Fault (FD003).
* **The Situation:** The operating conditions are familiar (Sea Level), so the inputs $X$ look normal. However, in the test set, an engine might be failing due to a Fan issue OR a Compressor issue.
* **The Uncertainty:** The model lacks the "Fault Code" input. It faces **Irreducible Uncertainty**: identical sensor readings now map to different possible RULs.
* **The Test:** Does the model admit "I can't distinguish between these two failures" (high aleatoric uncertainty), or does it guess one and get it wrong?

#### Tier 3: The Novelty Test (Epistemic Uncertainty)
* **Task:** Train on Single-Regime (FD001) $\rightarrow$ Test on Multi-Regime (FD002).
* **The Situation:** The fault mode is familiar, but the environment is alien. The test set contains engines flying at altitudes (30,000ft) the model has never seen. The input $X$ is **Out-of-Distribution**.
* **The Uncertainty:** The model is effectively blind. It faces **Reducible Uncertainty** (it simply lacks training data here).
* **The Test:** Does the model recognize "I have no idea where I am" (high epistemic uncertainty) and PASS, or does it blindly extrapolate and ACT?

### Ranking & Debate
There will be no single winner. Instead, we will rank teams on each Tier separately. 
* *Winner of Tier 1:* Best Engineer.
* *Winner of Tier 2/3:* Best Risk Manager.

If team and player rankings differ according to how models are evaluated, winners will be decided by heated debate.

---