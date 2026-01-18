
# Hackathon \- Uncertainty Quantification 

# in Predictive Maintenance

*20260123  \- BrainCreators, Maarten Stol*   
---


## 1\. Context and Learning Objectives {#1.-context-and-learning-objectives}

### Context {#context}

This hackathon is designed to make the Deep Learning (DL) team more familiar with the challenges of **Uncertainty Quantification (UQ)**. Currently, the team works primarily in computer vision for pavement management with a focus on image classification and object detection. Besides the basic softmax confidence scores provided by the DL models there is no real form of UQ in the product stack. 

One family of new product features will concern estimation of future pavement conditions based on observation in the present, and with very little access to longitudinal data. The challenge is two-fold. First, a reliable estimate of future asset condition is to be made. Second, a more formal approach towards quantifying the uncertainties of these estimates is desired. In other words, we want to believe our estimates, but also know how certain we really are. This is the domain of UQ. 

For a single day event, we need to simplify. While our product’s visual data is gradually being extended by relational metadata, making the product stack multi-modal, *we don’t need the visual data for our introduction to UQ*. The relational data is sufficient to demonstrate relevant notions of uncertainty and some of the standard methods and approaches to take. 

Additionally, since real-world relational data on pavement asset management is often messy, we have chosen to work with a well behaved synthetic dataset: **The NASA Turbofan dataset**. We then transform the dataset and the prediction task to make the analogy with our product goals as tight as possible. By doing this, the hackathon utilizes tabular data to isolate the statistical concepts without the engineering overhead of Bayesian CNNs or real-world data integration challenges. 

### Learning Objectives {#learning-objectives}

* **Distinguish Uncertainties:** Participants must distinguish between *Aleatoric Uncertainty* (inherent sensor noise) and *Epistemic Uncertainty* (model ignorance due to lack of data).  
* **Risk-Aware Decision Making:** Move beyond minimizing MSE/Accuracy. The goal is to minimize *financial risk* by using uncertainty to inform maintenance decisions.  
* **Methodological Comparison:** Contrast the theoretical Bayesian purity against the advantages of more pragmatic or scalable methods, as for example Gaussian Processes versus ensemble based methods. 

## 2\. The Dataset: modified NASA Turbofan data  {#2.-the-dataset:-modified-nasa-turbofan-data}

As a close analogy to the product’s pavement condition use case, we use the [**NASA Turbofan Engine Degradation Simulation data**](https://www.kaggle.com/datasets/bishals098/nasa-turbofan-engine-degradation-simulation).

### NASA Turbofan Engine Degradation  {#nasa-turbofan-engine-degradation}

***From the website:***

- *The Turbofan Engine Degradation Simulation Dataset represents one of the most widely used benchmark datasets in the prognostics and health management (PHM) community.*   
- *This dataset contains comprehensive prognostic data for turbofan engine degradation simulation, generated using NASA's Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)\[1\].* 

- *The dataset is part of NASA's Prognostics Data Repository, which focuses exclusively on prognostic datasets that can be used for developing predictive maintenance algorithms.*  
- *It provides run-to-failure time series data that captures the degradation patterns of aircraft engines under various operational conditions.*

### Analogies with pavement condition {#analogies-with-pavement-condition}

Analogies between pavement and engine condition estimation are: 

* Pavement Deterioration 		\= 	Engine Degradation  
* Absence of longitudinal data 	\= 	Work with snapshots of engine data  
* Inaccurate data records		\=	Noise augmentation of engine data   
* Out-of-distribution predictions	\=	Hold out set of specific operating conditions  
* Task: Predict time to repair		\= 	Task: Predict Remaining Useful Life (RUL)  
* UQ: Where to spend budget?	\=	UQ: Decide ACTION (repair) or PASS (wait). 

### Dataset Modifications  {#dataset-modifications}

To enforce a focus on "snapshot" inspection (analogous to current pavement inspection limitations) rather than time-series monitoring, the data will be modified and pre-processed as follows:

* **No Asset IDs:** The Unit\_ID column is removed from the training features to prevent the model from memorizing specific asset histories ("Clever Hans" effect).  
* **Shuffled Snapshots:** Rows are randomly shuffled. The model must learn   
  f(Sensors) → RUL purely from asset state, not temporal sequence or history. 

The resulting structure will be: 

1. **Format:** A cross-sectional tabular dataset (N x D).   
2. **Features (X):** Engine Age (Cycles) \+ Operational Settings \+ 21 Sensor Readings.  
3. **Target (y):** Remaining Useful Life (RUL).

### Test Set Modifications  {#test-set-modifications}

We introduce additional modifications in the test set to demonstrate the difference between **Aleatoric** and **Epistemic** uncertainties. 

* **Aleatoric Injection:** Noise is added to specific sensor columns to simulate unreliable sensor readings. This tests for robustness in the face of aleatoric uncertainties the model will inevitably encounter under deployment. 

* **Epistemic Blindspots:** Some meaningful cluster of data is withheld from the training set but present as a separate slice in the test set to penalize overconfident models.This tests the capacity for out-of-distribution generalization. 

## 3\. Team Tracks \- The Showdown {#3.-team-tracks---the-showdown}

The group will be split into two tracks to facilitate a debate on "Exactness vs. Expressivity". Methods and tooling are suggestions that have gone through a mild vetting process. 

On our particular problem format, this split is motivated by what we hope to learn from the hackathon: 

***What advantage does a Purist approach have over a Pragmatist approach, and vice versa?*** 

### Track A: The “Purists” or Exact Bayesians {#track-a:-the-“purists”-or-exact-bayesians}

* **Method:** Gaussian Processes (GPs), Sparse Gaussian Processes (SGPR) or Variational GPs.  
* **Hypothesis:** GPs provide the "gold standard" for uncertainty but may struggle with data dimensionality or non-linearities if the kernel is not well-chosen.  
* **Tooling:** [GPyTorch](https://gpytorch.ai/) A highly efficient and modular implementation of GPs, with GPU acceleration.


### Track B: The “Pragmatists” or Approximate Bayesians {#track-b:-the-“pragmatists”-or-approximate-bayesians}

* **Method:** Deep Ensembles (training 5+ independent models) or Monte Carlo Dropout.  
* **Hypothesis:** Neural Networks offer superior expressivity for complex functions, but their uncertainty estimates are often uncalibrated (overconfident).  
* **Tooling:** Standard torch.nn modules.


## 4\. The Afternoon Challenge \-  The Maintenance Portfolio Game {#4.-the-afternoon-challenge---the-maintenance-portfolio-game}

Decision making requires more than accuracy\! The winners are not decided merely by highest accuracy or F1-score of the prediction itself. Instead, we simulate the need to act and make decisions under uncertainty. And this needs a focus on UQ in addition to model accuracies. To keep things as simple as possible, though, we will simulate a decision making scenario. We’ll do this in the last hour of the day. 

### The Scenario {#the-scenario}

We simulate an Asset Management scenario with “credits” and an asset “portfolio” 

* All models act as Asset Managers with a limited budget in credits.   
* Each model is presented with the same portfolio of asset snapshots that require a decision.   
* For each asset, the model sees the sensor snapshot and gives a probabilistic prediction. 

### The Decision {#the-decision}

The decision is analogous to that of pavement repair based on a single observation. 

* Based on predicted probabilities, a decision is made for each asset.   
* The decision is a binary choice between ACT (Repair) or PASS (Ignore).  
* Depending on how much RUL the asset has left, a cost is incurred. See the cost matrix. 

### The Cost Matrix {#the-cost-matrix}

The scoring simulates the economic reality where failure is catastrophic, but early repair is wasteful. The figure indicates the costs as a confusion matrix with the following parameters. 

* T  :=	The “near future” time window between which repairs are not possible  
* F  :=	The cost of failure, a high penalty for decision to PASS on failing assets.   
* R  := 	The cost of repair, a fixed overhead for every decision to ACT  
* W :=	The waste penalty factor, to penalize false positive decisions to act on healthy assets.   
   ![][image1]

### The Winning Condition {#the-winning-condition}

Teams start with a hypothetical budget and pay all costs incurred on the test set. The team with the highest remaining budget wins.

### Why UQ Wins {#why-uq-wins}

* A deterministic model predicting RUL=25 (Safe) might miss a 10% risk of failure. The expected cost of passing ($0.10 \\times 1000 \= 100$) exceeds the repair cost ($50$), but the deterministic model will pass and eventually incur catastrophe.  
* A calibrated Bayesian model will detect the risk and choose to repair, paying 50 to save an expected 100\.

### Choice of Cost Matrix Parameters  {#choice-of-cost-matrix-parameters}

\[to be written\]

### Evaluation on the Modified Test Sets {#evaluation-on-the-modified-test-sets}

The test set comes in 3 distinct slices: 

1. Unmodified test samples: IID like the training data.   
2. Aleatoric noise test samples: on which some form of noise is introduced. The teams are not informed in advance about the type of distribution of noise.   
3. Out-of-Distribution test samples: clusters that constitute some meaningful shift in input features. The teams are not informed in advance about the meaning of these clusters. 

Evaluation is done in several ways: 

* Strictly on slice 1 only.   
* On all slices together.   
* On slice 2 separately.   
* On slice 3 separately. 

If team and player rankings differ according to how models are evaluated, winners will be decided by heated debate. 

---
