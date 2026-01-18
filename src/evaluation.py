#file: evaluation.py
#
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class MaintenanceScorer:
    def __init__(self, t=30, f=1000, r=50, w=1.25, initial_budget=100000):
        """
        Args:
            t (int): Danger threshold (cycles). RUL <= t is a Failure.
            f (float): Cost of Failure (False Negative).
            r (float): Cost of Repair (Action).
            w (float): Waste factor (penalty for early repair).
            initial_budget (float): Starting credits for the game.
        """
        self.T = t
        self.F = f
        self.R = r
        self.W = w
        self.initial_budget = initial_budget

    def calculate_costs(self, y_true_rul, y_action):
        """
        Calculates the detailed cost breakdown for a set of decisions.
        
        Args:
            y_true_rul (array): The actual RUL of the engines.
            y_action (array): 1 for ACT (Repair), 0 for PASS (Do Nothing).
            
        Returns:
            dict: Detailed stats (total_cost, final_budget, confusion_breakdown)
        """
        y_true_rul = np.array(y_true_rul)
        y_action = np.array(y_action)
        
        # 1. Determine True Class (Needs Repair?)
        # Positive (1) = Needs Repair (RUL <= T)
        # Negative (0) = Healthy (RUL > T)
        y_true_class = (y_true_rul <= self.T).astype(int)
        
        # 2. Calculate Costs per outcome
        costs = np.zeros_like(y_true_rul, dtype=float)
        
        # Scenario: True Positive (Correct Repair)
        # Decision: 1, Truth: 1. Cost = R
        mask_tp = (y_action == 1) & (y_true_class == 1)
        costs[mask_tp] = self.R
        
        # Scenario: False Negative (Failure / Missed Detection)
        # Decision: 0, Truth: 1. Cost = F
        mask_fn = (y_action == 0) & (y_true_class == 1)
        costs[mask_fn] = self.F
        
        # Scenario: True Negative (Correct Pass)
        # Decision: 0, Truth: 0. Cost = 0
        mask_tn = (y_action == 0) & (y_true_class == 0)
        costs[mask_tn] = 0
        
        # Scenario: False Positive (Early Repair / Waste)
        # Decision: 1, Truth: 0. Cost = R + W * (RUL - T)
        mask_fp = (y_action == 1) & (y_true_class == 0)
        wasted_rul = np.maximum(0, y_true_rul[mask_fp] - self.T)
        costs[mask_fp] = self.R + (self.W * wasted_rul)
        
        total_cost = np.sum(costs)
        
        return {
            "total_cost": total_cost,
            "final_budget": self.initial_budget - total_cost,
            "roi": ((self.initial_budget - total_cost) / self.initial_budget) * 100,
            "breakdown": {
                "TP_count": np.sum(mask_tp), "TP_cost": np.sum(costs[mask_tp]),
                "FN_count": np.sum(mask_fn), "FN_cost": np.sum(costs[mask_fn]),
                "TN_count": np.sum(mask_tn), "TN_cost": np.sum(costs[mask_tn]),
                "FP_count": np.sum(mask_fp), "FP_cost": np.sum(costs[mask_fp])
            },
            "y_true_class": y_true_class # For confusion matrix plotting
        }

    def plot_results(self, y_true_rul, y_action):
        """Generates a visual summary of the game results."""
        res = self.calculate_costs(y_true_rul, y_action)
        bd = res['breakdown']
        
        # Confusion Matrix Data
        cm = np.array([
            [bd['TN_count'], bd['FP_count']],
            [bd['FN_count'], bd['TP_count']]
        ])
        
        cost_cm = np.array([
            [bd['TN_cost'], bd['FP_cost']],
            [bd['FN_cost'], bd['TP_cost']]
        ])
        
        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. Count Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False,
                    xticklabels=['Pass', 'Act'], yticklabels=['Healthy', 'Failing'])
        axes[0].set_title("Decision Counts (Confusion Matrix)")
        axes[0].set_ylabel("True State")
        axes[0].set_xlabel("Model Decision")
        
        # 2. Cost Matrix
        # Custom annotations for the cost matrix
        labels = np.array([
            [f"Correct Pass\nCost: {int(cost_cm[0,0])}", f"Early Repair (Waste)\nCost: {int(cost_cm[0,1])}"],
            [f"FAILURE\nCost: {int(cost_cm[1,0])}", f"Correct Repair\nCost: {int(cost_cm[1,1])}"]
        ])
        
        sns.heatmap(cost_cm, annot=labels, fmt='', cmap='Reds', ax=axes[1], cbar=False,
                    xticklabels=['Pass', 'Act'], yticklabels=['Healthy', 'Failing'])
        axes[1].set_title(f"Financial Impact\nRemaining Budget: {int(res['final_budget'])}")
        axes[1].set_xlabel("Model Decision")
        
        plt.tight_layout()
        plt.show()
        
        return res

# --- Example Usage ---
if __name__ == "__main__":
    # Test with dummy data
    scorer = MaintenanceScorer(t=30, f=1000, r=50, w=1.25)
    
    # Simulate 100 engines
    # 85 Healthy (RUL=100), 15 Failing (RUL=10)
    rul = np.concatenate([np.full(85, 100), np.full(15, 10)])
    
    # 1. Lazy Strategy (Pass All)
    actions_lazy = np.zeros(100)
    print("\n--- Lazy Strategy ---")
    res = scorer.calculate_costs(rul, actions_lazy)
    print(f"Cost: {res['total_cost']}")
    
    # 2. Paranoid Strategy (Act All)
    actions_paranoid = np.ones(100)
    print("\n--- Paranoid Strategy ---")
    res = scorer.calculate_costs(rul, actions_paranoid)
    print(f"Cost: {res['total_cost']}")
    
    # 3. Random Strategy
    actions_rnd = np.random.randint(0, 2, 100)
    print("\n--- Random Strategy (Plot) ---")
    scorer.plot_results(rul, actions_rnd)