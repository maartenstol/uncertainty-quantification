#file: evaluation.py
#
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class MaintenanceScorer:
    def __init__(self, t=30, f=1000, r=50, w=1.25, initial_budget=100000):
        """
        The Official Hackathon Scorer.
        
        Parameters:
        -----------
        t : int (default=30)
            The "Danger Threshold". Engines with RUL <= t are considered failing.
        f : float (default=1000)
            Failure Cost. Penalty for missing a failing engine (False Negative).
        r : float (default=50)
            Repair Cost. Fixed cost for any maintenance action.
        w : float (default=1.25)
            Waste Factor. Penalty multiplier for repairing healthy engines too early.
            Cost = R + W * (Remaining_RUL - T)
        initial_budget : float (default=100000)
            Starting credits for the team.
        """
        self.T = t
        self.F = f
        self.R = r
        self.W = w
        self.initial_budget = initial_budget

    def calculate_costs(self, y_true_rul, y_action):
        """
        Calculates the financial outcome of a set of maintenance decisions.
        
        Args:
            y_true_rul (array-like): True Remaining Useful Life (RUL) values.
            y_action (array-like): Binary decisions (1 = Repair, 0 = Do Nothing).
            
        Returns:
            dict: Detailed performance metrics including total cost and ROI.
        """
        y_true_rul = np.array(y_true_rul)
        y_action = np.array(y_action)
        
        # 1. Determine True Class (1 = Needs Repair, 0 = Healthy)
        y_true_class = (y_true_rul <= self.T).astype(int)
        
        # 2. Calculate Costs per outcome
        costs = np.zeros_like(y_true_rul, dtype=float)
        
        # True Positive (Correct Repair): Just the fixed repair cost
        mask_tp = (y_action == 1) & (y_true_class == 1)
        costs[mask_tp] = self.R
        
        # False Negative (Failure): The heavy failure penalty
        mask_fn = (y_action == 0) & (y_true_class == 1)
        costs[mask_fn] = self.F
        
        # True Negative (Correct Pass): Zero cost
        mask_tn = (y_action == 0) & (y_true_class == 0)
        costs[mask_tn] = 0
        
        # False Positive (Early Repair/Waste): Repair cost + Waste penalty
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
            "y_true_class": y_true_class
        }

    def plot_results(self, y_true_rul, y_action):
        """Generates the Confusion Matrix and Cost Matrix plots."""
        res = self.calculate_costs(y_true_rul, y_action)
        bd = res['breakdown']
        
        # Prepare Matrix Data (Row=Truth, Col=Pred)
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
        
        # 1. Standard Confusion Matrix (Counts)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False,
                    xticklabels=['Pass', 'Act'], yticklabels=['Healthy', 'Failing'])
        axes[0].set_title("Decision Counts")
        axes[0].set_ylabel("True State")
        axes[0].set_xlabel("Model Decision")
        
        # 2. Cost Matrix (Financials)
        # Custom labels with costs
        labels = np.array([
            [f"Correct Pass\n$0", f"Early Repair\n${int(cost_cm[0,1]):,}"],
            [f"FAILURE\n${int(cost_cm[1,0]):,}", f"Correct Repair\n${int(cost_cm[1,1]):,}"]
        ])
        
        sns.heatmap(cost_cm, annot=labels, fmt='', cmap='Reds', ax=axes[1], cbar=False,
                    xticklabels=['Pass', 'Act'], yticklabels=['Healthy', 'Failing'])
        axes[1].set_title(f"Financial Impact\nRemaining Budget: ${int(res['final_budget']):,}")
        axes[1].set_xlabel("Model Decision")
        
        plt.tight_layout()
        plt.show()
        
        return res

if __name__ == "__main__":
    # Self-test when running this file directly
    print("Running MaintenanceScorer Test...")
    scorer = MaintenanceScorer()
    
    # Fake data: 10 engines. 2 are failing (RUL=10), 8 are healthy (RUL=100)
    fake_rul = np.array([10, 10, 100, 100, 100, 100, 100, 100, 100, 100])
    
    # 1. Perfect Strategy
    print("\n--- Test 1: Perfect Strategy ---")
    perfect_actions = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0] 
    scorer.plot_results(fake_rul, perfect_actions)
    
    # 2. Lazy Strategy (Misses the failures)
    print("\n--- Test 2: Lazy Strategy ---")
    lazy_actions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    res = scorer.calculate_costs(fake_rul, lazy_actions)
    print(f"Lazy Cost: ${res['total_cost']} (Expected: 2000)")