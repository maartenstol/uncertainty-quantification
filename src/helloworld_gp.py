# file: helloworld_gp.py
#
import torch
import gpytorch
import math

# Configuration
DATA_PATH = "data/processed_tensors.pt"
TRAIN_SIZE = 1000  # Subsample for Exact GP 'Hello World'

# 1. Load Data
print(f"Loading data from {DATA_PATH}...")
data = torch.load(DATA_PATH)
X_train_full = data["X_train"]
y_train_full = data["y_train"]
X_test = data["X_test"]
y_test = data["y_test"]

# Subsample for Hello World (Exact GP is too slow for 16k on CPU)
X_train = X_train_full[:TRAIN_SIZE]
y_train = y_train_full[:TRAIN_SIZE]

print(f"Subsampled Training Data: {X_train.shape}")

# 2. Define the GP Model
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # RBF Kernel for non-linear relationships
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Initialize Likelihood and Model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(X_train, y_train, likelihood)

# 3. Training Loop
print("Starting Training...")
model.train()
likelihood.train()

# Use the Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# "Loss" for GPs - the Marginal Log Likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter = 50
for i in range(training_iter):
    optimizer.zero_grad()
    output = model(X_train)
    loss = -mll(output, y_train)
    loss.backward()
    optimizer.step()
    
    if (i + 1) % 10 == 0:
        print(f"Iter {i+1}/{training_iter} - Loss: {loss.item():.3f}   noise: {model.likelihood.noise.item():.3f}")

# 4. Inference (Hello World Signal)
print("Starting Inference on Test Set...")
model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # Predict on first 100 test points
    test_x_subset = X_test[:100]
    test_y_subset = y_test[:100]
    
    observed_pred = likelihood(model(test_x_subset))
    
    # Get mean and uncertainty
    mean = observed_pred.mean
    lower, upper = observed_pred.confidence_region()
    
    # Calculate simple error (RMSE)
    rmse = torch.sqrt(torch.mean((mean - test_y_subset) ** 2))
    print(f"\n[Signal Received]")
    print(f"Test RMSE (Subset): {rmse.item():.3f}")
    print(f"Sample Prediction 0: True={test_y_subset[0]:.2f}, Pred={mean[0]:.2f}")
    print(f"Sample Uncertainty 0: Range [{lower[0]:.2f}, {upper[0]:.2f}]")