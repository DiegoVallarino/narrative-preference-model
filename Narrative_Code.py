# --------------------------------------------------------
# Narrative Preference Modeling – Mixture of Experts + t-SNE
# Author: Diego Vallarino
# Description: Simulation of a preference model for narrative structures,
#              combining interpretable features and latent embeddings
# --------------------------------------------------------

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

# --------------------------------------------------------
# Setup
# --------------------------------------------------------
np.random.seed(42)
torch.manual_seed(42)

# Define output directory (Desktop)
output_dir = os.path.expanduser("~/Desktop/story_outputs")
os.makedirs(output_dir, exist_ok=True)


# ----------------------------------------------------------
# Step 0: (Optional) Extract narrative features from text
# ----------------------------------------------------------

# Simulate synopses for each show (in practice, scraped from IMDb, Netflix, etc.)
synopses = {
    'Slow Horses': "A team of British intelligence agents... betrayals, secrets, and high-stakes missions.",
    'Unforgotten': "Detectives solve cold cases... long-buried truths emerge.",
    'Mobland': "A small-town man is drawn into a web of organized crime and violence.",
    'Dept Q': "A detective and his partner solve dark, unresolved cases in Denmark.",
    'Adolescence': "A troubled teenager navigates trauma, identity, and emotional chaos."
}

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler

# Load pre-trained semantic model (simulate text embeddings)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode all synopses
embeddings = model.encode(list(synopses.values()))

# For illustrative purposes, simulate narrative scores via PCA (or fixed projections)
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
narrative_scores = pca.fit_transform(embeddings)

# Normalize to [0,1]
scaler = MinMaxScaler()
scores_scaled = scaler.fit_transform(narrative_scores)

# Rebuild feature DataFrame
X = pd.DataFrame(scores_scaled, 
                 columns=['Non-Linearity', 'Early Intro', 'Intrigue', 'Surprise', 'Logic'],
                 index=synopses.keys())

# --------------------------------------------------------
# 1. Simulate Series Data
# --------------------------------------------------------
n = 200  # Number of TV series (simulated)
x = np.random.rand(n, 5)  # Structural features: N, E, I, S, L
z = np.random.randn(n, 3)  # Latent semantic embeddings

# Feature names (optional)
feature_names = ["Nonlinearity (N)", "Early Intro (E)", "Intrigue (I)", "Surprise (S)", "Logic (L)"]

# --------------------------------------------------------
# 2. Gating Network: Softmax over latent dimensions
# --------------------------------------------------------
def softmax(row):
    exp_row = np.exp(row - np.max(row))
    return exp_row / np.sum(exp_row)

gating_weights = np.apply_along_axis(softmax, 1, z)

# --------------------------------------------------------
# 3. Mixture of Experts: 3 expert models
# --------------------------------------------------------
beta_k = np.random.uniform(-1, 1, (3, 5))  # 3 experts × 5 features
expert_outputs = x @ beta_k.T
g_zi = np.sum(gating_weights * expert_outputs, axis=1)  # Weighted contribution from each expert

# --------------------------------------------------------
# 4. Linear Part and Latent Utility
# --------------------------------------------------------
theta = np.array([1.2, 0.9, 1.1, 1.5, 1.3])  # Linear weights for structural features
linear_part = x @ theta
U = linear_part + g_zi + np.random.logistic(0, 1, n)  # Total latent utility with noise

# --------------------------------------------------------
# 5. Pairwise Comparisons (Thurstone-Mosteller / Bradley-Terry)
# --------------------------------------------------------
n_pairs = 1000
i = np.random.randint(0, n, n_pairs)
j = np.random.randint(0, n, n_pairs)
mask = i != j
i = i[mask]
j = j[mask]
outcome = (U[i] > U[j]).astype(int)  # 1 if i is preferred over j

# Save delta features for logistic regression or preference learning
delta_x = x[i] - x[j]
df = pd.DataFrame(delta_x, columns=[f"delta_{f}" for f in ["N", "E", "I", "S", "L"]])
df["outcome"] = outcome
df.to_csv(os.path.join(output_dir, "pairwise_comparisons.csv"), index=False)

# --------------------------------------------------------
# 6. Visualization via t-SNE
# --------------------------------------------------------
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embedding = tsne.fit_transform(np.hstack([x, g_zi.reshape(-1, 1)]))

embedding_df = pd.DataFrame(embedding, columns=["TSNE_1", "TSNE_2"])
embedding_df["Utility"] = U

# --------------------------------------------------------
# 7. Plotting the Result
# --------------------------------------------------------
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    embedding_df["TSNE_1"],
    embedding_df["TSNE_2"],
    c=embedding_df["Utility"],
    cmap="viridis",
    s=60
)
plt.colorbar(scatter, label="Estimated Utility (U)")
plt.title("Latent Narrative Preferences – t-SNE Projection")
plt.xlabel("TSNE Dimension 1")
plt.ylabel("TSNE Dimension 2")

# Save figure to Desktop
plot_path = os.path.join(output_dir, "latent_preferences_plot.png")
plt.savefig(plot_path, dpi=300)
plt.close()


#################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Step 1: Define narrative features
# ---------------------------
narrative_dimensions = ['Non-Linearity', 'Early Intro', 'Intrigue', 'Surprise', 'Logic']

series_data = {
    'Slow Horses':     [0.75, 0.95, 0.90, 0.80, 0.70],
    'Unforgotten':     [0.60, 0.90, 0.85, 0.60, 0.85],
    'Mobland':         [0.80, 0.65, 0.75, 0.75, 0.50],
    'Dept Q':          [0.65, 0.90, 0.88, 0.70, 0.75],
    'Adolescence':     [0.50, 0.70, 0.95, 0.85, 0.90],
}

X = pd.DataFrame(series_data, index=narrative_dimensions).T

# ---------------------------
# Step 2: Simulate user preferences (latent vector w)
# ---------------------------
# Example user: likes intrigue and logic, dislikes excessive surprise or non-linearity
w = np.array([0.3, 0.7, 1.2, 0.4, 1.0])  # weights for [N, E, I, S, L]

# Normalize w (optional but helps interpretation)
w = w / np.linalg.norm(w)

# ---------------------------
# Step 3: Compute utilities and softmax probabilities
# ---------------------------
utilities = X.values @ w  # U_i = w^T x_i
exp_utilities = np.exp(utilities - np.max(utilities))  # for numerical stability
probs = exp_utilities / exp_utilities.sum()

# Create output DataFrame
output_df = pd.DataFrame({
    'Utility': utilities,
    'Probability': probs
}, index=X.index).sort_values(by='Probability', ascending=False)

print("=== Narrative Preference Model: Ranking ===")
print(output_df)

# ---------------------------
# Step 4: Visualize as bar plot
# ---------------------------
plt.figure(figsize=(10, 6))
plt.barh(output_df.index[::-1], output_df['Probability'][::-1], color='navy')
plt.xlabel("Selection Probability")
plt.title("Narrative Recommendation Based on Latent Preferences")
plt.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

##########################################
import matplotlib.pyplot as plt
import numpy as np

# Series and narrative metrics
series = ['Slow Horses', 'Unforgotten', 'Mobland', 'Dept Q', 'Adolescence']
labels = ['Non-linearity (N)', 'Early Intro (E)', 'Intrigue (I)', 'Surprise (S)', 'Logic (L)']

# Narrative scores based on analysis of reviews
scores = {
    'Slow Horses':     [0.75, 0.95, 0.90, 0.80, 0.70],  # Complex, cynical, layered
    'Unforgotten':     [0.60, 0.90, 0.85, 0.60, 0.85],  # Procedural, emotionally deep
    'Mobland':         [0.40, 0.80, 0.65, 0.75, 0.55],  # Crime drama, more stylized than tight
    'Dept Q':          [0.65, 0.90, 0.88, 0.70, 0.75],  # Nordic noir, solid plot layering
    'Adolescence':     [0.50, 0.70, 0.95, 0.85, 0.90]   # Indie-style, introspective, logical arcs
}

# Radar setup
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # Loop closure

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
plt.title("Narrative Signature of Five Shows and Films", size=16, pad=20)

# Plot each series
for show, values in scores.items():
    values += values[:1]  # Loop closure
    ax.plot(angles, values, label=show, linewidth=2)
    ax.fill(angles, values, alpha=0.1)

# Axis styling
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_ylim(0, 1)
ax.grid(True)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)

plt.show()



