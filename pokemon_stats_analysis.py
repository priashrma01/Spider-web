# ==========================================
# Integrated Pokémon Stats Analysis Script
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, openpyxl
# Input files: pokemon_rs.xlsx, pokemon_bw.xlsx, pokemon_swsh.xlsx
# Output files: pokemon_bst_boxplot.png, pokemon_stat_corr_heatmap.png,
#               pokemon_pca_scatter.png, pokemon_cluster_scatter.png,
#               pokemon_mean_bst_lineplot.png
# ==========================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ---------- Configuration ----------
STAT_COLS = ["HP", "Att", "Def", "S.Att", "S.Def", "Spd"]
FILES = {
    "RS":   "pokemon_rs.xlsx",
    "BW":   "pokemon_bw.xlsx",
    "SWSH": "pokemon_swsh.xlsx",
}
GEN_ORDER = ["RS", "BW", "SWSH"]   # earliest → latest

# ---------- Load Datasets Robustly ----------
datasets = {}
for gen, path in FILES.items():
    df = pd.read_excel(path)
    # Keep first occurrence per Name (base form) to remove alternate forms
    df = df.drop_duplicates(subset=["Name"], keep="first").reset_index(drop=True)
    datasets[gen] = df

# ---------- Align by Pokémon Identifier (Name) ----------
common_names = (
    set(datasets["RS"]["Name"])
    & set(datasets["BW"]["Name"])
    & set(datasets["SWSH"]["Name"])
)
common_names = sorted(common_names)

aligned = {}
for gen in GEN_ORDER:
    df = datasets[gen]
    df = df[df["Name"].isin(common_names)].copy()
    df = df.set_index("Name").loc[common_names].reset_index()
    aligned[gen] = df

n_aligned = len(common_names)
print(f"Pokémon remaining after alignment: {n_aligned}")

# ---------- Compute Total Base Stats (BST) per Generation ----------
for gen in GEN_ORDER:
    aligned[gen]["TotalBST"] = aligned[gen][STAT_COLS].sum(axis=1)

# ---------- Mean Absolute Difference in BST (Earliest vs Latest) ----------
bst_earliest = aligned["RS"]["TotalBST"].values
bst_latest   = aligned["SWSH"]["TotalBST"].values
mad_bst = np.mean(np.abs(bst_latest - bst_earliest))
print(f"Mean absolute difference in total BST (RS vs SWSH): {mad_bst:.4f}")

# ---------- Standardise Base Stats (Pooled Mean & Population Std) ----------
all_stats = pd.concat(
    [aligned[g][STAT_COLS] for g in GEN_ORDER], ignore_index=True
)
pooled_mean = all_stats.mean()
pooled_std  = all_stats.std(ddof=0)          # population std

std_data = {}
for gen in GEN_ORDER:
    std_data[gen] = (aligned[gen][STAT_COLS] - pooled_mean) / pooled_std

# ---------- PCA on Standardised Stats (All Components) ----------
std_all = pd.concat([std_data[g] for g in GEN_ORDER], ignore_index=True)
pca = PCA(n_components=len(STAT_COLS))
pca.fit(std_all)
pc_scores_all = pca.transform(std_all)

var_explained_pc1 = pca.explained_variance_ratio_[0]
print(f"Variance explained by PC1: {var_explained_pc1:.4f}")

# ---------- Multiple Linear Regression (Predict SWSH TotalBST) ----------
X_reg = std_data["SWSH"].values
y_reg = aligned["SWSH"]["TotalBST"].values

reg = LinearRegression().fit(X_reg, y_reg)
y_pred = reg.predict(X_reg)
ss_res = np.sum((y_reg - y_pred) ** 2)
ss_tot = np.sum((y_reg - np.mean(y_reg)) ** 2)
r2 = 1 - ss_res / ss_tot
n_obs = len(y_reg)
p_vars = X_reg.shape[1]
adj_r2 = 1 - (1 - r2) * (n_obs - 1) / (n_obs - p_vars - 1)
print(f"Regression adjusted R²: {adj_r2:.4f}")

# ---------- K-Means Clustering (k=5) on Standardised Stats ----------
std_combined = std_all.values
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(std_combined)

# Re-assign clusters by ascending centroid total-stat magnitude
centroid_totals = kmeans.cluster_centers_.sum(axis=1)
order = np.argsort(centroid_totals)
label_map = {old: new for new, old in enumerate(order)}
final_labels = np.array([label_map[l] for l in cluster_labels])

sil_score = silhouette_score(std_combined, final_labels)
print(f"Silhouette score (k=5): {sil_score:.4f}")

# ---------- Coefficient of Variation by Primary Type ----------
# Use SWSH dataset for type; primary type = first type before comma
type_bst = aligned["SWSH"][["Name", "TotalBST"]].copy()
type_bst["PrimaryType"] = aligned["SWSH"]["Type"].str.split(",").str[0].str.strip()

type_stats = type_bst.groupby("PrimaryType")["TotalBST"].agg(
    mean="mean",
    std_pop=lambda x: np.std(x, ddof=0),
)
type_stats["cv"] = type_stats["std_pop"] / type_stats["mean"]
max_cv_type = type_stats["cv"].idxmax()
max_cv_val  = type_stats.loc[max_cv_type, "cv"]
print(f"Highest CV type: {max_cv_type} (CV = {max_cv_val:.4f})")

# ---------- Rank Pokémon by SWSH TotalBST ----------
rank_df = aligned["SWSH"][["No", "Name", "TotalBST"]].copy()
rank_df = rank_df.sort_values(
    by=["TotalBST", "Name"], ascending=[False, True]
).reset_index(drop=True)
top_pokemon_no = rank_df.iloc[0]["No"]
print(f"Highest-ranked Pokémon identifier (SWSH): {top_pokemon_no}")

# ---------- Pearson Correlation (RS vs SWSH TotalBST) ----------
pearson_r, _ = sp_stats.pearsonr(bst_earliest, bst_latest)
print(f"Pearson r (RS vs SWSH TotalBST): {pearson_r:.4f}")

# ---------- Most Consistent Pokémon Across Generations ----------
bst_matrix = np.column_stack(
    [aligned[g]["TotalBST"].values for g in GEN_ORDER]
)
bst_range = bst_matrix.max(axis=1) - bst_matrix.min(axis=1)
min_change_idx = np.argmin(bst_range)
most_consistent = common_names[min_change_idx]
print(f"Most consistent Pokémon (least BST change): {most_consistent}")

# ---------- Composite Importance Score ----------
# Component 1: absolute PC1 loading
pc1_loadings = np.abs(pca.components_[0])

# Component 2: absolute standardised regression coefficients
# Standardise coefficients: coef * std_X / std_y  (but X already standardised)
# Use beta coefficients directly since X is standardised
beta_abs = np.abs(reg.coef_)

# Component 3: normalised range of cluster centroid means per stat
# Reorder centroids to final cluster ordering
reordered_centroids = kmeans.cluster_centers_[order]
centroid_range = reordered_centroids.max(axis=0) - reordered_centroids.min(axis=0)

# Normalise each component to [0, 1]
def normalise_01(arr):
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

norm_pc1   = normalise_01(pc1_loadings)
norm_beta  = normalise_01(beta_abs)
norm_range = normalise_01(centroid_range)

composite = norm_pc1 + norm_beta + norm_range
importance_df = pd.DataFrame({
    "Stat": STAT_COLS,
    "Composite": composite,
})
importance_df = importance_df.sort_values(
    by=["Composite", "Stat"], ascending=[False, True]
)
most_important_stat = importance_df.iloc[0]["Stat"]
print(f"Most important base stat (composite): {most_important_stat}")

# ===================== VISUALISATIONS =====================

# ---------- 1. Boxplot: Total BST Distribution by Generation ----------
fig, ax = plt.subplots(figsize=(8, 5))
box_data = [aligned[g]["TotalBST"].values for g in GEN_ORDER]
bp = ax.boxplot(box_data, labels=GEN_ORDER, patch_artist=True,
                boxprops=dict(facecolor="#4C72B0", alpha=0.7),
                medianprops=dict(color="black", linewidth=2))
ax.set_xlabel("Generation")
ax.set_ylabel("Total Base Stats")
ax.set_title("Distribution of Total Base Stats by Generation")
plt.tight_layout()
plt.savefig("pokemon_bst_boxplot.png", dpi=150)
plt.close()

# ---------- 2. Correlation Heatmap of Base Stats ----------
corr = std_all.corr()
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(STAT_COLS)))
ax.set_yticks(range(len(STAT_COLS)))
ax.set_xticklabels(STAT_COLS, rotation=45, ha="right")
ax.set_yticklabels(STAT_COLS)
for i in range(len(STAT_COLS)):
    for j in range(len(STAT_COLS)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}",
                ha="center", va="center", fontsize=9,
                color="white" if abs(corr.values[i, j]) > 0.5 else "black")
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Base Stat Correlation Heatmap")
plt.tight_layout()
plt.savefig("pokemon_stat_corr_heatmap.png", dpi=150)
plt.close()

# ---------- 3. PCA Scatter (PC1 vs PC2) ----------
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(pc_scores_all[:, 0], pc_scores_all[:, 1],
           alpha=0.4, s=15, c="#4C72B0")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title("PCA – First Two Principal Components")
ax.axhline(0, color="grey", lw=0.5, ls="--")
ax.axvline(0, color="grey", lw=0.5, ls="--")
plt.tight_layout()
plt.savefig("pokemon_pca_scatter.png", dpi=150)
plt.close()

# ---------- 4. Cluster Scatter (PC1 vs PC2, coloured by cluster) ----------
fig, ax = plt.subplots(figsize=(8, 6))
palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
for cl in range(5):
    mask = final_labels == cl
    ax.scatter(pc_scores_all[mask, 0], pc_scores_all[mask, 1],
               alpha=0.45, s=15, c=palette[cl], label=f"Cluster {cl}")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title("K-Means Clusters on PCA Space")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("pokemon_cluster_scatter.png", dpi=150)
plt.close()

# ---------- 5. Line Chart: Mean BST Across Generations ----------
gen_means = [aligned[g]["TotalBST"].mean() for g in GEN_ORDER]
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(GEN_ORDER, gen_means, marker="o", linewidth=2, color="#4C72B0")
for i, v in enumerate(gen_means):
    ax.annotate(f"{v:.1f}", (GEN_ORDER[i], v),
                textcoords="offset points", xytext=(0, 10), ha="center")
ax.set_xlabel("Generation")
ax.set_ylabel("Mean Total Base Stats")
ax.set_title("Mean Total Base Stats Across Generations")
plt.tight_layout()
plt.savefig("pokemon_mean_bst_lineplot.png", dpi=150)
plt.close()

print("\nAll visualisations saved.")
