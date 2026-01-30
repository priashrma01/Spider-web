# ==========================================
# Integrated Athletics Performance Analysis Script
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, openpyxl
# Input files: Discus Throw Men.xlsx, Shot Put Men.xlsx, Javelin Throw Men.xlsx
# Output files: violin_standardized.png, ecdf_upper_tail.png,
#               rolling_trajectories.png, efficiency_scatter.png,
#               lorenz_curves.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# ---------- Configuration ----------
FILES = {
    'Discus': 'Discus Throw Men.xlsx',
    'Shot Put': 'Shot Put Men.xlsx',
    'Javelin': 'Javelin Throw Men.xlsx'
}
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------- Load Excel Files Robustly ----------
def load_datasets():
    """Load all three discipline datasets."""
    datasets = {}
    for discipline, filename in FILES.items():
        df = pd.read_excel(filename)
        datasets[discipline] = df
    return datasets

# ---------- Helper Functions ----------
def population_std(arr):
    """Compute population standard deviation (ddof=0)."""
    return np.std(arr, ddof=0)

def population_var(arr):
    """Compute population variance (ddof=0)."""
    return np.var(arr, ddof=0)

def coefficient_of_variation(arr):
    """Compute CV as (pop_std / mean) * 100."""
    mean_val = np.mean(arr)
    std_val = population_std(arr)
    if mean_val == 0:
        return np.nan
    return (std_val / mean_val) * 100

def silverman_bandwidth(data):
    """Compute Silverman's rule bandwidth."""
    n = len(data)
    std = population_std(data)
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    A = min(std, iqr / 1.34)
    return 0.9 * A * (n ** (-1/5))

def trimmed_mean_20(arr):
    """Compute 20% symmetrically trimmed mean."""
    return stats.trim_mean(arr, 0.2)

def gini_coefficient(arr):
    """Compute Gini coefficient."""
    arr = np.array(arr)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan
    arr = np.sort(arr)
    n = len(arr)
    cumulative = np.cumsum(arr)
    return (2 * np.sum((np.arange(1, n + 1) * arr)) - (n + 1) * np.sum(arr)) / (n * np.sum(arr))

def z_score_normalize(arr):
    """Standardize using z-score normalization."""
    arr = np.array(arr)
    mean_val = np.mean(arr)
    std_val = population_std(arr)
    if std_val == 0:
        return arr - mean_val
    return (arr - mean_val) / std_val

# ---------- Task 1: Standardized Distributions and IQR ----------
def task1_standardized_distributions(datasets):
    """
    Standardize throw distances within each dataset.
    Compute Gaussian KDE with Silverman bandwidth.
    Report minimum IQR across disciplines.
    """
    print("\n" + "="*60)
    print("TASK 1: Standardized Distributions and IQR Analysis")
    print("="*60)

    standardized_data = {}
    iqr_values = {}

    for discipline, df in datasets.items():
        # Extract distance column (assume column name contains 'distance' or similar)
        dist_col = None
        for col in df.columns:
            if 'distance' in col.lower() or 'result' in col.lower() or 'mark' in col.lower():
                dist_col = col
                break
        if dist_col is None:
            dist_col = df.select_dtypes(include=[np.number]).columns[-1]

        distances = df[dist_col].dropna().values

        # Standardize using population std
        mean_dist = np.mean(distances)
        std_dist = population_std(distances)
        standardized = (distances - mean_dist) / std_dist

        standardized_data[discipline] = standardized

        # Compute IQR of standardized distribution
        q75, q25 = np.percentile(standardized, [75, 25])
        iqr = q75 - q25
        iqr_values[discipline] = iqr

        # Compute KDE with Silverman bandwidth
        bw = silverman_bandwidth(standardized)
        kde = gaussian_kde(standardized, bw_method=bw / standardized.std(ddof=0))

        print(f"{discipline}: IQR = {iqr:.6f}, Silverman BW = {bw:.6f}")

    min_iqr = min(iqr_values.values())
    print(f"\nMinimum IQR across disciplines: {min_iqr:.3f}")

    # Generate violin plot
    fig, ax = plt.subplots(figsize=(10, 6))
    data_for_violin = [standardized_data[d] for d in FILES.keys()]
    parts = ax.violinplot(data_for_violin, positions=range(len(FILES)), showmeans=True, showmedians=True)
    ax.set_xticks(range(len(FILES)))
    ax.set_xticklabels(list(FILES.keys()))
    ax.set_xlabel('Discipline')
    ax.set_ylabel('Standardized Distance')
    ax.set_title('Standardized Distance Distributions by Discipline')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('violin_standardized.png', dpi=150)
    plt.close()
    print("Generated: violin_standardized.png")

    return standardized_data, min_iqr

# ---------- Task 2: Kolmogorov-Smirnov Distance Analysis ----------
def task2_ks_distance(datasets):
    """
    Exclude lowest 10% of throws within each dataset.
    Construct ECDFs and compute KS distances between pairs.
    Report minimum KS distance.
    """
    print("\n" + "="*60)
    print("TASK 2: Kolmogorov-Smirnov Distance Analysis")
    print("="*60)

    upper_tail_data = {}

    for discipline, df in datasets.items():
        # Extract distance column
        dist_col = None
        for col in df.columns:
            if 'distance' in col.lower() or 'result' in col.lower() or 'mark' in col.lower():
                dist_col = col
                break
        if dist_col is None:
            dist_col = df.select_dtypes(include=[np.number]).columns[-1]

        distances = df[dist_col].dropna().values

        # Exclude lowest 10%
        threshold = np.percentile(distances, 10)
        upper_distances = distances[distances > threshold]
        upper_tail_data[discipline] = upper_distances
        print(f"{discipline}: {len(distances)} throws -> {len(upper_distances)} after exclusion")

    # Compute KS distances for all pairs
    disciplines = list(upper_tail_data.keys())
    ks_distances = {}

    for i in range(len(disciplines)):
        for j in range(i + 1, len(disciplines)):
            d1, d2 = disciplines[i], disciplines[j]
            ks_stat, _ = stats.ks_2samp(upper_tail_data[d1], upper_tail_data[d2])
            pair = f"{d1} vs {d2}"
            ks_distances[pair] = ks_stat
            print(f"KS distance ({pair}): {ks_stat:.6f}")

    min_ks = min(ks_distances.values())
    print(f"\nMinimum KS distance: {min_ks:.3f}")

    # Generate ECDF plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['blue', 'red', 'green']

    for idx, (discipline, data) in enumerate(upper_tail_data.items()):
        sorted_data = np.sort(data)
        ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax.step(sorted_data, ecdf, where='post', label=discipline, color=colors[idx], linewidth=1.5)

    ax.set_xlabel('Throw Distance (m)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Empirical CDF - Upper Tail Comparison (Top 90%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ecdf_upper_tail.png', dpi=150)
    plt.close()
    print("Generated: ecdf_upper_tail.png")

    return min_ks

# ---------- Task 3: Brown-Forsythe Test on CV ----------
def task3_brown_forsythe(datasets):
    """
    Compute athlete-level CV within each dataset.
    Compare variability distributions using Brown-Forsythe test.
    Report F-statistic.
    """
    print("\n" + "="*60)
    print("TASK 3: Brown-Forsythe Test on Athlete CV")
    print("="*60)

    cv_by_discipline = {}

    for discipline, df in datasets.items():
        # Identify athlete and distance columns
        athlete_col = None
        dist_col = None

        for col in df.columns:
            col_lower = col.lower()
            if 'athlete' in col_lower or 'name' in col_lower or 'competitor' in col_lower:
                athlete_col = col
            if 'distance' in col_lower or 'result' in col_lower or 'mark' in col_lower:
                dist_col = col

        if athlete_col is None:
            athlete_col = df.columns[0]
        if dist_col is None:
            dist_col = df.select_dtypes(include=[np.number]).columns[-1]

        # Remove rows with missing values in referenced variables
        df_clean = df[[athlete_col, dist_col]].dropna()

        # Compute CV for each athlete
        athlete_cvs = []
        for athlete, group in df_clean.groupby(athlete_col):
            distances = group[dist_col].values
            if len(distances) >= 2:
                cv = coefficient_of_variation(distances)
                if not np.isnan(cv):
                    athlete_cvs.append(cv)

        cv_by_discipline[discipline] = np.array(athlete_cvs)
        print(f"{discipline}: {len(athlete_cvs)} athletes with valid CV")

    # Brown-Forsythe test (Levene with median)
    cv_groups = list(cv_by_discipline.values())
    f_stat, p_val = stats.levene(*cv_groups, center='median')

    print(f"\nBrown-Forsythe F-statistic: {f_stat:.3f}")
    print(f"P-value: {p_val:.6f}")

    return f_stat

# ---------- Task 4: Rolling Trimmed Means ----------
def task4_rolling_trimmed_means(datasets):
    """
    Compute rolling trimmed means with window=3, trim=20%.
    Aggregate using median at each window index.
    Report maximum terminal rolling median.
    """
    print("\n" + "="*60)
    print("TASK 4: Rolling Trimmed Mean Analysis")
    print("="*60)

    rolling_medians = {}

    for discipline, df in datasets.items():
        # Identify columns
        athlete_col = None
        dist_col = None
        attempt_col = None

        for col in df.columns:
            col_lower = col.lower()
            if 'athlete' in col_lower or 'name' in col_lower or 'competitor' in col_lower:
                athlete_col = col
            if 'distance' in col_lower or 'result' in col_lower or 'mark' in col_lower:
                dist_col = col
            if 'attempt' in col_lower or 'round' in col_lower or 'trial' in col_lower:
                attempt_col = col

        if athlete_col is None:
            athlete_col = df.columns[0]
        if dist_col is None:
            dist_col = df.select_dtypes(include=[np.number]).columns[-1]

        # If no explicit attempt column, use row order within athlete
        if attempt_col is None:
            df = df.copy()
            df['_attempt_idx'] = df.groupby(athlete_col).cumcount() + 1
            attempt_col = '_attempt_idx'

        # Remove rows with missing values in referenced variables
        cols_to_check = [athlete_col, dist_col, attempt_col]
        df_clean = df[cols_to_check].dropna()

        # Compute rolling trimmed means for each athlete
        window_size = 3
        all_rolling_values = {}

        for athlete, group in df_clean.groupby(athlete_col):
            group_sorted = group.sort_values(attempt_col)
            distances = group_sorted[dist_col].values

            if len(distances) >= window_size:
                rolling_vals = []
                for i in range(len(distances) - window_size + 1):
                    window = distances[i:i + window_size]
                    trimmed_mean = trimmed_mean_20(window)
                    rolling_vals.append(trimmed_mean)

                for idx, val in enumerate(rolling_vals):
                    window_idx = idx + 1
                    if window_idx not in all_rolling_values:
                        all_rolling_values[window_idx] = []
                    all_rolling_values[window_idx].append(val)

        # Compute median at each window index
        median_by_idx = {}
        for window_idx, values in sorted(all_rolling_values.items()):
            median_by_idx[window_idx] = np.median(values)

        rolling_medians[discipline] = median_by_idx

        if median_by_idx:
            terminal_idx = max(median_by_idx.keys())
            terminal_val = median_by_idx[terminal_idx]
            print(f"{discipline}: Terminal window index={terminal_idx}, Median={terminal_val:.4f}")

    # Find maximum terminal rolling median
    terminal_values = []
    for discipline, medians in rolling_medians.items():
        if medians:
            terminal_val = medians[max(medians.keys())]
            terminal_values.append(terminal_val)

    max_terminal = max(terminal_values)
    print(f"\nMaximum terminal rolling median: {max_terminal:.3f}")

    # Generate line plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['blue', 'red', 'green']

    for idx, (discipline, medians) in enumerate(rolling_medians.items()):
        if medians:
            x = list(medians.keys())
            y = list(medians.values())
            ax.plot(x, y, marker='o', label=discipline, color=colors[idx], linewidth=2)

    ax.set_xlabel('Window Index (Attempt)')
    ax.set_ylabel('Rolling Trimmed Mean Distance (m)')
    ax.set_title('Rolling Performance Trajectories by Discipline')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('rolling_trajectories.png', dpi=150)
    plt.close()
    print("Generated: rolling_trajectories.png")

    return max_terminal

# ---------- Task 5: Convex Hull Analysis ----------
def task5_convex_hull(datasets):
    """
    Compute athlete-level mean and std in mean-std space.
    Construct convex hull and compute interior proportion.
    Report minimum interior proportion.
    """
    print("\n" + "="*60)
    print("TASK 5: Convex Hull Efficiency Analysis")
    print("="*60)

    hull_data = {}
    interior_proportions = {}

    for discipline, df in datasets.items():
        # Identify columns
        athlete_col = None
        dist_col = None

        for col in df.columns:
            col_lower = col.lower()
            if 'athlete' in col_lower or 'name' in col_lower or 'competitor' in col_lower:
                athlete_col = col
            if 'distance' in col_lower or 'result' in col_lower or 'mark' in col_lower:
                dist_col = col

        if athlete_col is None:
            athlete_col = df.columns[0]
        if dist_col is None:
            dist_col = df.select_dtypes(include=[np.number]).columns[-1]

        # Remove rows with missing values
        df_clean = df[[athlete_col, dist_col]].dropna()

        # Compute athlete-level statistics
        athlete_stats = []
        for athlete, group in df_clean.groupby(athlete_col):
            distances = group[dist_col].values
            mean_dist = np.mean(distances)
            std_dist = population_std(distances)
            athlete_stats.append({'athlete': athlete, 'mean': mean_dist, 'std': std_dist})

        stats_df = pd.DataFrame(athlete_stats)
        points = stats_df[['mean', 'std']].values

        # Construct convex hull
        if len(points) >= 3:
            hull = ConvexHull(points)
            hull_vertices = set(hull.vertices)

            # Count interior points (excluding vertices)
            n_total = len(points)
            n_vertices = len(hull_vertices)
            n_interior = n_total - n_vertices
            interior_prop = n_interior / n_total

            hull_data[discipline] = {
                'points': points,
                'hull': hull,
                'stats_df': stats_df
            }
            interior_proportions[discipline] = interior_prop

            print(f"{discipline}: {n_total} athletes, {n_vertices} hull vertices, Interior proportion = {interior_prop:.6f}")

    min_interior = min(interior_proportions.values())
    print(f"\nMinimum interior proportion: {min_interior:.3f}")

    # Generate scatter plot with hull overlays
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['blue', 'red', 'green']

    for idx, (discipline, data) in enumerate(hull_data.items()):
        ax = axes[idx]
        points = data['points']
        hull = data['hull']

        ax.scatter(points[:, 0], points[:, 1], alpha=0.6, color=colors[idx], s=30)

        # Plot hull
        for simplex in hull.simplices:
            ax.plot(points[simplex, 0], points[simplex, 1], 'k-', linewidth=1.5)

        # Close hull
        hull_pts = points[hull.vertices]
        hull_pts = np.vstack([hull_pts, hull_pts[0]])
        ax.plot(hull_pts[:, 0], hull_pts[:, 1], 'k-', linewidth=1.5)

        ax.set_xlabel('Mean Distance (m)')
        ax.set_ylabel('Standard Deviation (m)')
        ax.set_title(f'{discipline}')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Efficiency Comparison: Mean vs Standard Deviation with Convex Hull', y=1.02)
    plt.tight_layout()
    plt.savefig('efficiency_scatter.png', dpi=150)
    plt.close()
    print("Generated: efficiency_scatter.png")

    return min_interior, hull_data

# ---------- Task 6: Gini Coefficient and Dagum Decomposition ----------
def task6_gini_dagum(datasets):
    """
    Compute athlete-level total distance.
    Calculate Gini and decompose using Dagum method.
    Report between-discipline inequality percentage.
    """
    print("\n" + "="*60)
    print("TASK 6: Gini Coefficient and Dagum Decomposition")
    print("="*60)

    athlete_totals = {}

    for discipline, df in datasets.items():
        # Identify columns
        athlete_col = None
        dist_col = None

        for col in df.columns:
            col_lower = col.lower()
            if 'athlete' in col_lower or 'name' in col_lower or 'competitor' in col_lower:
                athlete_col = col
            if 'distance' in col_lower or 'result' in col_lower or 'mark' in col_lower:
                dist_col = col

        if athlete_col is None:
            athlete_col = df.columns[0]
        if dist_col is None:
            dist_col = df.select_dtypes(include=[np.number]).columns[-1]

        # Remove rows with missing values
        df_clean = df[[athlete_col, dist_col]].dropna()

        # Compute total distance per athlete
        totals = df_clean.groupby(athlete_col)[dist_col].sum().values
        athlete_totals[discipline] = totals

        gini = gini_coefficient(totals)
        print(f"{discipline}: {len(totals)} athletes, Gini = {gini:.6f}")

    # Dagum decomposition
    all_totals = []
    group_labels = []

    for discipline, totals in athlete_totals.items():
        all_totals.extend(totals)
        group_labels.extend([discipline] * len(totals))

    all_totals = np.array(all_totals)
    group_labels = np.array(group_labels)

    # Overall Gini
    overall_gini = gini_coefficient(all_totals)
    print(f"\nOverall Gini: {overall_gini:.6f}")

    # Dagum decomposition components
    disciplines = list(athlete_totals.keys())
    n = len(all_totals)
    total_sum = np.sum(all_totals)
    mean_overall = np.mean(all_totals)

    # Within-group component
    G_w = 0
    for discipline in disciplines:
        totals_g = athlete_totals[discipline]
        n_g = len(totals_g)
        mean_g = np.mean(totals_g)
        gini_g = gini_coefficient(totals_g)
        p_g = n_g / n
        s_g = (n_g * mean_g) / total_sum
        G_w += gini_g * p_g * s_g

    # Between-group component (gross)
    G_gb = 0
    for i, d1 in enumerate(disciplines):
        for j, d2 in enumerate(disciplines):
            if i != j:
                totals_i = athlete_totals[d1]
                totals_j = athlete_totals[d2]
                n_i, n_j = len(totals_i), len(totals_j)
                mean_i, mean_j = np.mean(totals_i), np.mean(totals_j)
                p_i, p_j = n_i / n, n_j / n
                s_i, s_j = (n_i * mean_i) / total_sum, (n_j * mean_j) / total_sum

                # Mean absolute difference
                diff_sum = 0
                for x in totals_i:
                    for y in totals_j:
                        diff_sum += abs(x - y)
                d_ij = diff_sum / (n_i * n_j * (mean_i + mean_j))

                G_gb += d_ij * p_i * s_j

    G_gb = G_gb / 2

    # Net between component (Dagum)
    G_nb = 0
    for i, d1 in enumerate(disciplines):
        for j, d2 in enumerate(disciplines):
            if i < j:
                totals_i = athlete_totals[d1]
                totals_j = athlete_totals[d2]
                n_i, n_j = len(totals_i), len(totals_j)
                mean_i, mean_j = np.mean(totals_i), np.mean(totals_j)
                p_i, p_j = n_i / n, n_j / n
                s_i, s_j = (n_i * mean_i) / total_sum, (n_j * mean_j) / total_sum

                # First moment of transvariation
                d_ij = abs(mean_i - mean_j) / (mean_i + mean_j)
                G_nb += d_ij * (p_i * s_j + p_j * s_i)

    # Between contribution percentage
    between_pct = (G_nb / overall_gini) * 100 if overall_gini > 0 else 0

    print(f"Within-group component: {G_w:.6f}")
    print(f"Between-group contribution: {between_pct:.2f}%")

    # Generate Lorenz curves
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'black']

    # Individual discipline curves
    for idx, (discipline, totals) in enumerate(athlete_totals.items()):
        sorted_totals = np.sort(totals)
        cumulative_share = np.cumsum(sorted_totals) / np.sum(sorted_totals)
        population_share = np.arange(1, len(sorted_totals) + 1) / len(sorted_totals)

        cumulative_share = np.insert(cumulative_share, 0, 0)
        population_share = np.insert(population_share, 0, 0)

        ax.plot(population_share, cumulative_share, label=discipline, color=colors[idx], linewidth=2)

    # Overall curve
    sorted_all = np.sort(all_totals)
    cumulative_share = np.cumsum(sorted_all) / np.sum(sorted_all)
    population_share = np.arange(1, len(sorted_all) + 1) / len(sorted_all)
    cumulative_share = np.insert(cumulative_share, 0, 0)
    population_share = np.insert(population_share, 0, 0)
    ax.plot(population_share, cumulative_share, label='Overall', color='black', linewidth=2, linestyle='--')

    # Equality line
    ax.plot([0, 1], [0, 1], 'k:', linewidth=1, label='Perfect Equality')

    ax.set_xlabel('Cumulative Athlete Share')
    ax.set_ylabel('Cumulative Performance Share')
    ax.set_title('Lorenz Curves for Inequality Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lorenz_curves.png', dpi=150)
    plt.close()
    print("Generated: lorenz_curves.png")

    return between_pct, athlete_totals

# ---------- Task 7: PCA Analysis ----------
def task7_pca(datasets):
    """
    Perform PCA on athlete-level standardized mean distances.
    Report maximum variance explained by PC1.
    """
    print("\n" + "="*60)
    print("TASK 7: Principal Component Analysis")
    print("="*60)

    variance_explained = {}

    for discipline, df in datasets.items():
        # Identify columns
        athlete_col = None
        dist_col = None

        for col in df.columns:
            col_lower = col.lower()
            if 'athlete' in col_lower or 'name' in col_lower or 'competitor' in col_lower:
                athlete_col = col
            if 'distance' in col_lower or 'result' in col_lower or 'mark' in col_lower:
                dist_col = col

        if athlete_col is None:
            athlete_col = df.columns[0]
        if dist_col is None:
            dist_col = df.select_dtypes(include=[np.number]).columns[-1]

        # Remove rows with missing values
        df_clean = df[[athlete_col, dist_col]].dropna()

        # Compute athlete-level means
        athlete_means = df_clean.groupby(athlete_col)[dist_col].mean().values

        # Standardize within discipline
        standardized = z_score_normalize(athlete_means)

        # For single variable, variance explained by PC1 = 1.0
        # But we need at least 2D for proper PCA
        # Use mean and variance as two features
        athlete_stats = []
        for athlete, group in df_clean.groupby(athlete_col):
            distances = group[dist_col].values
            athlete_stats.append([np.mean(distances), population_std(distances)])

        data_matrix = np.array(athlete_stats)

        if len(data_matrix) >= 2 and data_matrix.shape[1] >= 2:
            # Standardize
            data_standardized = (data_matrix - data_matrix.mean(axis=0)) / data_matrix.std(axis=0, ddof=0)

            # Covariance-based PCA
            cov_matrix = np.cov(data_standardized.T, ddof=0)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

            # Sort by descending eigenvalue
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]

            # Variance explained by PC1
            total_var = np.sum(eigenvalues)
            pc1_var = eigenvalues[0] / total_var if total_var > 0 else 0

            variance_explained[discipline] = pc1_var
            print(f"{discipline}: PC1 variance explained = {pc1_var:.6f}")

    max_var_explained = max(variance_explained.values())
    print(f"\nMaximum PC1 variance explained: {max_var_explained:.3f}")

    return max_var_explained

# ---------- Task 8: Permutation Test ----------
def task8_permutation_test(datasets):
    """
    Compare standardized athlete mean distances using permutation test.
    Test statistic: absolute difference between discipline medians.
    Report p-value.
    """
    print("\n" + "="*60)
    print("TASK 8: Permutation Test")
    print("="*60)

    standardized_means = {}

    for discipline, df in datasets.items():
        # Identify columns
        athlete_col = None
        dist_col = None

        for col in df.columns:
            col_lower = col.lower()
            if 'athlete' in col_lower or 'name' in col_lower or 'competitor' in col_lower:
                athlete_col = col
            if 'distance' in col_lower or 'result' in col_lower or 'mark' in col_lower:
                dist_col = col

        if athlete_col is None:
            athlete_col = df.columns[0]
        if dist_col is None:
            dist_col = df.select_dtypes(include=[np.number]).columns[-1]

        # Remove rows with missing values
        df_clean = df[[athlete_col, dist_col]].dropna()

        # Compute athlete-level means
        athlete_means = df_clean.groupby(athlete_col)[dist_col].mean().values

        # Standardize within discipline
        standardized = z_score_normalize(athlete_means)
        standardized_means[discipline] = standardized
        print(f"{discipline}: {len(standardized)} athletes, Median = {np.median(standardized):.6f}")

    # Pairwise permutation tests
    disciplines = list(standardized_means.keys())
    min_p_value = 1.0

    for i in range(len(disciplines)):
        for j in range(i + 1, len(disciplines)):
            d1, d2 = disciplines[i], disciplines[j]
            data1, data2 = standardized_means[d1], standardized_means[d2]

            # Observed statistic
            observed_stat = abs(np.median(data1) - np.median(data2))

            # Combined data
            combined = np.concatenate([data1, data2])
            n1 = len(data1)

            # Permutation test
            np.random.seed(RANDOM_SEED)
            n_permutations = 10000
            count_extreme = 0

            for _ in range(n_permutations):
                np.random.shuffle(combined)
                perm_stat = abs(np.median(combined[:n1]) - np.median(combined[n1:]))
                if perm_stat >= observed_stat:
                    count_extreme += 1

            p_value = count_extreme / n_permutations
            print(f"{d1} vs {d2}: Observed stat = {observed_stat:.6f}, P-value = {p_value:.4f}")

            if p_value < min_p_value:
                min_p_value = p_value

    print(f"\nPermutation p-value: {min_p_value:.4f}")

    return min_p_value

# ---------- Task 9: Stability Scores ----------
def task9_stability_scores(datasets):
    """
    Compute stability as reciprocal of median CV.
    Standardize using z-score across disciplines.
    Report maximum standardized stability score.
    """
    print("\n" + "="*60)
    print("TASK 9: Stability Score Analysis")
    print("="*60)

    stability_scores = {}

    for discipline, df in datasets.items():
        # Identify columns
        athlete_col = None
        dist_col = None

        for col in df.columns:
            col_lower = col.lower()
            if 'athlete' in col_lower or 'name' in col_lower or 'competitor' in col_lower:
                athlete_col = col
            if 'distance' in col_lower or 'result' in col_lower or 'mark' in col_lower:
                dist_col = col

        if athlete_col is None:
            athlete_col = df.columns[0]
        if dist_col is None:
            dist_col = df.select_dtypes(include=[np.number]).columns[-1]

        # Remove rows with missing values
        df_clean = df[[athlete_col, dist_col]].dropna()

        # Compute CV for each athlete
        athlete_cvs = []
        for athlete, group in df_clean.groupby(athlete_col):
            distances = group[dist_col].values
            if len(distances) >= 2:
                cv = coefficient_of_variation(distances)
                if not np.isnan(cv) and cv > 0:
                    athlete_cvs.append(cv)

        # Median CV
        median_cv = np.median(athlete_cvs)
        stability = 1.0 / median_cv if median_cv > 0 else 0
        stability_scores[discipline] = stability

        print(f"{discipline}: Median CV = {median_cv:.6f}, Stability = {stability:.6f}")

    # Z-score normalization
    scores = np.array(list(stability_scores.values()))
    z_scores = z_score_normalize(scores)

    discipline_z = dict(zip(stability_scores.keys(), z_scores))

    print("\nStandardized stability scores:")
    for discipline, z in discipline_z.items():
        print(f"  {discipline}: {z:.6f}")

    max_z = np.max(z_scores)
    print(f"\nMaximum standardized stability score: {max_z:.3f}")

    return max_z

# ---------- Task 10: Composite Dominance Score ----------
def task10_composite_dominance(datasets, hull_data):
    """
    Compute composite dominance score for each athlete.
    Identify athlete with maximum score.
    """
    print("\n" + "="*60)
    print("TASK 10: Composite Dominance Score")
    print("="*60)

    all_athletes = []

    for discipline, df in datasets.items():
        # Identify columns
        athlete_col = None
        dist_col = None

        for col in df.columns:
            col_lower = col.lower()
            if 'athlete' in col_lower or 'name' in col_lower or 'competitor' in col_lower:
                athlete_col = col
            if 'distance' in col_lower or 'result' in col_lower or 'mark' in col_lower:
                dist_col = col

        if athlete_col is None:
            athlete_col = df.columns[0]
        if dist_col is None:
            dist_col = df.select_dtypes(include=[np.number]).columns[-1]

        # Remove rows with missing values
        df_clean = df[[athlete_col, dist_col]].dropna()

        # Compute athlete-level statistics
        athlete_stats = []
        discipline_total = df_clean[dist_col].sum()

        for athlete, group in df_clean.groupby(athlete_col):
            distances = group[dist_col].values
            mean_dist = np.mean(distances)
            std_dist = population_std(distances)
            total_dist = np.sum(distances)
            cv = coefficient_of_variation(distances) if len(distances) >= 2 else np.nan
            gini_contrib = total_dist / discipline_total if discipline_total > 0 else 0

            athlete_stats.append({
                'athlete': athlete,
                'discipline': discipline,
                'mean': mean_dist,
                'std': std_dist,
                'cv': cv,
                'gini_contrib': gini_contrib
            })

        # Identify hull vertices
        if discipline in hull_data:
            hull = hull_data[discipline]['hull']
            stats_df = hull_data[discipline]['stats_df']
            hull_vertices = set(stats_df.iloc[hull.vertices]['athlete'].values)
        else:
            hull_vertices = set()

        # Add hull vertex indicator
        for stat in athlete_stats:
            stat['is_hull_vertex'] = 1 if stat['athlete'] in hull_vertices else 0

        # Z-score normalization within discipline
        cvs = np.array([s['cv'] for s in athlete_stats if not np.isnan(s['cv'])])
        gini_contribs = np.array([s['gini_contrib'] for s in athlete_stats])

        cv_z = z_score_normalize(cvs)
        gini_z = z_score_normalize(gini_contribs)

        cv_idx = 0
        for stat in athlete_stats:
            if not np.isnan(stat['cv']):
                stat['cv_z'] = cv_z[cv_idx]
                cv_idx += 1
            else:
                stat['cv_z'] = 0

        for i, stat in enumerate(athlete_stats):
            stat['gini_z'] = gini_z[i]

        # Compute composite score
        for stat in athlete_stats:
            # Negative z-score of CV (lower CV = better)
            neg_cv_z = -stat['cv_z']
            # Hull vertex bonus
            hull_bonus = stat['is_hull_vertex']
            # Minus Gini contribution z-score
            gini_penalty = stat['gini_z']

            stat['composite'] = neg_cv_z + hull_bonus - gini_penalty

        all_athletes.extend(athlete_stats)

    # Find maximum composite score
    max_athlete = max(all_athletes, key=lambda x: x['composite'])

    print(f"\nTop athletes by composite dominance score:")
    sorted_athletes = sorted(all_athletes, key=lambda x: x['composite'], reverse=True)[:5]
    for i, a in enumerate(sorted_athletes):
        print(f"  {i+1}. {a['athlete']} ({a['discipline']}): {a['composite']:.6f}")

    print(f"\nAthlete with maximum composite dominance score: {max_athlete['athlete']}")

    return max_athlete['athlete']

# ---------- Main Execution ----------
def main():
    print("="*60)
    print("ATHLETICS PERFORMANCE ANALYSIS")
    print("Analyzing: Discus Throw, Shot Put, Javelin Throw")
    print("="*60)

    # Load datasets
    datasets = load_datasets()

    # Execute all tasks
    standardized_data, min_iqr = task1_standardized_distributions(datasets)
    min_ks = task2_ks_distance(datasets)
    bf_f_stat = task3_brown_forsythe(datasets)
    max_terminal = task4_rolling_trimmed_means(datasets)
    min_interior, hull_data = task5_convex_hull(datasets)
    between_pct, athlete_totals = task6_gini_dagum(datasets)
    max_var = task7_pca(datasets)
    p_value = task8_permutation_test(datasets)
    max_stability = task9_stability_scores(datasets)
    top_athlete = task10_composite_dominance(datasets, hull_data)

    # Summary of key outputs
    print("\n" + "="*60)
    print("KEY OUTPUTS SUMMARY")
    print("="*60)
    print(f"Task 1 - Minimum IQR: {min_iqr:.3f}")
    print(f"Task 2 - Minimum KS distance: {min_ks:.3f}")
    print(f"Task 3 - Brown-Forsythe F-statistic: {bf_f_stat:.3f}")
    print(f"Task 4 - Maximum terminal rolling median: {max_terminal:.3f}")
    print(f"Task 5 - Minimum interior proportion: {min_interior:.3f}")
    print(f"Task 6 - Between-discipline inequality: {between_pct:.2f}%")
    print(f"Task 7 - Maximum PC1 variance explained: {max_var:.3f}")
    print(f"Task 8 - Permutation p-value: {p_value:.4f}")
    print(f"Task 9 - Maximum standardized stability: {max_stability:.3f}")
    print(f"Task 10 - Top athlete: {top_athlete}")

if __name__ == "__main__":
    main()
