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

# ---------- Column Names ----------
ATHLETE_COL = 'Competitor'
DISTANCE_COL = 'Mark'
ATTEMPT_COL = 'Rank'

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
    if A == 0:
        A = std
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
    """Standardize using z-score normalization with population std."""
    arr = np.array(arr, dtype=float)
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
        # Remove records with missing distance values
        df_clean = df[[DISTANCE_COL]].dropna()
        distances = df_clean[DISTANCE_COL].values

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
        # Remove records with missing distance values
        df_clean = df[[DISTANCE_COL]].dropna()
        distances = df_clean[DISTANCE_COL].values

        # Exclude lowest 10% using empirical quantile
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
        # Remove rows with missing values in referenced variables
        df_clean = df[[ATHLETE_COL, DISTANCE_COL]].dropna()

        # Compute CV for each athlete (requires >= 2 observations for meaningful CV)
        athlete_cvs = []
        for athlete, group in df_clean.groupby(ATHLETE_COL):
            distances = group[DISTANCE_COL].values
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
        # Remove rows with missing values in referenced variables
        df_clean = df[[ATHLETE_COL, DISTANCE_COL, ATTEMPT_COL]].dropna()

        # Compute rolling trimmed means for each athlete
        window_size = 3
        all_rolling_values = {}

        for athlete, group in df_clean.groupby(ATHLETE_COL):
            # Order by attempt index (Rank)
            group_sorted = group.sort_values(ATTEMPT_COL)
            distances = group_sorted[DISTANCE_COL].values

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
        # Remove rows with missing values in referenced variables
        df_clean = df[[ATHLETE_COL, DISTANCE_COL]].dropna()

        # Compute athlete-level statistics
        athlete_stats = []
        for athlete, group in df_clean.groupby(ATHLETE_COL):
            distances = group[DISTANCE_COL].values
            mean_dist = np.mean(distances)
            std_dist = population_std(distances)
            athlete_stats.append({'athlete': athlete, 'mean': mean_dist, 'std': std_dist})

        stats_df = pd.DataFrame(athlete_stats)
        points = stats_df[['mean', 'std']].values

        # Construct convex hull using Quickhull
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
    Report between-discipline inequality percentage (using gross between-group component).
    """
    print("\n" + "="*60)
    print("TASK 6: Gini Coefficient and Dagum Decomposition")
    print("="*60)

    athlete_totals = {}

    for discipline, df in datasets.items():
        # Remove rows with missing values in referenced variables
        df_clean = df[[ATHLETE_COL, DISTANCE_COL]].dropna()

        # Compute total distance per athlete
        totals = df_clean.groupby(ATHLETE_COL)[DISTANCE_COL].sum().values
        athlete_totals[discipline] = totals

        gini = gini_coefficient(totals)
        print(f"{discipline}: {len(totals)} athletes, Gini = {gini:.6f}")

    # Dagum decomposition
    all_totals = np.concatenate(list(athlete_totals.values()))
    overall_gini = gini_coefficient(all_totals)
    print(f"\nOverall Gini: {overall_gini:.6f}")

    disciplines = list(athlete_totals.keys())
    n = len(all_totals)
    total_sum = np.sum(all_totals)

    # Gross between-group component (Ggb) - Dagum method
    Ggb = 0
    for i, d_i in enumerate(disciplines):
        for j, d_j in enumerate(disciplines):
            if i < j:
                y_i = athlete_totals[d_i]
                y_j = athlete_totals[d_j]
                n_i, n_j = len(y_i), len(y_j)
                mu_i, mu_j = np.mean(y_i), np.mean(y_j)
                p_i, p_j = n_i / n, n_j / n
                s_i = (n_i * mu_i) / total_sum
                s_j = (n_j * mu_j) / total_sum

                # Gross between d_ij (mean absolute difference)
                diff_sum = 0
                for x in y_i:
                    for y in y_j:
                        diff_sum += abs(x - y)
                d_ij = diff_sum / (n_i * n_j * (mu_i + mu_j))

                Ggb += d_ij * (p_i * s_j + p_j * s_i)

    between_pct = (Ggb / overall_gini) * 100 if overall_gini > 0 else 0

    print(f"Gross between-group component (Ggb): {Ggb:.6f}")
    print(f"Between-discipline inequality contribution: {between_pct:.2f}%")

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
    Perform PCA on athlete-level standardized mean distances (single variable).
    With single variable, PC1 explains 100% of variance.
    Report maximum variance explained by PC1.
    """
    print("\n" + "="*60)
    print("TASK 7: Principal Component Analysis")
    print("="*60)

    variance_explained = {}

    for discipline, df in datasets.items():
        # Remove rows with missing values in referenced variables
        df_clean = df[[ATHLETE_COL, DISTANCE_COL]].dropna()

        # Compute athlete-level means
        athlete_means = df_clean.groupby(ATHLETE_COL)[DISTANCE_COL].mean().values

        # Standardize within discipline
        standardized = z_score_normalize(athlete_means)

        # Single variable PCA - reshape for matrix operations
        X = standardized.reshape(-1, 1)
        X_centered = X - X.mean(axis=0)

        # Covariance-based PCA with population variance
        cov_matrix = np.dot(X_centered.T, X_centered) / len(X_centered)
        eigenvalues = np.linalg.eigvalsh(cov_matrix)

        # With single variable, PC1 explains 100% of variance
        total_var = np.sum(eigenvalues)
        pc1_var = eigenvalues[0] / total_var if total_var > 0 else 1.0

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
        # Remove rows with missing values in referenced variables
        df_clean = df[[ATHLETE_COL, DISTANCE_COL]].dropna()

        # Compute athlete-level means
        athlete_means = df_clean.groupby(ATHLETE_COL)[DISTANCE_COL].mean().values

        # Standardize within discipline
        standardized = z_score_normalize(athlete_means)
        standardized_means[discipline] = standardized
        print(f"{discipline}: {len(standardized)} athletes, Median = {np.median(standardized):.6f}")

    # Pairwise permutation tests
    disciplines = list(standardized_means.keys())
    p_values = {}

    for i in range(len(disciplines)):
        for j in range(i + 1, len(disciplines)):
            d1, d2 = disciplines[i], disciplines[j]
            data1, data2 = standardized_means[d1].copy(), standardized_means[d2].copy()

            # Observed statistic
            observed_stat = abs(np.median(data1) - np.median(data2))

            # Combined data
            combined = np.concatenate([data1, data2])
            n1 = len(data1)

            # Permutation test with fixed seed
            np.random.seed(RANDOM_SEED)
            n_permutations = 10000
            count_extreme = 0

            for _ in range(n_permutations):
                np.random.shuffle(combined)
                perm_stat = abs(np.median(combined[:n1]) - np.median(combined[n1:]))
                if perm_stat >= observed_stat:
                    count_extreme += 1

            p_value = count_extreme / n_permutations
            pair = f"{d1} vs {d2}"
            p_values[pair] = p_value
            print(f"{pair}: Observed stat = {observed_stat:.6f}, P-value = {p_value:.4f}")

    # Report minimum p-value
    min_p_value = min(p_values.values())
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
        # Remove rows with missing values in referenced variables
        df_clean = df[[ATHLETE_COL, DISTANCE_COL]].dropna()

        # Compute CV for each athlete (requires >= 2 observations)
        athlete_cvs = []
        for athlete, group in df_clean.groupby(ATHLETE_COL):
            distances = group[DISTANCE_COL].values
            if len(distances) >= 2:
                cv = coefficient_of_variation(distances)
                if not np.isnan(cv) and cv > 0:
                    athlete_cvs.append(cv)

        # Median CV
        median_cv = np.median(athlete_cvs) if athlete_cvs else 0
        stability = 1.0 / median_cv if median_cv > 0 else 0
        stability_scores[discipline] = stability

        print(f"{discipline}: Median CV = {median_cv:.6f}, Stability = {stability:.6f}")

    # Z-score normalization across disciplines
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
    Formula: -z(CV) + z(hull_vertex) - z(gini_contrib)
    All components z-score normalized within discipline.
    Include athletes with single observations (CV=0).
    """
    print("\n" + "="*60)
    print("TASK 10: Composite Dominance Score")
    print("="*60)

    all_athletes = []

    for discipline, df in datasets.items():
        # Remove rows with missing values in referenced variables
        df_clean = df[[ATHLETE_COL, DISTANCE_COL]].dropna()
        discipline_total = df_clean[DISTANCE_COL].sum()

        # Compute athlete-level statistics
        athlete_stats = []
        for athlete, group in df_clean.groupby(ATHLETE_COL):
            distances = group[DISTANCE_COL].values
            total_dist = np.sum(distances)
            # CV for single observation is 0 (std=0)
            cv = coefficient_of_variation(distances)
            gini_contrib = total_dist / discipline_total if discipline_total > 0 else 0

            athlete_stats.append({
                'athlete': athlete,
                'discipline': discipline,
                'cv': cv,
                'gini_contrib': gini_contrib
            })

        # Identify hull vertices
        if discipline in hull_data:
            hull = hull_data[discipline]['hull']
            stats_df_hull = hull_data[discipline]['stats_df']
            hull_vertices = set(stats_df_hull.iloc[hull.vertices]['athlete'].values)
        else:
            hull_vertices = set()

        # Add hull vertex indicator
        for stat in athlete_stats:
            stat['is_hull_vertex'] = 1 if stat['athlete'] in hull_vertices else 0

        # Include ALL athletes (including those with CV=0 from single observation)
        valid_stats = [s for s in athlete_stats if not np.isnan(s['cv'])]

        # Z-score normalize ALL components within discipline
        cvs = np.array([s['cv'] for s in valid_stats])
        hull_indicators = np.array([s['is_hull_vertex'] for s in valid_stats])
        gini_contribs = np.array([s['gini_contrib'] for s in valid_stats])

        cv_z = z_score_normalize(cvs)
        hull_z = z_score_normalize(hull_indicators)
        gini_z = z_score_normalize(gini_contribs)

        # Compute composite score
        for i, stat in enumerate(valid_stats):
            stat['cv_z'] = cv_z[i]
            stat['hull_z'] = hull_z[i]
            stat['gini_z'] = gini_z[i]
            # Formula: -z(CV) + z(hull) - z(gini_contrib)
            stat['composite'] = -cv_z[i] + hull_z[i] - gini_z[i]

        all_athletes.extend(valid_stats)

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
