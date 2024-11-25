import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from io import StringIO

from scipy.stats import f_oneway, ttest_ind
from scipy.stats import kruskal

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Define the data as a multiline string
data_str = '''
'''

# Replace commas with dots for proper numeric conversion
data_str = data_str.replace(',', '.')

# Read data into a DataFrame using StringIO
# Reading tab-separated values and extracting EIC values for further analysis
df = pd.read_csv(StringIO(data_str), sep='\t')

# Extract EIC values and set them as the index
df['EIC'] = df['T: Chromatogram'].str.extract(r'EIC\s*(.*)')
df.set_index('EIC', inplace=True)

# Drop unnecessary columns that are not relevant to the analysis
df = df.drop(columns=['T: Nr', 'T: RT [min]', 'T: Chromatogram'])

# Convert data to numeric values for computations
df = df.apply(pd.to_numeric)

# Fix potential trailing dots in EIC values (e.g., '609.35.')
df.index = df.index.str.rstrip('.')

# Map EIC values to compound names
eic_to_compound = {

}

# Keep only selected EIC values
df = df.loc[eic_to_compound.keys()]

# Map sample numbers to tea types
sample_to_tea = {}
for i in range(1, 21):
    if 1 <= i <= 5:
        tea_type = ''
    elif 6 <= i <= 10:
        tea_type = ''
    elif 11 <= i <= 15:
        tea_type = ''
    elif 16 <= i <= 20:
        tea_type = ''
    sample_to_tea[i] = tea_type

# ------------------------
# 2. Data Reshaping
# ------------------------

# Reset index to turn EIC into a column
df_reset = df.reset_index()

# Melt the DataFrame to long format
df_melt = df_reset.melt(id_vars='EIC', var_name='Sample_Replicate', value_name='Value')

# Split 'Sample_Replicate' into 'Sample' and 'Replicate'
df_melt[['Sample', 'Replicate']] = df_melt['Sample_Replicate'].str.split('.', expand=True).astype(float)
df_melt['Sample'] = df_melt['Sample'].astype(int)
df_melt['Replicate'] = df_melt['Replicate'].astype(int)

# Map EIC to Compound names
df_melt['Compound'] = df_melt['EIC'].map(eic_to_compound)

# Map Sample to Tea Type
df_melt['Tea Type'] = df_melt['Sample'].map(sample_to_tea)

# Drop the now redundant 'EIC' and 'Sample_Replicate' columns
df_melt = df_melt.drop(columns=['EIC', 'Sample_Replicate'])

# ------------------------
# 3. Outlier Removal Using IQR Method
# ------------------------

# Function to remove outliers using the IQR method
def remove_iqr_outliers(df):
    """
    Remove outliers using the IQR method for each group defined by 'Tea Type' and 'Compound'.
    """
    def iqr_filter(group):
        # Calculate Q1, Q3, and IQR for the group
        Q1 = group['Value'].quantile(0.25)
        Q3 = group['Value'].quantile(0.75)
        IQR = Q3 - Q1
        # Determine lower and upper bounds for outlier detection
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Filter out outliers
        filtered_group = group[(group['Value'] >= lower_bound) & (group['Value'] <= upper_bound)]
        # Identify and mark outliers
        outliers = group[(group['Value'] < lower_bound) | (group['Value'] > upper_bound)].copy()
        outliers['Reason'] = 'Outside IQR Range'
        return filtered_group, outliers

    filtered_groups = []
    outliers_list = []

    # Apply the IQR filter to each group
    for name, group in df.groupby(['Tea Type', 'Compound']):
        filtered_group, outliers = iqr_filter(group)
        filtered_groups.append(filtered_group)
        if not outliers.empty:
            outliers_list.append(outliers)

    filtered_df = pd.concat(filtered_groups, ignore_index=True)
    outliers_df = pd.concat(outliers_list, ignore_index=True) if outliers_list else pd.DataFrame()

    return filtered_df, outliers_df

# Apply IQR-based outlier removal
iqr_filtered_df, iqr_removed_outliers = remove_iqr_outliers(df_melt)

# Remove negative values from data
iqr_filtered_df = iqr_filtered_df[iqr_filtered_df['Value'] >= 0].copy()

# ------------------------
# 4. Visualization Using Median and IQR
# ------------------------

# Define color palette for tea types
tea_type_palette = {
    '': '#1f77b4',    # blue
    '': '#ff7f0e',    # orange
    '': '#2ca02c',                # green
    '': '#d62728'              # red
}

# Function to create bar plot with median and IQR
def create_bar_plot_median_iqr(data, title, filename):
    # Calculate median and IQR
    data_grouped = data.groupby(['Tea Type', 'Compound'])['Value'].agg(['median', 'quantile']).reset_index()
    data_grouped['Q1'] = data.groupby(['Tea Type', 'Compound'])['Value'].quantile(0.25).values
    data_grouped['Q3'] = data.groupby(['Tea Type', 'Compound'])['Value'].quantile(0.75).values
    data_grouped['IQR'] = data_grouped['Q3'] - data_grouped['Q1']
    data_grouped['Lower'] = data_grouped['Q1']
    data_grouped['Upper'] = data_grouped['Q3']

    median_pivot = data_grouped.pivot(index='Tea Type', columns='Compound', values='median')
    Q1_pivot = data_grouped.pivot(index='Tea Type', columns='Compound', values='Q1')
    Q3_pivot = data_grouped.pivot(index='Tea Type', columns='Compound', values='Q3')

    compound_order = list(eic_to_compound.values())
    median_pivot = median_pivot.reindex(columns=compound_order)
    Q1_pivot = Q1_pivot.reindex(columns=compound_order)
    Q3_pivot = Q3_pivot.reindex(columns=compound_order)

    fig, ax = plt.subplots(figsize=(16, 8))
    bar_width = 0.2
    n_tea_types = len(median_pivot.index)
    index = np.arange(len(compound_order))
    offsets = np.linspace(-bar_width*(n_tea_types-1)/2, bar_width*(n_tea_types-1)/2, n_tea_types)

    for i, (tea_type, offset) in enumerate(zip(median_pivot.index, offsets)):
        medians = median_pivot.loc[tea_type].values
        Q1 = Q1_pivot.loc[tea_type].values
        Q3 = Q3_pivot.loc[tea_type].values
        lower_errors = medians - Q1
        upper_errors = Q3 - medians
        asymmetric_error = [lower_errors, upper_errors]
        ax.bar(
            index + offset, medians, bar_width, yerr=asymmetric_error,
            label=tea_type, color=tea_type_palette.get(tea_type, None),
            capsize=5, edgecolor='grey'
        )

    ax.set_xlabel('Compounds', fontsize=14)
    ax.set_ylabel('Median Signal Intensity', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(index)
    ax.set_xticklabels(compound_order, rotation=45, ha='right', fontsize=12)
    ax.legend(title='Tea Type', fontsize=12, title_fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, format='pdf', dpi=300)
    plt.show()

# Create bar plot after IQR-based outlier removal
create_bar_plot_median_iqr(iqr_filtered_df, "Median Intensities (IQR-Based Outlier Removal)", "bar_chart_iqr.pdf")

# Function to create box plot
def create_box_plot_iqr(df, title, filename):
    # Remove negative values
    df = df[df['Value'] >= 0]
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.boxplot(x='Compound', y='Value', hue='Tea Type', data=df, palette=tea_type_palette, ax=ax)
    ax.set_xlabel('Compounds', fontsize=14)
    ax.set_ylabel('Signal Intensity', fontsize=14)
    ax.set_title(title, fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.legend(title='Tea Type', fontsize=12, title_fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, format='pdf', dpi=300)
    plt.show()

# Create box plot after IQR-based outlier removal
create_box_plot_iqr(iqr_filtered_df, "Box Plot (IQR-Based Outlier Removal)", "box_plot_iqr.pdf")

# ------------------------
# 5. Save Outliers and Statistics
# ------------------------

# Save removed outliers to CSV
if not iqr_removed_outliers.empty:
    iqr_removed_outliers.to_csv("iqr_removed_outliers.csv", index=False)

# Calculate and save statistics
def calculate_stats_median_iqr(df):
    grouped = df.groupby(['Tea Type', 'Compound'])
    stats_df = grouped['Value'].agg(['median']).reset_index()
    stats_df['Q1'] = grouped['Value'].quantile(0.25).values
    stats_df['Q3'] = grouped['Value'].quantile(0.75).values
    stats_df['IQR'] = stats_df['Q3'] - stats_df['Q1']
    return stats_df

iqr_stats = calculate_stats_median_iqr(iqr_filtered_df)
iqr_stats.to_csv("iqr_filtered_statistics.csv", index=False)

# ------------------------
# 6. Log Transformation for Further Analysis
# ------------------------

# Add a small constant to avoid log(0)
iqr_filtered_df['Log_Value'] = np.log(iqr_filtered_df['Value'] + 1e-6)

# ------------------------
# 7. PCA Analysis on Log-Transformed Data
# ------------------------

# Prepare data for PCA
pivot_df_log = iqr_filtered_df.pivot_table(index=['Sample', 'Tea Type'], columns='Compound', values='Log_Value')
pivot_df_log = pivot_df_log.dropna()

# Standardize the data
scaler = StandardScaler()
scaled_data_log = scaler.fit_transform(pivot_df_log)

# Perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data_log)

# Create a DataFrame with principal components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Tea Type'] = pivot_df_log.reset_index()['Tea Type']

# Plot PCA results
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Tea Type', data=pca_df, palette=tea_type_palette)
plt.title('PCA of Log-Transformed Compound Intensities')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Tea Type')
plt.tight_layout()
plt.savefig('pca_plot_log.pdf', format='pdf', dpi=300)
plt.show()

# ------------------------
# 8. Correlation Analysis on Log-Transformed Data
# ------------------------

# Calculate correlation matrix
correlation_matrix_log = pivot_df_log.corr()

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix_log, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Log-Transformed Compounds')
plt.tight_layout()
plt.savefig('correlation_heatmap_log.pdf', format='pdf', dpi=300)
plt.show()

# ------------------------
# 9. ANOVA Analysis on Log-Transformed Data
# ------------------------

print("\nANOVA Results on Log-Transformed Data:")
for compound in iqr_filtered_df['Compound'].unique():
    groups = []
    for tea_type in iqr_filtered_df['Tea Type'].unique():
        group = iqr_filtered_df[(iqr_filtered_df['Compound'] == compound) & (iqr_filtered_df['Tea Type'] == tea_type)]['Log_Value']
        if len(group) > 1:
            groups.append(group)
    if len(groups) > 1:
        f_stat, p_value = f_oneway(*groups)
        print(f"{compound}: F-statistic={f_stat:.2f}, p-value={p_value:.4f}")
    else:
        print(f"{compound}: Not enough data for ANOVA.")

# ------------------------
# 10. Pairwise T-tests (Optional)
# ------------------------

# Pairwise T-tests between tea types for each compound
from itertools import combinations

print("\nPairwise T-tests on Log-Transformed Data:")
tea_types = iqr_filtered_df['Tea Type'].unique()
for compound in iqr_filtered_df['Compound'].unique():
    print(f"\nCompound: {compound}")
    for tea1, tea2 in combinations(tea_types, 2):
        group1 = iqr_filtered_df[(iqr_filtered_df['Compound'] == compound) & (iqr_filtered_df['Tea Type'] == tea1)]['Log_Value']
        group2 = iqr_filtered_df[(iqr_filtered_df['Compound'] == compound) & (iqr_filtered_df['Tea Type'] == tea2)]['Log_Value']
        if len(group1) > 1 and len(group2) > 1:
            t_stat, p_val = ttest_ind(group1, group2, equal_var=False)
            print(f"{tea1} vs {tea2}: t-statistic={t_stat:.2f}, p-value={p_val:.4f}")
        else:
            print(f"Not enough data for {tea1} vs {tea2}")

# ------------------------
# 11. Histograms of Data Distribution (Optional)
# ------------------------

# Histogram before log transformation
plt.figure(figsize=(10, 6))
sns.histplot(iqr_filtered_df['Value'], bins=30, kde=True)
plt.title('Histogram of Signal Intensity (After Outlier Removal)')
plt.xlabel('Signal Intensity')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('histogram_signal_intensity.pdf', format='pdf', dpi=300)
plt.show()

# Histogram after log transformation
plt.figure(figsize=(10, 6))
sns.histplot(iqr_filtered_df['Log_Value'], bins=30, kde=True)
plt.title('Histogram of Log-Transformed Signal Intensity')
plt.xlabel('Log(Signal Intensity)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('histogram_log_signal_intensity.pdf', format='pdf', dpi=300)
plt.show()

# ------------------------
# 12. Kruskal-Wallis H-test (Optional)
# ------------------------

print("\nKruskal-Wallis H-test Results on Log-Transformed Data:")
for compound in iqr_filtered_df['Compound'].unique():
    groups = []
    for tea_type in iqr_filtered_df['Tea Type'].unique():
        group = iqr_filtered_df[(iqr_filtered_df['Compound'] == compound) & (iqr_filtered_df['Tea Type'] == tea_type)]['Log_Value']
        if len(group) > 1:
            groups.append(group)
    if len(groups) > 1:
        h_stat, p_value = kruskal(*groups)
        print(f"{compound}: H-statistic={h_stat:.2f}, p-value={p_value:.4f}")
    else:
        print(f"{compound}: Not enough data for Kruskal-Wallis test.")

# ------------------------
# 13. Mann-Whitney U Test (Optional)
# ------------------------

from scipy.stats import mannwhitneyu

print("\nPairwise Mann-Whitney U Tests on Log-Transformed Data:")
for compound in iqr_filtered_df['Compound'].unique():
    print(f"\nCompound: {compound}")
    for tea1, tea2 in combinations(tea_types, 2):
        group1 = iqr_filtered_df[(iqr_filtered_df['Compound'] == compound) & (iqr_filtered_df['Tea Type'] == tea1)]['Log_Value']
        group2 = iqr_filtered_df[(iqr_filtered_df['Compound'] == compound) & (iqr_filtered_df['Tea Type'] == tea2)]['Log_Value']
        if len(group1) > 1 and len(group2) > 1:
            u_stat, p_val = mannwhitneyu(group1, group2, alternative='two-sided')
            print(f"{tea1} vs {tea2}: U-statistic={u_stat:.2f}, p-value={p_val:.4f}")
        else:
            print(f"Not enough data for {tea1} vs {tea2}")

# ------------------------
# 14. Save Final Cleaned Data (Optional)
# ------------------------

# Save the cleaned data to CSV
iqr_filtered_df.to_csv("cleaned_data_iqr.csv", index=False)