import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Ustawienie czcionki
plt.rcParams['font.family'] = 'Times New Roman'

# Dane jako ciąg znaków
data_str = '''
'''

# Replace commas with dots for correct numeric conversion
data_str = data_str.replace(',', '.')

# Read data into DataFrame
df = pd.read_csv(StringIO(data_str), sep='\t')

# Extract EIC values and set as index
df['EIC'] = df['T: Chromatogram'].str.extract(r'EIC\s*(.*)')
df.set_index('EIC', inplace=True)

# Drop unnecessary columns
df = df.drop(columns=['T: Nr', 'T: RT [min]', 'T: Chromatogram'])

# Convert data to numeric values
df = df.apply(pd.to_numeric)

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

# Initialize DataFrame for standard deviations
std_df = pd.DataFrame()
samples = []
tea_types = []

# Calculate standard deviations for each sample and compound
for i in range(1, 21):  # Samples 1 to 20
    sample_cols = [f'{i}.{j}' for j in range(1, 4)]
    if all(col in df.columns for col in sample_cols):
        sample_name = f'Sample {i}'
        samples.append(sample_name)
        tea_types.append(sample_to_tea[i])
        stds = []
        for eic in df.index:
            measurements = df.loc[eic, sample_cols].values
            std = measurements.std(ddof=1)
            stds.append(std)
        std_df[sample_name] = stds

# Transpose DataFrame for convenience
std_df.index = [eic_to_compound[eic] for eic in df.index]
std_df = std_df.T

# Add tea type information to DataFrame
std_df['Tea Type'] = tea_types

# Set sample names as index
std_df.index.name = 'Sample'

# Rearrange columns to have "Tea Type" as the last column
cols = [col for col in std_df.columns if col != 'Tea Type'] + ['Tea Type']
std_df = std_df[cols]

# Create a custom color palette for tea types
tea_type_palette = {
    '': '#1f77b4',  # blue
    '': '#ff7f0e',  # orange
    '': '#2ca02c',              # green
    '': '#d62728'            # red
}

# Create a list of colors for rows based on tea types
row_colors = std_df['Tea Type'].map(tea_type_palette)

# Remove the "Tea Type" column before plotting
std_df_plot = std_df.drop(columns=['Tea Type'])

# Set seaborn theme
sns.set_theme(style='white')

# Create figure and axis
fig, ax = plt.subplots(figsize=(14, 10))

# Create heatmap for standard deviations
cmap = sns.color_palette("crest", as_cmap=True)
sns.heatmap(std_df_plot, annot=True, fmt=".2f", cmap=cmap, linewidths=.5,
            cbar_kws={'label': 'Standard Deviation'}, ax=ax)

# Add colored labels on the left side of the plot
for tick_label, color in zip(ax.get_yticklabels(), row_colors):
    tick_label.set_color(color)

# Axis labels and title settings
ax.set_xlabel('Compounds', fontsize=14)
ax.set_ylabel('Samples', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)

# Optional: Create a legend for tea types below the plot
# from matplotlib.patches import Patch
# handles = [Patch(facecolor=color, label=label) for label, color in tea_type_palette.items()]
# ax.legend(handles=handles, title='Tea Type', loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
plt.savefig('standard_deviations.pdf', format='pdf', dpi=300)
plt.show()