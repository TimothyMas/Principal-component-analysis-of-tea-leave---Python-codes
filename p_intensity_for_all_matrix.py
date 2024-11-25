import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from io import StringIO

# Ustawienie czcionki na Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# Dane jako ciąg znaków
data_str = '''
'''

# Replace commas with dots for proper numeric conversion
data_str = data_str.replace(',', '.')

# Read data into DataFrame
df = pd.read_csv(StringIO(data_str), sep='\t')

# Extract EIC values and set as index
df['EIC'] = df['T: Chromatogram'].str.extract(r'EIC\s*(.*)')
df.set_index('EIC', inplace=True)

# Drop unnecessary columns
df = df.drop(columns=['T: Nr', 'T: RT [min]', 'T: Chromatogram'])

# Map EIC values to compound names
eic_to_compound = {

}

# Rename the DataFrame index using compound names
df.index = df.index.map(eic_to_compound)

# Rename columns to match "sample number.measurement"
df.columns = [f"{i // 3 + 1}.{i % 3 + 1}" for i in range(df.shape[1])]

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

# Map tea types to a color palette
tea_type_palette = {
    '': '#1f77b4',
    '': '#ff7f0e',
    '': '#2ca02c',
    '': '#d62728'
}

# Update column labels with tea type for the first measurement
updated_columns = []
for col in df.columns:
    sample_number = int(col.split('.')[0])  # Extract sample number
    measurement = int(col.split('.')[1])   # Extract measurement number
    if measurement == 1:  # First measurement of a sample
        updated_columns.append(sample_to_tea[sample_number])  # Replace with tea type
    else:
        updated_columns.append(col)  # Keep the original column name

# Apply the updated column names
df.columns = updated_columns

# Assign colors for the tea types in the x-axis labels
label_colors = [tea_type_palette[sample_to_tea[int(col.split('.')[0])]] if '.' in col else tea_type_palette[col] for col in updated_columns]

# Plotting the heatmap with viridis colormap
plt.figure(figsize=(18, 10))
sns.heatmap(df, cmap="viridis", annot=True, fmt=".2f", cbar_kws={'label': 'Signal Intensity'}, linewidths=0.5)
plt.xlabel("Sample and Measurement")
plt.ylabel("Compounds")
plt.xticks(rotation=45, ha='right', fontsize=10)

# Add colored tick labels for the x-axis
for tick_label, color in zip(plt.gca().get_xticklabels(), label_colors):
    tick_label.set_color(color)

plt.tight_layout()
plt.show()