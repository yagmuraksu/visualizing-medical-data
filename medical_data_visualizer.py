import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv("medical_examination.csv")

# Add 'overweight' columns
df['overweight'] = df['overweight'] = df['weight'] / ((df['height'] * 0.01)**2)
df['overweight'] = df['overweight'].apply(lambda x: 1 if x > 25 else 0)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df['gluc'] = df['gluc'].apply(lambda x: 1 if x > 1 else 0)
df['cholesterol'] = df['cholesterol'].apply(lambda x: 1 if x > 1 else 0)


# Draw Categorical Plot
def draw_cat_plot():
  # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
  df_cat = df.melt(id_vars=['cardio'],
                   value_vars=[
                     'active', 'alco', 'cholesterol', 'gluc', 'overweight',
                     'smoke', 'cardio'
                   ])

  # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
  df_cat = pd.DataFrame(df_cat.value_counts()).reset_index()
  df_cat.columns = ['cardio', 'variable', 'value', 'total']
  df_cat.sort_values(by=['variable'], inplace=True)
  df_cat.dropna()
  # Draw the catplot with 'sns.catplot()'
  # Get the figure for the output
  fig = sns.catplot(data=df_cat,
                    kind='bar',
                    x='variable',
                    y='total',
                    col='cardio',
                    hue='value').fig

  # Do not modify the next two lines
  fig.savefig('catplot.png')
  return fig


# Draw Heat Map
def draw_heat_map():
  # Clean the data
  df_heat = df
  df_heat = df_heat[(df_heat['ap_lo'] <= df_heat['ap_hi'])
                    & (df_heat['height'] >= df_heat['height'].quantile(0.025))
                    & (df_heat['height'] <= df_heat['height'].quantile(0.975))
                    & (df_heat['weight'] >= df_heat['weight'].quantile(0.025))
                    & (df_heat['weight'] <= df_heat['weight'].quantile(0.975))]

  # Calculate the correlation matrix
  corr = df_heat.corr()

  # Generate a mask for the upper triangle
  mask = np.triu(np.ones_like(corr, dtype=bool))

  # Set up the matplotlib figure
  fig, ax = plt.subplots()

  # Draw the heatmap with 'sns.heatmap()'
  sns.heatmap(corr, annot=True, mask=mask, vmin=-0.10, vmax=0.25, fmt='0.1f')

  # Do not modify the next two lines
  fig.savefig('heatmap.png')
  return fig
