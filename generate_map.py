import pandas as pd
import plotly.express as px
import os

os.makedirs('visuals', exist_ok=True)

df = pd.read_csv('data/df_features.csv')

# ── Aggregate default rate by state ──────────────────────────────────────────
state_defaults = df.groupby('State')['default'].mean().reset_index()
state_defaults['default'] = (state_defaults['default'] * 100).round(2)
state_defaults.rename(columns={'default': 'Default Rate (%)'}, inplace=True)

# ── Build choropleth ──────────────────────────────────────────────────────────
fig = px.choropleth(
    state_defaults,
    locations='State',
    locationmode='USA-states',
    color='Default Rate (%)',
    scope='usa',
    color_continuous_scale='Reds',
    title='SBA Default Rates by State (1987-2014)',
    labels={'Default Rate (%)': 'Default Rate (%)'}
)

fig.update_layout(
    title_font_size=18,
    geo=dict(bgcolor='rgba(0,0,0,0)'),
    margin=dict(l=0, r=0, t=40, b=0)
)

# ── Save for Streamlit ────────────────────────────────────────────────────────
fig.write_html('visuals/choropleth.html', include_plotlyjs='cdn')
print("Saved: visuals/choropleth.html")
