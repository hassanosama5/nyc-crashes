import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import re

# Load cleaned dataset
df = pd.read_csv("cleaned_data.csv")

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Standardize text columns
for col in ["BOROUGH", "CONTRIBUTING_FACTOR_1", "VEHICLE_TYPE"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.upper()

# Convert datetime
df['CRASH_DATETIME'] = pd.to_datetime(df.get('CRASH_DATETIME', pd.Series([])), errors='coerce')

# Create YEAR, HOUR, DAY_OF_WEEK if missing
df['YEAR'] = df.get('YEAR', df['CRASH_DATETIME'].dt.year)
df['HOUR'] = df.get('HOUR', df['CRASH_DATETIME'].dt.hour)
df['DAY_OF_WEEK'] = df.get('DAY_OF_WEEK', df['CRASH_DATETIME'].dt.day_name())

# Ensure LATITUDE and LONGITUDE are numeric
df["LATITUDE"] = pd.to_numeric(df.get("LATITUDE", 0), errors='coerce').fillna(0)
df["LONGITUDE"] = pd.to_numeric(df.get("LONGITUDE", 0), errors='coerce').fillna(0)

# Dropdown options
borough_options = [{"label": b.title(), "value": b} for b in sorted(df["BOROUGH"].dropna().unique())]
year_options = [{"label": str(y), "value": y} for y in sorted(df["YEAR"].dropna().unique())]
vehicle_options = [{"label": v.title(), "value": v} for v in sorted(df["VEHICLE_TYPE"].dropna().unique())]
factor_options = [{"label": f.title(), "value": f} for f in sorted(df["CONTRIBUTING_FACTOR_1"].dropna().unique())]
injury_options = [
    {"label": "Persons Injured", "value": "NUMBER OF PERSONS INJURED"},
    {"label": "Pedestrians Injured", "value": "NUMBER OF PEDESTRIANS INJURED"},
    {"label": "Cyclists Injured", "value": "NUMBER OF CYCLIST INJURED"},
    {"label": "Motorists Injured", "value": "NUMBER OF MOTORIST INJURED"}
]

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "NYC Crash Explorer"

server = app.server

# App layout
app.layout = html.Div([
    html.H1("NYC Crash Explorer", style={'textAlign': 'center'}),
    
    html.Div([
        dcc.Dropdown(id='borough-select', options=borough_options, placeholder='Select Borough'),
        dcc.Dropdown(id='year-select', options=year_options, placeholder='Select Year'),
        dcc.Dropdown(id='vehicle-select', options=vehicle_options, placeholder='Select Vehicle Type'),
        dcc.Dropdown(id='factor-select', options=factor_options, placeholder='Select Contributing Factor'),
        dcc.Dropdown(id='injury-select', options=injury_options, placeholder='Select Injury Type'),
        dcc.Input(id='search-input', type='text', placeholder='Type search query...', style={'width': '100%'}),
        html.Button('Generate Report', id='generate-btn', n_clicks=0, style={'marginTop': '10px'})
    ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(3, 1fr)', 'gap': '10px', 'margin': '20px'}),
    
    html.Div([
        dcc.Graph(id='crash-map'),
        dcc.Graph(id='crash-bar'),
        dcc.Graph(id='crash-line'),
        dcc.Graph(id='crash-pie'),
        dcc.Graph(id='crash-heatmap')
    ])
])

# Function to filter dataframe
def filter_dataframe(df, borough, year, vehicle, factor, injury, search):
    dff = df.copy()

    if borough:
        dff = dff[dff["BOROUGH"] == borough]
    if year:
        dff = dff[dff["YEAR"] == year]
    if vehicle:
        dff = dff[dff["VEHICLE_TYPE"] == vehicle]
    if factor:
        dff = dff[dff["CONTRIBUTING_FACTOR_1"] == factor]
    if injury:
        dff = dff[dff[injury] > 0]

    if search:
        search_lower = search.lower()
        if "brooklyn" in search_lower:
            dff = dff[dff["BOROUGH"] == "BROOKLYN"]
        elif "manhattan" in search_lower:
            dff = dff[dff["BOROUGH"] == "MANHATTAN"]

        year_match = re.search(r"\b(20\d{2})\b", search)
        if year_match:
            dff = dff[dff["YEAR"] == int(year_match.group(1))]

        if "pedestrian" in search_lower:
            dff = dff[dff["NUMBER OF PEDESTRIANS INJURED"] > 0]
        elif "cyclist" in search_lower:
            dff = dff[dff["NUMBER OF CYCLIST INJURED"] > 0]
        elif "motorist" in search_lower:
            dff = dff[dff["NUMBER OF MOTORIST INJURED"] > 0]

    # Remove invalid coordinates
    dff = dff[(dff['LATITUDE'] != 0) & (dff['LONGITUDE'] != 0)]
    return dff

# Main callback
@app.callback(
    [
        Output('crash-map', 'figure'),
        Output('crash-bar', 'figure'),
        Output('crash-line', 'figure'),
        Output('crash-pie', 'figure'),
        Output('crash-heatmap', 'figure')
    ],
    [Input('generate-btn', 'n_clicks')],
    [
        State('borough-select', 'value'),
        State('year-select', 'value'),
        State('vehicle-select', 'value'),
        State('factor-select', 'value'),
        State('injury-select', 'value'),
        State('search-input', 'value')
    ]
)
def update_visuals(n_clicks, borough, year, vehicle, factor, injury, search):
    dff = filter_dataframe(df, borough, year, vehicle, factor, injury, search)

    if dff.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No data available for selected filters")
        return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig

    # Map
    map_fig = px.scatter_mapbox(
        dff, lat="LATITUDE", lon="LONGITUDE", hover_name="BOROUGH",
        mapbox_style="carto-positron", zoom=9, height=500
    )

    # Bar
    bar_data = dff.groupby("BOROUGH").size().reset_index(name='Count')
    bar_fig = px.bar(bar_data, x='BOROUGH', y='Count', color='BOROUGH', title="Crashes by Borough")

    # Line
    line_data = dff.groupby(dff['CRASH_DATETIME'].dt.date).size().reset_index(name='Count')
    line_fig = px.line(line_data, x='CRASH_DATETIME', y='Count', title="Crashes Over Time")

    # Pie
    pie_data = dff['CONTRIBUTING_FACTOR_1'].value_counts().reset_index()
    pie_data.columns = ['factor', 'count']
    pie_fig = px.pie(pie_data, names='factor', values='count', title="Contributing Factors")

    # Heatmap
    heatmap_data = dff.groupby(['DAY_OF_WEEK', 'HOUR']).size().reset_index(name='Count')
    heatmap_fig = px.density_heatmap(
        heatmap_data, x='HOUR', y='DAY_OF_WEEK', z='Count',
        color_continuous_scale='Viridis', title="Crashes by Hour and Day"
    )

    return map_fig, bar_fig, line_fig, pie_fig, heatmap_fig

# Run app
if __name__ == "__main__":
    app.run(debug=True)