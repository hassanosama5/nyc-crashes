import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import re
import glob
from datetime import datetime
import numpy as np


MAX_MAP_POINTS = 3000  
class CrashDataLoader:
    """Handles loading and managing split data files"""

    def _init_(self):
        self.df = None
        self.loaded = False

    def load_all_data(self):
        """Load and combine all split data files"""
        if self.loaded and self.df is not None:
            return self.df

        all_parts = []
        part_files = sorted(glob.glob('data_part_*.csv'))

        if not part_files:
            raise FileNotFoundError("No data files found. Please upload data_part_*.csv files")

        print(f"Loading {len(part_files)} data files...")

        for i, file in enumerate(part_files):
            print(f"Loading {file}...")

            df_part = pd.read_csv(file, low_memory=True)
            all_parts.append(df_part)

 
        self.df = pd.concat(all_parts, ignore_index=True)
        self.loaded = True

        print(f"Combined dataset: {len(self.df)} rows, {self.df.shape[1]} columns")
        return self.df

    def preprocess_data(self, df):
        """Preprocess the combined dataset"""

        df.columns = df.columns.str.strip()

        text_columns = ["BOROUGH", "CONTRIBUTING_FACTOR_1", "VEHICLE_TYPE",
                       "DRIVER_SEX", "VEHICLE_MAKE", "DAY_OF_WEEK"]

        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()
                df[col] = df[col].astype('category')  

        datetime_columns = ['CRASH DATE', 'CRASH_DATETIME', 'CRASH_DATETIME_VEHICLE', 'CRASH_DATETIME_CRASH']
        for col in datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

    
        if 'CRASH_DATETIME' in df.columns and df['CRASH_DATETIME'].notna().any():
            df['CRASH_DATETIME'] = pd.to_datetime(df['CRASH_DATETIME'], errors='coerce')
        elif 'CRASH DATE' in df.columns and 'CRASH TIME' in df.columns:
            df['CRASH_DATETIME'] = pd.to_datetime(
                df['CRASH DATE'].astype(str) + ' ' + df['CRASH TIME'].astype(str),
                errors='coerce'
            )

       
        if 'CRASH_DATETIME' in df.columns:
            df['YEAR'] = df['CRASH_DATETIME'].dt.year.fillna(df.get('YEAR', 2023)).astype('Int16')
            df['HOUR'] = df['CRASH_DATETIME'].dt.hour.fillna(df.get('HOUR', 12)).astype('Int8')
            df['DAY_OF_WEEK'] = df['CRASH_DATETIME'].dt.day_name().fillna(df.get('DAY_OF_WEEK', 'UNKNOWN'))
        else:
            df['YEAR'] = df.get('YEAR', df.get('YEAR_CRASH', 2023))
            df['HOUR'] = df.get('HOUR', df.get('HOUR_CRASH', 12))
            df['DAY_OF_WEEK'] = df.get('DAY_OF_WEEK', df.get('DAY_OF_WEEK_CRASH', 'UNKNOWN'))

        
        coord_columns = ["LATITUDE", "LONGITUDE"]
        for col in coord_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        
        if 'LATITUDE' in df.columns and 'LONGITUDE' in df.columns:
            invalid_coords = (
                (df['LATITUDE'] == 0) | (df['LONGITUDE'] == 0) |
                ~df['LATITUDE'].between(40.4, 41.0) |
                ~df['LONGITUDE'].between(-74.5, -73.5)
            )
            df.loc[invalid_coords, ['LATITUDE', 'LONGITUDE']] = np.nan

        return df


data_loader = CrashDataLoader()

try:
    df = data_loader.load_all_data()
    df = data_loader.preprocess_data(df)
    print("Data loaded and preprocessed successfully!")
except Exception as e:
    print(f"Error loading data: {e}")
    df = pd.DataFrame()

def create_dropdown_options():
    if df.empty:
        return [], [], [], [], []

    borough_options = [{"label": b.title(), "value": b} for b in sorted(df["BOROUGH"].dropna().unique()) if b != 'NAN']
    year_options = [{"label": str(int(y)), "value": int(y)} for y in sorted(df["YEAR"].dropna().unique())]
    vehicle_options = [{"label": v.title(), "value": v} for v in sorted(df["VEHICLE_TYPE"].dropna().unique()) if v != 'NAN']
    factor_options = [{"label": f.title(), "value": f} for f in sorted(df["CONTRIBUTING_FACTOR_1"].dropna().unique()) if f != 'NAN']
    injury_options = [
        {"label": "Persons Injured", "value": "NUMBER OF PERSONS INJURED"},
        {"label": "Pedestrians Injured", "value": "NUMBER OF PEDESTRIANS INJURED"},
        {"label": "Cyclists Injured", "value": "NUMBER OF CYCLIST INJURED"},
        {"label": "Motorists Injured", "value": "NUMBER OF MOTORIST INJURED"}
    ]
    return borough_options, year_options, vehicle_options, factor_options, injury_options

borough_options, year_options, vehicle_options, factor_options, injury_options = create_dropdown_options()

app = dash.Dash(_name_)
app.title = "NYC Crash Explorer"
server = app.server

app.layout = html.Div([
    html.H1("NYC Crash Explorer", style={'textAlign': 'center', 'marginBottom': '20px'}),

    html.Div([
        dcc.Dropdown(id='borough-select', options=borough_options, placeholder='Select Borough', multi=False),
        dcc.Dropdown(id='year-select', options=year_options, placeholder='Select Year', multi=False),
        dcc.Dropdown(id='vehicle-select', options=vehicle_options, placeholder='Select Vehicle Type', multi=False),
        dcc.Dropdown(id='factor-select', options=factor_options, placeholder='Select Contributing Factor', multi=False),
        dcc.Dropdown(id='injury-select', options=injury_options, placeholder='Select Injury Type', multi=False),
        dcc.Input(id='search-input', type='text', placeholder='Search: brooklyn, 2023, pedestrian...',
                 style={'width': '100%', 'padding': '10px'}),
        html.Button('Generate Report', id='generate-btn', n_clicks=0,
                   style={'marginTop': '10px', 'padding': '10px', 'backgroundColor': '#0074D9', 'color': 'white', 'border': 'none'})
    ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(3, 1fr)', 'gap': '15px', 'margin': '20px', 'padding': '20px', 'backgroundColor': '#f8f9fa'}),

    html.Div([
        dcc.Graph(id='crash-map'),
        dcc.Graph(id='crash-bar'),
        dcc.Graph(id='crash-line'),
        dcc.Graph(id='crash-pie'),
        dcc.Graph(id='crash-heatmap')
    ], style={'display': 'flex', 'flexDirection': 'column', 'gap': '20px'})
])

def filter_dataframe(df, borough, year, vehicle, factor, injury, search):
    """Filter dataframe based on user selections - OPTIMIZED with boolean mask"""
    if df.empty:
        return df


    mask = pd.Series(True, index=df.index)

    if borough:
        mask &= (df["BOROUGH"] == borough)
    if year:
        mask &= (df["YEAR"] == year)
    if vehicle:
        mask &= (df["VEHICLE_TYPE"] == vehicle)
    if factor:
        mask &= (df["CONTRIBUTING_FACTOR_1"] == factor)
    if injury:
        mask &= (df[injury] > 0)


    if search:
        search_lower = search.lower()

        borough_mapping = {
            'brooklyn': 'BROOKLYN',
            'manhattan': 'MANHATTAN',
            'queens': 'QUEENS',
            'bronx': 'BRONX',
            'staten': 'STATEN ISLAND',
            'staten island': 'STATEN ISLAND'
        }

        for search_term, borough_name in borough_mapping.items():
            if search_term in search_lower:
                mask &= (df["BOROUGH"] == borough_name)
                break

        year_match = re.search(r"\b(20\d{2})\b", search)
        if year_match:
            mask &= (df["YEAR"] == int(year_match.group(1)))

        if "pedestrian" in search_lower:
            mask &= (df["NUMBER OF PEDESTRIANS INJURED"] > 0)
        elif "cyclist" in search_lower or "bicycle" in search_lower:
            mask &= (df["NUMBER OF CYCLIST INJURED"] > 0)
        elif "motorist" in search_lower:
            mask &= (df["NUMBER OF MOTORIST INJURED"] > 0)
        elif "injured" in search_lower or "injury" in search_lower:
            mask &= (df["NUMBER OF PERSONS INJURED"] > 0)
        elif "killed" in search_lower or "fatality" in search_lower:
            mask &= (df["NUMBER OF PERSONS KILLED"] > 0)

    return df[mask]

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
    if df.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No data available - please check data files", height=400)
        return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig

    dff = filter_dataframe(df, borough, year, vehicle, factor, injury, search)

    if dff.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No data available for selected filters", height=400)
        return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig

    try:

        map_df = dff.dropna(subset=['LATITUDE', 'LONGITUDE'])
        total_points = len(map_df)
        
        if len(map_df) > MAX_MAP_POINTS:
            map_df = map_df.sample(n=MAX_MAP_POINTS, random_state=42)
        
        
        map_fig = go.Figure(go.Scattermapbox(
            lat=map_df["LATITUDE"],
            lon=map_df["LONGITUDE"],
            mode='markers',
            marker=dict(size=6, color='red', opacity=0.6),
            text=map_df["BOROUGH"],
            hovertemplate="<b>%{text}</b><extra></extra>"
        ))
        map_fig.update_layout(
            mapbox=dict(style="carto-positron", center=dict(lat=40.7128, lon=-73.9), zoom=9),
            margin={"r":0,"t":30,"l":0,"b":0},
            height=400,
            title=f"Crash Locations (showing {len(map_df):,} of {total_points:,} points)"
        )

    
        bar_data = dff.groupby("BOROUGH").size().reset_index(name='Count')
        bar_fig = px.bar(bar_data, x='BOROUGH', y='Count', color='BOROUGH',
                        title="Crashes by Borough", height=400)

    
        if 'CRASH_DATETIME' in dff.columns:
            line_data = dff.set_index('CRASH_DATETIME').resample('M').size().reset_index(name='Count')
            line_data.columns = ['CRASH_DATETIME', 'Count']
            line_fig = px.line(line_data, x='CRASH_DATETIME', y='Count',
                              title="Crashes Over Time (Monthly)", height=400)
        else:
            line_fig = go.Figure()
            line_fig.update_layout(title="No datetime data available", height=400)

 
        pie_data = dff['CONTRIBUTING_FACTOR_1'].value_counts().head(10).reset_index()
        pie_data.columns = ['factor', 'count']
        pie_fig = px.pie(pie_data, names='factor', values='count',
                        title="Top 10 Contributing Factors", height=400)

        
        heatmap_data = dff.groupby(['DAY_OF_WEEK', 'HOUR']).size().reset_index(name='Count')
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data['DAY_OF_WEEK'] = pd.Categorical(heatmap_data['DAY_OF_WEEK'], categories=day_order, ordered=True)
        heatmap_data = heatmap_data.sort_values('DAY_OF_WEEK')

        heatmap_fig = px.density_heatmap(heatmap_data, x='HOUR', y='DAY_OF_WEEK', z='Count',
                                         color_continuous_scale='Viridis',
                                         title="Crashes by Hour and Day of Week", height=400)

        return map_fig, bar_fig, line_fig, pie_fig, heatmap_fig

    except Exception as e:
        print(f"Error creating visualizations: {e}")
        error_fig = go.Figure()
        error_fig.update_layout(title=f"Error: {str(e)}", height=400)
        return error_fig, error_fig, error_fig, error_fig, error_fig

if _name_ == "_main_":
    app.run(debug=False)