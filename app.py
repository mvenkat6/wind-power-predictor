"""
Plotly Dash application for visualization
"""

__author__ = "Maitreya Venkataswamy"

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
import dash_core_components as dcc
import dash_html_components as html
import matplotlib.pyplot as plt

from database import MongoDBInterface
from predict import linear_model

mongodb_interface = MongoDBInterface()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', '/assets/style.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


def map_figure():
    """Creates a map figure that shows the wind patterns"""
    # Get the turbine weather data
    data = mongodb_interface.fetch_turbine_weather_data(datetime.utcnow() - timedelta(days=7),
                                                        datetime.utcnow() + timedelta(days=1))

    # Extract the data from the result of the fetch
    time = [d["time"] for d in data]
    vel = np.vstack([np.sqrt(d["u"]**2 + d["v"]**2) for d in data])
    lat = data[0]["lat"]
    lon = data[0]["lon"]

    # Get the total weather data
    w_data = mongodb_interface.fetch_weather_data(datetime.utcnow() - timedelta(days=7),
                                                  datetime.utcnow() + timedelta(days=1))

    # Get the wind data on a coarser grid
    skp = 16
    u = [d["u"][15:-5:skp, ::skp, 0] for d in w_data]
    v = [d["v"][15:-5:skp, ::skp, 0] for d in w_data]
    x_lon = w_data[0]["lon"][::skp]
    y_lat = w_data[0]["lat"][15:-5:skp]

    # Load the turbine metadata data
    turbine_data = pd.read_csv("turbine_metadata.csv", index_col=0)

    # Initialize the figure
    fig = go.Figure()

    # Add data to the plot for multiple points in time
    freq = 24
    for i in range(0, vel.shape[0], freq):
        # Compute the streamlines
        sp = plt.streamplot(x_lon, y_lat, u[i], v[i], density=2)
        lines = sp.lines

        # Combine the streamlines into a single list for plotting
        pts = []
        for pt in lines.get_segments():
            pts.append(pt)
            pts.append(np.array([np.nan, np.nan]))
        pts = np.vstack(pts)

        # Add the streamlines
        fig.add_trace(
            go.Scattermapbox(
                visible=False,
                lon=pts[:, 0],
                lat=pts[:, 1],
                mode="lines+markers",
                showlegend=False,
                hoverinfo="skip"
            )
        )

        # Add the turbine locations as a scatterplot
        fig.add_trace(
            go.Scattermapbox(
                visible=False,
                lon=lon,
                lat=lat,
                mode="markers",
                customdata=turbine_data["t_hh"],
                marker=dict(
                    size=turbine_data["t_cap"],
                    sizeref=1e2,
                    color=vel[i, :],
                    showscale=True,
                    colorscale="speed",
                    colorbar=dict(),
                    cmin=0,
                    cmax=20),
                showlegend=False,
                hovertemplate="Rated Power Capacity: %{marker.size:,} kW<br>Turbine Hub Height: %{customdata}<extra></extra>"))

    # Make the last set of data visible
    fig.data[-1].visible = True
    fig.data[-2].visible = True

    # Set up the steps for the time slider
    steps = []
    for i in range(len(fig.data) // 2):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)}],
            label=str(time[freq * i])
        )
        step["args"][0]["visible"][2 * i] = True
        step["args"][0]["visible"][2 * i + 1] = True
        steps.append(step)

    # Create the time slider
    sliders = [dict(
        active=10,
        currentvalue={"prefix": "Time Selection: "},
        pad={"t": 50},
        steps=steps
    )]

    # Add the time slider to the figure
    fig.update_layout(
        sliders=sliders
    )

    # Update the layout of the figure
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_center_lon=180,
        #width=1500,
        #height=500,
        mapbox=dict(
            center=dict(
                lat=46,
                lon=-121
            ),
            zoom=6
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )

    # Return the figure
    return fig


def power_figure():
    """Creates the power history figure"""
    # Fetch the turbine weather data from the database
    data = mongodb_interface.fetch_turbine_weather_data(datetime.utcnow() - timedelta(days=7),
                                                        datetime.utcnow() + timedelta(days=2))

    # Extract the data from the result of the fetch
    time = [d["time"] for d in data]
    power = [d["power"] for d in data]
    vel = np.vstack([np.sqrt(d["u"]**2 + d["v"]**2) for d in data])

    # Initialize the figurer
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add the power history
    fig.add_trace(
        go.Scatter(
            x=time,
            y=power,
            line_color='blue',
            name='Actual Wind Power',
            hoverinfo="skip"))

    # Add the lower bound of the wind speed interval
    fig.add_trace(
        go.Scatter(
            x=time,
            y=np.quantile(vel, 0.05, axis=1),
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip"),
        secondary_y=True)

    # Add the upper bound of the wind speed interval
    fig.add_trace(
        go.Scatter(
            x=time,
            y=np.quantile(vel, 0.95, axis=1),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(0, 100, 80, 0.2)',
            fill='tonexty',
            name="90% IQR for Wind Speed",
            hoverinfo="skip"),
        secondary_y=True)

    # Fit a linear model and predict the power
    t_test, y_test, t_new, y_new = linear_model(data)

    # Plot the linear model test results
    fig.add_trace(
        go.Scatter(
            x=t_test + t_new,
            y=np.clip(np.hstack((y_test, y_new)), 0., np.inf),
            line_color='red',
            name='Polynomial Regression',
            hoverinfo="skip"))

    # Set the layout parameters of the figure
    fig.update_layout(
        #width=1000,
        #height=500,
        autosize=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.1,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)")
    )

    # Set the axes settings
    fig.update_yaxes(title_text="Wind Power Generation [MW]",
                     secondary_y=False,
                     showgrid=False,
                     zeroline=False)

    # Set the axes settings
    fig.update_yaxes(title_text="Wind Speed [m/s]",
                     secondary_y=True,
                     showgrid=False,
                     zeroline=False)

    # Set the axes settings
    fig.update_xaxes(showgrid=False,
                     range=[time[0], time[-1]])

    # Set the plot mode to a line plot
    fig.update_traces(mode='lines')

    # Return the figure
    return fig

def description():
    """Returns project desription text"""
    return html.Div(children=[dcc.Markdown('''
        # Predicting Pacific Northwest Wind Power Using Weather Simulations
        Wind power is increasingly becoming more popular as a renewable energy source, however the generation of wind power is not the most stable source of power we don’t have 100% control over the availability of the power generation at an arbitrary time-scale. Ultimately, the availability of wind power is of course dependent on weather conditions. The good news is that weather forecasting is becoming a mature field of research. Huge increases in the availability of computational resources have allowed the generation of CFD based weather forecasts by harnessing supercomputers to perform simulations of the Earth’s atmosphere and weather at a global scale with extremely fine resolution. This project aims to demonstrate the ability to use these weather forecasts to predict the availability of wind power in the Pacific Northwest of the United States.
        ### Data Sources
        There are three sources of data for this project:
        * [Wind power production levels for the Pacific Northwest from the BPA Balancing Authority](https://transmission.bpa.gov/Business/Operations/Wind/twndbspt.aspx)
        * [Wind turbine metadata (geographic location, height, etc.) at the individual turbine level from the U.S. Wind Turbine Database](https://eerscmap.usgs.gov/uswtdb/data/)
        * [Atmospheric forecast simulation results from the Global Forecast System made available by the National Centers for Environmental Prediction](https://nomads.ncep.noaa.gov/)

        The turbine metadata can be downloaded offline once and stored either on disk or in a database, since it is relatively small and doesn’t need to be updated. Both the power availability and forecasts will need to be incrementally downloaded since they are the live component of this project. The wind power level is a small dataset and can be downloaded and put into a database as it becomes available. The forecast data is a very large dataset that is updated four times each day. This data contains wind information across the entire planet at various altitudes at hourly increments. However, only the surface wind data at the turbine locations is required, so upon accessing the data from through the appropriate API, the wind data can be interpolated using bilinear interpolation to only the locations of the turbines. Only this information needs to be stored in the database, since it is what is required to make the predictions on power availability. For the purpose of visualizing the overall wind patterns in the Pacific Northwest, it may be required to store all of the surface wind data, but it would be restricted to the Pacific Northwest in order to reduce the database requirements.

        > Note: All times in this project are in the UTC timezone.
        ''', className='eleven columns', style={'paddingLeft': '5%'})], className="row")


def map_figure_info():
    """Returns information about the map figure"""
    return html.Div(children=[dcc.Markdown('''
        ### Weather Map
        This weather map shows the wind speed at the turbines, as well as the surface wind streamlines. For the turbine winds, hovering over each individual turbine shows the rated power output and the hub height (the height above ground of the turbine hub), and the color of the circle on the map indicates the wind speed. The streamlines have dots along them that help indicate the wind speed. The closer the dots are together, the lower the wind-speed, and the larger the gap between the dots, the higher the wind speed. The slider along the bottom allows you to select a specific time to view.

        ''', className='eleven columns', style={'paddingLeft': '5%'})], className="row")


def serve_layout():
    # Contruct and return the application layout
    return html.Div([
        description(),
        map_figure_info(),

        dcc.Graph(
            id='map-figure',
            figure=map_figure(),
            style={'padding-left':'5%', 'padding-right':'5%'}
        ),

        dcc.Graph(
            id='power-figure',
            figure=power_figure(),
            style={'padding-left':'5%'}
        )
    ], className='row', id='content')


def main():
    """Main program execution"""
    # Set the layout server of the application
    app.layout = serve_layout

    # Run the application
    app.run_server(debug=True)


if __name__ == "__main__":
    main()
