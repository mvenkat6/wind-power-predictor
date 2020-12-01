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

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


def map_figure():
    data = mongodb_interface.fetch_turbine_weather_data(datetime.utcnow() - timedelta(days=7),
                                                        datetime.utcnow() + timedelta(days=2))

    # Extract the data from the result of the fetch
    time = [d["time"] for d in data]
    vel = np.vstack([np.sqrt(d["u"]**2 + d["v"]**2) for d in data])
    lat = data[0]["lat"]
    lon = data[0]["lon"]

    w_data = mongodb_interface.fetch_weather_data(datetime.utcnow() - timedelta(days=7),
                                                  datetime.utcnow() + timedelta(days=2))

    u = [d["u"][:,:,0] for d in w_data]
    v = [d["v"][:,:,0] for d in w_data]
    x_lon = w_data[0]["lon"]
    y_lat = w_data[0]["lat"]

    turbine_data = pd.read_csv("turbine_metadata.csv", index_col=0)

    fig = go.Figure()

    freq = 12

    for i in range(0, vel.shape[0], freq):
        lines = plt.streamplot(x_lon, y_lat, u[i], v[i]).lines
        pts = []
        for pt in lines.get_segments():
            pts.append(pt)
            pts.append(np.array([np.nan, np.nan]))
        pts = np.vstack(pts)

        fig.add_trace(
            go.Scattermapbox(
                visible=False,
                lon=pts[:,0],
                lat=pts[:,1],
                mode="lines"
            )
        )

        fig.add_trace(
            go.Scattermapbox(
                visible=False,
                lon = lon,
                lat = lat,
                mode="markers",
                customdata=turbine_data["t_hh"],
                marker=dict(
                    size=turbine_data["t_cap"],
                    sizeref=1e2,
                    color=vel[i,:],
                    showscale=True,
                    colorscale="speed",
                    colorbar=dict(),
                    cmin=0,
                    cmax=20),
                hovertemplate="Rated Power Capacity: %{marker.size:,} kW<br>Turbine Hub Height: %{customdata}<extra></extra>"),
            )

    fig.data[-1].visible=True
    fig.data[-2].visible=True

    steps = []
    for i in range(len(fig.data) // 2):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)}],
            label=str(time[freq*i])
        )
        step["args"][0]["visible"][2*i] = True
        step["args"][0]["visible"][2*i + 1] = True
        steps.append(step)

    sliders = [dict(
        active=10,
        currentvalue={"prefix": "Time: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders
    )

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_center_lon=180,
        width=1000,
        height=500,
        mapbox=dict(
            center=dict(
                lat=46,
                lon=-121
            ),
            zoom=6
        )
    )

    fig.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0})

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
            y=np.hstack((y_test, y_new)),
            line_color='red',
            name='Polynomial Regression',
            hoverinfo="skip"))

    # Set the layout parameters of the figure
    fig.update_layout(
        width=1000,
        height=500,
        autosize=False,
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


def serve_layout():
    return html.Div(children=[
        html.H1(children='Hello Dash'),

        html.Div(children="Dash: A web application framework for Python. {}".format(datetime.now())),

        #dcc.Graph(
        #    id='map-figure',
        #    figure=map_figure()
        #),

        dcc.Graph(
            id='power-figure',
            figure=power_figure()
        )
    ])


def main():
    """Main program execution"""
    app.layout = serve_layout
    app.run_server(debug=True)


if __name__ == "__main__":
    main()
