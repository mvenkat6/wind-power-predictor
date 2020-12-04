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
from subprocess import Popen

from database import MongoDBInterface
from predict import linear_model

mongodb_interface = MongoDBInterface()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', '/assets/style.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
application = app.server

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

    # Load the turbine metadata data
    turbine_data = pd.read_csv("turbine_metadata.csv", index_col=0)

    # Initialize the figure
    fig = go.Figure()

    # Add data to the plot for multiple points in time
    freq = 6
    for i in range(0, vel.shape[0], freq):
        # Add the streamlines
        fig.add_trace(
            go.Scattermapbox(
                visible=False,
                lon=w_data[i]["pts"][:, 0],
                lat=w_data[i]["pts"][:, 1],
                mode="lines",
                line={"color": "purple"},
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
                    colorscale="Teal",
                    colorbar=dict(),
                    cmin=0,
                    cmax=15),
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
            y=np.clip(np.hstack((y_test, y_new)), 0., np.inf) if y_new is not None else np.clip(y_test, 0, np.inf),
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
        Wind power is increasingly becoming more popular as a renewable energy source, however the generation of wind power is not the most stable source of power; we don’t have 100% control over the availability of the power generation at an arbitrary time-scale. Ultimately, the availability of wind power is of course dependent on weather conditions. The good news is that weather forecasting is becoming a mature field of research. Huge increases in the availability of computational resources have allowed the generation of CFD based weather forecasts by harnessing supercomputers to perform simulations of the Earth’s atmosphere and weather at a global scale with extremely fine resolution. This project aims to demonstrate the ability to use these weather forecasts to predict the availability of wind power in the Pacific Northwest of the United States.
        ### Data Sources
        There are three sources of data for this project:
        * [Wind power production levels for the Pacific Northwest from the BPA Balancing Authority](https://transmission.bpa.gov/Business/Operations/Wind/twndbspt.aspx)
        * [Wind turbine metadata (geographic location, height, etc.) at the individual turbine level from the U.S. Wind Turbine Database](https://eerscmap.usgs.gov/uswtdb/data/)
        * [Atmospheric forecast simulation results from the Global Forecast System made available by the National Centers for Environmental Prediction](https://nomads.ncep.noaa.gov/)

        Both the power availability and forecasts are incrementally downloaded since they are the live component of this project. The wind power level is a small dataset and is be downloaded and put into a MongoDB database as it becomes available. The forecast data is a very large dataset that is updated four times each day. This data contains wind information across the entire planet at various altitudes at hourly increments. However, only the wind data at the turbine locations is required, so upon accessing the data from the appropriate API, the wind data is interpolated using bilinear interpolation to only the locations of the turbines. This information is stored in the MongoDB database as well for the purpose of training a model to predict the power levels, as well as for making predictions of future power levels using the forecasted weather. For the purpose of visualizing the overall wind patterns in the Pacific Northwest, it is required to store all of the surface wind data, but it is restricted to the Pacific Northwest in order to reduce the MongoDB database size.

        > Note: All times in this project are in the UTC timezone.
        ''', className='eleven columns', style={'paddingLeft': '5%'})], className="row")


def map_figure_info():
    """Returns information about the map figure"""
    return html.Div(children=[dcc.Markdown('''
        ### Weather Map
        This weather map shows the wind speed at the turbines, as well as the surface wind streamlines. For the turbine winds, hovering over each individual turbine shows the rated power output and the hub height (the height above ground of the turbine hub), and the color of the circle on the map indicates the wind speed. The slider along the bottom allows you to select a specific time to view.

        ''', className='eleven columns', style={'paddingLeft': '5%'})], className="row")


def power_figure_info():
    """Returns information about the power figure"""
    return html.Div(children=[dcc.Markdown('''
        ### Power Predictions
        The wind and temperature information at each turbine can be used as a predictor for the power generation. The figure below shows the recent history of the power generation levels, alongside a 90% inter-quantile band for the wind speed at the turbine locations. It is clear that there is a correlation between the wind speed and the power levels, which is the mechanism by which predictions can be made. A linear model with polynomial feature generation and L2 regularization is fitted to most of data that is in the figure, with the most recent available portion being reserved as an evaluation for the model. The red line below shows the predictions of the model for the power generation, and the portion of the predictions that overal with the most recent power generation data can be used to assess how accurate the model may be in the future predictions, which are shown to the right of that in the figure.
        ''', className='eleven columns', style={'paddingLeft': '5%'})], className="row")


def about_info():
    """Returns information for the about section"""
    return html.Div(children=[dcc.Markdown('''
        ### About
        This project was carried out by Maitreya Venkataswamy as a final project for the course DATA1050 (Data Engineering) at Brown University in the Fall semester of 2020. This project was inspired by these two papers: [paper 1](https://www.nrel.gov/docs/fy12osti/52233.pdf) and [paper 2](https://aip.scitation.org/doi/10.1063/1.4940208)
        ''', className='eleven columns', style={'paddingLeft': '5%'})], className="row")


def implementation_info():
    """Returns information for the implementation section"""
    return html.Div(children=[dcc.Markdown('''
        ### Implementation Details
        As mentioned before, the data is incrementally downloaded by a Python program, that checks every hour for new data and downloads it. The power data from the BPA Balancing Authority is ready to be stored in the MongoDB database, since it is just a single number for every hour. The weather data from the GFS forecasts needs to be processed. The raw wind and temperature data is extracted from the downloaded data, and then interpolated to the turbine locations. The NumPy arrays that contain this information are serialized and stored in the MongoDB database, where each document in the database contains all the information associated with a specific day and hour. In addition, the surface wind is also stored in the database along with the latitude and longitude information required to interpret it.

        The Web application that you are seeing now is implemented using Plotly Dash, and it fetches the required data from the MongoDB database whenever the user refreshed their webpage. So if you refresh this page now, the most recent data will be pulled from the database, the polynomial model will be retrained, an the webpage will be generated again. This is why the loading of page is a bit slow, since the model is always being retrained on the most recent historical data.
        ''', className='eleven columns', style={'paddingLeft': '5%'})], className="row")


def no_data_info():
    """Returns information about not having enough information yet to display"""
    return html.Div(children=[dcc.Markdown('''
        # Please wait a little bit...
        The MongoDB database was probably just initialized and is currently empty. You will need to wait a bit (~30 min) for it to populate with initial data before using the application.
        ''', className='eleven columns', style={'paddingLeft': '5%'})], className="row")


def serve_layout():
    # Contruct and return the application layout
    try:
        return html.Div([
            description(),
            map_figure_info(),
            dcc.Graph(
                id='map-figure',
                figure=map_figure(),
                style={'padding-left':'5%', 'padding-right':'5%'}
            ),
            power_figure_info(),
            dcc.Graph(
                id='power-figure',
                figure=power_figure(),
                style={'padding-left':'5%'}
            ),
            about_info()
        ], className='row', id='content')
    except Exception:
        return html.Div([
            no_data_info()
        ], className='row', id='content')


app.layout = serve_layout


def main():
    """Main program execution"""
    # Run the application
    application.run(port=8080)


if __name__ == "__main__":
    main()
