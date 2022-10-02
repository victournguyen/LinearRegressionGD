"""
Title: Linear Regression with Gradient Descent (An Interactive Webpage)

Author: Victor Nguyen
Date: 9/22/2022 - 10/1/2022

This program is a web application that displays an interactive graph, an
implementation of linear regression (with one independent and one dependent
variable) with gradient descent. It uses the Dash library, made by Plotly, which
uses React.js and Plotly.js to create a Flask web application. Other tools used
include Plotly, Materialize (CSS), pandas, NumPy, ImageIO, and OS.

Users can change the maximum number of iterations, or epochs, that the algorithm
goes through, its learning rate, and even the points that it attempts to find a
relationship for.

This project is optimized for uploading to Amazon Web Services, specifically for
an Elastic Beanstalk Web Server application and environment.

As for the code itself, I have used the 80-character standard to improve code
readability. Each function has a description of what it does, and comments
describe certain sections of code. The functions that are used during the
calculation process are placed at near the top of the program while those used
for the callback inputs of the interactive webpage are at the bottom of the
code. The words 'epoch' and 'iteration' are used interchangeably in variable
names and comments.
"""

#--------------------------------Library Imports-------------------------------#
from dash import Dash, Input, Output, State, html, dcc
from dash.dash_table import DataTable
from dash.dash_table.Format import Format
import dash
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import imageio.v2 as imageio
import os

#---------------------Calculation & Non-Callback Functions---------------------#
# None of these functions use global variables.
def mean_squared_error(df, b0, b1):
    """
    This function calculates the mean squared error (MSE) of the data using the
    regression equation, y = b0 + b1 * x.
    Arguments:
        - df: pandas.core.frame.DataFrame
            -- pandas dataframe with two columns, x and y (both numerical)
        - b0: float
            -- Constant value of the linear regression line equation
        - b1: float
            -- Coefficient of the x in the linear regression line equation
    """
    mean = 0
    for index, row in df.iterrows():
        # Use the difference between the actual y-value and the experimental
        # y-value (calculated from b0, b1, and the x-value)
        mean += pow(row['y'] - (b0 + b1 * row['x']), 2)
    mean /= df.shape[0]

    return mean

def make_figure(df, b0, b1, max_epochs, epoch):
    """
    This function makes figures to cache for the graph display to be able to
    update quickly when users select a different epoch to view. It uses NumPy's
    linspace method to draw the least squares regression line (LSRL) and Plotly
    to format and generate the actual figure.
    Arguments:
        - df: pandas.core.frame.DataFrame
            -- pandas dataframe with two columns, x and y (both numerical)
        - b0: float
            -- Constant value of the linear regression line equation
        - b1: float
            -- Coefficient of the x in the linear regression line equation
        - max_epochs: int
            -- Number of iterations the algorithm underwent
        - epoch: int
            -- Current epoch
    """
    # Prepare the display domain of the LSRL
    range_x_wt = 0.1 * (df['x'].max() - df['x'].min())
    lsrl_x = np.linspace(
        df['x'].min() - range_x_wt,
        df['x'].max() + range_x_wt,
        100
    )
    # Draw the scatter plot and line graph
    scatter = px.scatter(data_frame=df, x='x', y='y')
    lsrl = px.line(x=lsrl_x, y=b0 + b1 * lsrl_x)
    # Combine the scatter and line plots into one graph
    fig = go.Figure(data=scatter.data + lsrl.data, layout={
        'title': {
            # String format the current epoch to a uniform length so that
            # viewers' eyes are not required to move far when looking at
            # multiple figures in quick succession
            'text': 'Epoch {epoch:{ln}d}'.format(ln=len(str(max_epochs)),
                epoch=epoch),
            # Center title
            'x': 0.5,
        },
        'xaxis': {'title': 'x'},
        'yaxis': {'title': 'y'},
        'height': 500,
    })
    # Set the x-limits and y-limits of the figure to be the range of the x and y
    # columns of the dataframe +- 10%
    fig.update_xaxes(range=[df['x'].min() - range_x_wt, df['x'].max()
        + range_x_wt])
    range_y_wt = 0.1 * (df['y'].max() - df['y'].min())
    fig.update_yaxes(range=[df['y'].min() - range_y_wt, df['y'].max()
        + range_y_wt])
    
    return fig

def iterate_linreg(df, b0_start=1, b1_start=1, max_epochs=100,
    learning_rate=0.1, start=0, append=False):
    """
    This function is the gradient descent iteration. It iterates {max_epochs}
    times and gradually improves the LSRL. At each epoch, it takes a step in the
    opposite direction of the slope of the cost function (MSE), which is toward
    its global minimum. For linear regression, MSE is a sufficient cost function
    because it only has one local minimum, which is therefore the global
    minimum. Gradient descent is most optimal with one local minimum because it
    can be difficult to find the global minimum if multiple are present.
    Arguments:
        - df: pandas.core.frame.DataFrame
            -- pandas dataframe with two columns, x and y (both numerical)
        - b0_start: float
            -- Initial constant value of the regression
        - b1_start: float
            -- Initial x-coefficient of the regression
            -- Default starts with y = 1 + x
        - max_epochs: int
            -- Number of iterations to calculate
        - learning_rate: float
            -- Learning rate of the gradient descent (typically less than 1)
        - start: int
            -- Epoch to start at (purely for graphical display purposes)
            -- b0_start and b1_start will be specified
        - append: bool
            -- Boolean representing whether to append to the list of figures
               (True) or to create a new list (False, default)
            -- Will be True if start != 0
    """
    n = df.shape[0]
    b0 = b0_start
    b1 = b1_start
    # Make the figure with initial conditions if not appending
    figs = [make_figure(
        df=df,
        b0=b0,
        b1=b1,
        max_epochs=max_epochs,
        epoch=start
    )] if not append else []
    # Gradient descent
    for i in range(start + 1, max_epochs + 1):
        # Calculate new values for b0 and b1
        db0 = db1 = 0
        for index, row in df.iterrows():
            db0 += row['y'] - (b0 + b1 * row['x'])
            db1 += (row['y'] - (b0 + b1 * row['x'])) * row['x']
        db0 *= -2 / n
        db1 *= -2 / n
        b0 -= learning_rate * db0
        b1 -= learning_rate * db1
        # Cache the figure
        figs.append(make_figure(
            df=df,
            b0=b0,
            b1=b1,
            max_epochs=max_epochs,
            epoch=i
        ))
    # Returns b0 and b1 to store in case future calculations for the same
    # dataframe and learning rate require more epochs (reduces redundancy)
    return b0, b1, figs

def make_gif(figs, filepath, width, height):
    """
    This function stitches the images of the figures together into a single
    animated GIF, making it easier to see the progression of the gradient
    descent. First, the figures are saved to PNG images in the same location as
    the program. Then, each individual file is read and appended to the GIF.
    This process can be lengthy as Plotly does not have a feature to convert a
    list of figures directly into GIF format.
    Arguments:
        - figs: list of Plotly figures
            -- List used to cache the graphs
        - filepath: string
            -- Location of the output image(s)
        - width: int
            -- Output width of the animated GIF
        - height: int
            -- Output height of the animated GIF
    """
    # Save each of the figures individually
    i = 0
    for f in figs:
        f.write_image(
            'temp_{index}.png'.format(index=i),
            width=width,
            height=height
        )
        i += 1
    # Read all of the images and append to the animated image
    with imageio.get_writer(filepath, mode='I') as writer:
        for j in range(len(figs)):
            path = 'temp_{index}.png'.format(index=j)
            writer.append_data(imageio.imread(path))
            # Repeats the last frame 9 additional times to add a pause at the
            # end (many viewing platforms repeat the animation indefinitely)
            if j == len(figs) - 1:
                for z in range(9):
                    writer.append_data(imageio.imread(path))
            os.remove(path)

#-------------------------------Global Variables-------------------------------#
# Create the Dash app
app = Dash(__name__)
# Used for deployment
application = app.server

# Default settings
df_default = pd.read_csv('default.csv')
max_epochs_default = 100
learning_rate_default = 0.1
beta0_default, beta1_default, figures_default = iterate_linreg(
    df_default,
    1,
    1,
    max_epochs_default,
    learning_rate_default
)

# Main global variables (reset to defaults when page refreshes)
df = df_default.copy()
max_epochs = max_epochs_default
learning_rate = learning_rate_default
beta0 = beta0_default
beta1 = beta1_default
figures = figures_default.copy()
gif_df_save = gif_max_epochs_save = gif_lr_save = None

# For the download button
gif_filepath = 'figure.gif'

# Set up columns and formatting for the editable numeric DashTable: defaults to
# 0 if user enters a non-numeric value
cols = [{
    'name': i,
    'id': i,
    'type': 'numeric',
    'format': Format(),
    'on_change': {
        'action': 'coerce',
        'failure': 'default'
    },
    'validation': {
        'default': 0
    }
} for i in df_default.columns]

#-----------------------------Dash/HTML Integration----------------------------#
app.layout = html.Div(className='body', children=[
    # Page header
    html.Center(children=[
        html.H3(children='Linear Regression with Gradient Descent'),
        html.Div(id='author', children=[
            html.H5(children='Victor Nguyen')
        ])
    ]),
    # Graph and epoch slider
    html.Div(className='row', children=[
        html.Div(className='col s10 offset-s1', children=[
            dcc.Graph(id='scatter-plot'),
            dcc.Slider(
                id='epoch-slider',
                min=0,
                max=max_epochs_default,
                step=1,
                value=0,
                # 10-11 labels on the slider unless max_epochs < 10
                marks={str(z): str(z) for z in range(
                    0,
                    max_epochs_default + 1,
                    max(max_epochs_default // 10, 1)
                )}
            )
        ])
    ]),
    # Inputs displayed in row format
    html.Div(className='row', children=[
        # Input for max number of epochs - the type is text in order to
        # handle validation in the backend
        html.Div(className='col s3 offset-s3', children=[
            dcc.Input(
                id='max-epochs-input',
                type='text',
                value=max_epochs_default
            ),
            html.Label(htmlFor='max-epochs-input', children='Max Epochs'),
        ]),
        # Input for learning rate - the type is text due to issues regarding
        # numeric input with precision that is past two decimal places
        html.Div(className='col s3', children=[
            dcc.Input(
                id='learning-rate-input',
                type='text',
                value=learning_rate_default
            ),
            html.Label(htmlFor='learning-rate-input', children='Learning Rate')
        ])
    ]),
    html.Center(children=[
        # Button row
        html.Div(id='generate', className='row', children=[
            # - Submit button to generate a new list of graphs with input for
            #   max epochs, learning rate, and data points
            # - Prevents unwanted calculations for intermediate changes
            html.Div(className='col s2 offset-s4', children=[
                html.Button(
                    id='generate-button',
                    className='waves-effect waves-light btn right',
                    children='Generate Graph'
                )
            ]),
            # Download button for the animated GIF
            html.Div(className='col s2', children=[
                html.Button(
                    id='download-button',
                    className='waves-effect waves-light btn left',
                    children='Download GIF'
                ),
                dcc.Download(id='download-gif')
            ])
        ]),
        # - Editable DashTable, displays current points and calculations
        # - Users can add and remove rows
        html.Div(className='row', children=[
            html.Div(id='table-holder', className='col s2 offset-s5', children=[
                DataTable(
                    id='points',
                    data=df.to_dict('records'),
                    columns=cols,
                    editable=True,
                    row_deletable=True,
                    # Keep the table at a constant width
                    style_cell={
                        'maxWidth': 0,
                        'overflow': 'hidden',
                        'textOverflow': 'ellipsis'
                    }
                )
            ])
        ]),
        # Button to add a new observation
        html.Button(
            id='add-row',
            className='waves-effect waves-light btn',
            children='Add Row'
        )
    ]),
    # Throwaway input and output for the reset callback
    html.Div(id='hidden', className='hidden')
])

#------------------------------Callback Functions------------------------------#
# All global variables are explicitly stated after each function description.
@app.callback(
    Output('hidden', 'children'),
    Input('hidden', 'children')
)
def reset(abc):
    """
    This function resets the global variables to their default settings, which
    are defined in the 'Global Variables' section. The callback runs after every
    page refresh since it does not use any user input.
    Arguments:
        - abc: string
            -- Not used, throwaway input
    """
    global df_default, learning_rate_default, max_epochs_default, beta0_default
    global beta1_default, figures_default
    global df, learning_rate, max_epochs, beta0, beta1, figures
    global gif_df_save, gif_lr_save, gif_max_epochs_save

    # Reset to defaults
    df = df_default
    learning_rate = learning_rate_default
    max_epochs = max_epochs_default
    beta0 = beta0_default
    beta1 = beta1_default
    figures = figures_default
    gif_df_save = gif_lr_save = gif_max_epochs_save = None

    # Throwaway output
    return ''

@app.callback(
    Output('scatter-plot', 'figure'),
    Input('epoch-slider', 'value'),
)
def update_epoch(epoch):
    """
    This function is responsible for setting the display of the graph element.
    It takes input from the epoch slider, and since the figures are cached, all
    the function needs to do is select the correct one from the list.
    Arguments:
        - epoch: int
            -- Epoch selected by the user for which to display a graph
    """
    global figures

    return figures[epoch]

@app.callback(
    Output('download-gif', 'data'),
    Input('download-button', 'n_clicks'),
    prevent_initial_call=True
)
def download_gif(n_clicks):
    """
    This function generates a GIF animation of the linear regression. It takes
    input from the download button and sends the file back to the user.
    Arguments:
        - n_clicks: int
            -- Not used, only the button click is considered
    """
    global df, gif_df_save, learning_rate, gif_lr_save, max_epochs
    global gif_max_epochs_save, figures, gif_filepath

    # Since no cache is used (to reduce load times for other features), only
    # make a new GIF if the figures change, which is indicated by a change in
    # any of the inputs.
    if not (df.equals(gif_df_save) and learning_rate == gif_lr_save and
        max_epochs == gif_max_epochs_save):
        # Make and save an animation with all graphs in 960x540 resolution
        make_gif(figures, gif_filepath, 960, 540)

    # Save the state of this function call to avoid redundant generation
    gif_df_save = df
    gif_lr_save = learning_rate
    gif_max_epochs_save = max_epochs

    return dcc.send_file(gif_filepath)

@app.callback(
    Output('epoch-slider', 'max'),
    Output('epoch-slider', 'marks'),
    Input('generate-button', 'n_clicks'),
    # State is used to prevent unwanted calculations for intermediate changes
    State('max-epochs-input', 'value'),
    State('learning-rate-input', 'value'),
    State('points', 'data'),
    State('points', 'columns'),
    prevent_initial_call=True
)
def generate(n_clicks, new_max, new_lr, rows, columns):
    """
    This function is the callback function for any user input that changes the
    state of the algorithm (max epochs, learning rate, or points in the graph).
    It recalculates necessary epochs whether that be starting from where the
    iteration previously ended or completely restarting from Epoch 0. The
    handling for these three 'state' variables is done in the same function
    because they all use the 'Generate Graph' button to update the graph.
    Arguments:
        - n_clicks: int
            -- Not used, only the button click is considered
        - new_max: int
            -- New maximum number of iterations
        - new_lr: float
            -- New learning rate
        - rows: list
            -- Collection of dictionaries that each describe an observation or
               point, e.g. [{'x': 0, 'y': 6}, {'x': 1, 'y': 4}, ...]
        - columns: list 
            -- Collection of dictionaries for each column that describe the
               display properties
    """
    global df, beta0, beta1, figures, learning_rate, max_epochs
    
    # Input validation, no update if fails
    try:
        new_max = int(new_max)
        new_lr = float(new_lr)
        new_df = pd.DataFrame(data=rows, columns=[c['name'] for c in columns])
        # - New number of iterations must be positive
        # - Learning rate cannot be negative
        # - No need to update if there is no change
        if new_max <= 0 or new_lr < 0 or (
            new_max == max_epochs and
            new_lr == learning_rate and
            df.equals(new_df)
        ):
            return dash.no_update
    except:
        return dash.no_update
    
    # Don't calculate every epoch if the learning rate and points stay the same
    if new_lr == learning_rate and df.equals(new_df):
        # Only need to calculate extra epochs if they are not in the figure list
        if new_max > max_epochs:
            beta0, beta1, temp = iterate_linreg(
                new_df,
                beta0,
                beta1,
                new_max,
                new_lr,
                start=len(figures) - 1, 
                append=True
            )
            # Add new graphs to existing ones
            figures.extend(temp)
    else:
        # Go through gradient descent from the beginning
        beta0, beta1, figures = iterate_linreg(
            new_df,
            1,
            1,
            new_max,
            new_lr
        )

    # Replace the old values
    max_epochs = new_max
    learning_rate = new_lr
    df = new_df

    # Update the slider and webpage variables
    return new_max, {str(z): str(z) for z in range(0, new_max + 1,
        max(new_max // 10, 1))}

@app.callback(
    Output('points', 'data'),
    Input('add-row', 'n_clicks'),
    State('points', 'data'),
    prevent_initial_call=True
)
def add_row(n_clicks, rows):
    """
    This function adds new rows to the DataTable embedded in the user interface.
    The global dataframe is not updated because this new row is only confirmed
    when the generate button is clicked.
    Arguments:
        - n_clicks: int
            -- Not used, only the button click is considered
        - rows: list
            -- Rows as dictionary entries, e.g. [{'x': 0, 'y': 6},
               {'x': 1, 'y': 4}, ...]
    """
    # Add a new point, default (0, 0)
    rows.append({'x': 0, 'y': 0})

    return rows

# Run the application
if __name__ == '__main__':
    # For running locally
    app.run_server(debug=True, port=8080)
    # For deployment
    # application.run(port=8080)