import os

from model_operations import Model_Operator
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from connection import Api_Connection

# Function to create the line plot
def create_line_plot(x, y):
    fig = px.line(x=x, y=y, title='Inference')
    # Rename the x and y axes
    fig.update_xaxes(title_text='Time')
    fig.update_yaxes(title_text='Glucose (mg/dL)')
    # Add a red vertical line at y=128
    fig.add_shape(
        dict(
            type='line',
            y0=min(y),
            y1=max(y),
            x0=127,
            x1=127,
            line=dict(color='red', width=2),
        )
    )
    return fig

# Initialize the Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1("AI-based glucose prediction 🤖✨🔮"),

    # Input fields for email and password
    html.Div([
        html.Label("Email:"),
        dcc.Input(id='email-input', type='email', value=''),
        html.Label("Password:"),
        dcc.Input(id='password-input', type='password', value=''),
        html.Button('Submit', id='submit-button')
    ]),

    # Line plot
    dcc.Graph(id='line-plot')
])

# Callback to update the line plot when the user submits email and password
@app.callback(
    Output('line-plot', 'figure'),
    [Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('email-input', 'value'),
     dash.dependencies.State('password-input', 'value')]
)

def update_line_plot(n_clicks, email, password):
    if email and password:
        con = Api_Connection(email,password)
        data, status = con.get_connection()
        if status == 0:
            op = Model_Operator(128, 8, 0.001, 500, 512, 'lstm', 'glcse')
            x, y, x_future, y_futre = op.inference_data(email, password)
            return create_line_plot(x, y)
        else:
            print('Invalide Input')
    else:
        # If authentication fails or fields are empty, return an empty figure
        print('authentication fails')
        return px.line()

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
