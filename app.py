from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

from dash.dependencies import Input, Output
import plotly.graph_objs as go
from dash import dcc, html
import dash

import pandas as pd
import numpy as np


# 1. Data Processing
df = pd.read_csv('CHDdata.csv')
df['famhist'] = df['famhist'].replace({'Present': 1, 'Absent': 0})
df.head()


# 2. Risk Modelling
# Define feature columns and target column
feature_cols = ['sbp', 'tobacco', 'ldl', 'adiposity', 'famhist', 'typea', 'obesity', 'alcohol', 'age']
target_col = 'chd'

# Separate features and target
X = df[feature_cols]
y = df[target_col]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the numerical features
scaler = StandardScaler()
numerical_features = ['sbp', 'tobacco', 'ldl', 'adiposity', 'famhist', 'typea', 'obesity', 'alcohol', 'age']
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Initialize and train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Get the probabilities for chd being 1
prob_chd_1 = y_pred_proba[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)

# 3. Simulate Forecast
def simulate_years(initial_data, delta, N):
    data = [initial_data.copy()]
    for year in range(1, N+1):
        new_data = data[-1].copy()
        new_data['age'] = initial_data['age'] + year
        new_data['famhist'] = initial_data['famhist']
        new_data['typea'] *= delta['typea'] # (may be constant)
        new_data['tobacco'] *= delta['tobacco']
        new_data['obesity'] *= delta['obesity']
        new_data['alcohol'] *= delta['alcohol']
        new_data['sbp'] *= delta['sbp']
        new_data['ldl'] *= delta['ldl']
        new_data['adiposity'] *= delta['adiposity']
        data.append(new_data)
    
    df = pd.DataFrame(data)
    return df

def simulate_risk(initial_data, delta, N):
    return model.predict_proba(
        scaler.transform(
            simulate_years(
                initial_data, delta, N)))[:, 1]

# Define the initial health metrics for the individual

initial_data = {
    'sbp': 160,
    'tobacco': 5,
    'ldl': 5.0,
    'adiposity': 25.0,
    'famhist': 1,
    'typea': 50,
    'obesity': 25.0,
    'alcohol': 20.0,
    'age': 25
}


delta = {'sbp': 1, 
         'tobacco': 1,
         'ldl': 1,
         'adiposity': 1,
         'typea': 1,
         'obesity': 1,
         'alcohol': 1
        }


# 4. Interactive Visualtion

N = 10

# x: age
age = np.arange(initial_data['age'], initial_data['age'] + (N + 1), 1)

inaction_delta = {'sbp': 1, 
                  'tobacco': 1,
                  'ldl': 1,
                  'adiposity': 1,
                  'typea': 1,
                  'obesity': 1, 
                  'alcohol': 1
                } 


# y: inaction   
inaction_data = simulate_risk(initial_data, delta=inaction_delta, N=N)

# Create the Dash app
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.Div(
        dcc.Graph(id='risk-plot', animate=True),
        style={'display': 'flex', 'justify-content': 'center'}
    ),
    html.Div([
        html.Label('Tobacco %'),
        dcc.Slider(
            id='tobacco-slider',
            min=0,
            max=100,
            step=10,
            value=0,
            marks={i: f'-{i}%' for i in range(0, 110, 5)}
        )
    ], style={'margin': '20px', 'font-family': 'Helvetica, sans-serif', 'color': 'rgb(32,31,31)', 'padding-left': '100px', 'padding-right': '100px'}),
    html.Div([
        html.Label('Alcohol %'),
        dcc.Slider(
            id='alcohol-slider',
            min=0,
            max=100,
            step=10,
            value=0,
            marks={i: f'-{i}%' for i in range(0, 110, 5)}
        )
    ], style={'margin': '20px', 'font-family': 'Helvetica, sans-serif', 'color': 'rgb(32,31,31)', 'padding-left': '100px', 'padding-right': '100px'})
])


@app.callback(
    Output('risk-plot', 'figure'),
    Input('tobacco-slider', 'value'),
    Input('alcohol-slider', 'value')
)
def update_figure(tobacco, alcohol):
    def update_y(tobacco, alcohol):
        delta = {
            'sbp': 1,
            'tobacco': (1 - tobacco / 100),
            'ldl': 1,
            'adiposity': 1,
            'typea': 1,
            'obesity': 1,
            'alcohol': (1 - alcohol / 100)
        }
        return simulate_risk(initial_data, delta, N=N)


    y_data = update_y(tobacco, alcohol)

    # Calculate the shaded area
    shaded_area = np.trapz(inaction_data, age) - np.trapz(y_data, age)

    # Create the figure
    fig = go.Figure()

    # Add the blue line trace
    if tobacco == 0 and alcohol == 0:
        fig.add_trace(go.Scatter(
            x=age,
            y=y_data,
            mode='lines',
            line=dict(color='rgb(100,0,0)', width=2, shape='spline'),
            name='Current Risk Projection',
            showlegend=True
        ))
    else:
        fig.add_trace(go.Scatter(
            x=age,
            y=y_data,
            mode='lines',
            line=dict(color='rgb(32,31,31)', width=2, shape='spline'),
            name='Action Risk Projection',
            showlegend=True
        ))


    if tobacco != 0 or alcohol != 0:
        # If both tobacco and alcohol values are 0, return an empty figure
        fig.add_trace(go.Scatter(
        x=age,
        y=inaction_data,
        mode='lines',
        line=dict(color='red', width=2, dash='dash', shape='spline'),
        name='Inaction Risk Projection'
        ))

    # Add the shaded area trace
        fig.add_trace(go.Scatter(
            x=np.concatenate([age, age[::-1]]),
            y=np.concatenate([y_data, inaction_data[::-1]]),
            fill='toself',
            fillcolor='rgba(151,136,111,0.2)',  # Define the color and transparency of the shaded area
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            name='Optimsation'
        ))

        # Add the annotation
        fig.add_annotation(
            x=age[0] + 0.95,  # X coordinate for the label (center of x-axis)
            yref='paper',
            y=0.98,  # Y coordinate for the label (center of shaded area)
            text=f'Risk decreased by {shaded_area*100:.2f}%',  # Text for the label
            showarrow=False,
            font=dict(family='Helvetica, sans-serif', size=20, color='black'),  # Font style for the label
            bgcolor='rgba(255,255,255,0.8)',  # Background color for the label
            bordercolor='black',  # Border color for the label
            borderwidth=1,  # Border width for the label
            borderpad=4  # Padding for the border
        )

    fig.update_layout(
        title="Steven's Coronary Heart Disease Risk | 10 year projection",
        title_font=dict(size=24, family='Helvetica, sans-serif', color='rgb(32,31,31)'),
        yaxis=dict(
            title=dict(text='Risk', standoff=10),
            range=[0.2, 0.50],
            tickformat=',.0%',
            ticks='inside',
            ticklen=10,
            showline=True,
            linewidth=1,
            linecolor='black'
        ),
        xaxis=dict(
            title='Age',
            range=[min(age), max(age)],
            ticks='inside',
            ticklen=1,
            dtick=1,
            showline=True,
            linewidth=1,
            linecolor='black'
        ),
        # autosize=False,
        transition=dict(
            duration=600,  # Duration of the transition in milliseconds
            easing='cubic-in-out'  # Easing function for the transition
        ),
        width=1800,
        height=900,
        template='ggplot2',
        font=dict(family='Helvetica, sans-serif'),  
        plot_bgcolor='rgb(242,237,233)',  # Background color of the plot
    )

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
