from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd

df = pd.read_csv("manual_paper_annotations.csv")

app = Dash()

app.layout = [
    html.H1(children='Empirical RL Papers 2018-2025', style={'textAlign':'center'}),
    dcc.Checklist(df.conference.unique(), df.conference.unique(), id='conference-selection'),
    dcc.RangeSlider(2018, 2025, 1, value=[2018, 2025], id='year-slider'),
    dcc.Dropdown(['empirical', "seeds", "year", "conference", None], 'year', id='x-axis'),
    dcc.Dropdown(['empirical', "seeds", "year", "conference", None], None, id='y-axis'),
    dcc.Dropdown(['empirical', "seeds", "year", "conference", None], 'conference', id='color'),
    dcc.Graph(id='graph-content')
]

@callback(
    Output('graph-content', 'figure'),
    Input('conference-selection', 'conf'),
    Input('year-slider', 'year'),
    Input('x-axis', 'x'),
    Input('y-axis', 'y'),
    Input('color', 'c')
)
def update_graph(conf, year, x, y, c):
    dff = df[(df.year==year) & df.conference==conf]
    if c is not None and x is not None and y is not None:
        return px.histogram(dff, x=x, y=y, color=c)
    elif x is not None and y is not None:
        return px.histogram(dff, x=x, y=y)
    elif x is not None and c is not None:
        return px.histogram(dff, x=x, color=c)
    elif y is not None and c is not None:
        return px.histogram(dff, y=y, color=c)

if __name__ == '__main__':
    app.run(debug=True)