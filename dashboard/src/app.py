import os
import dash
import base64
import numpy as np
import plotly.graph_objs as go
from .audio import read_audio
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

audio_path = os.path.join(
    os.path.dirname(
        os.path.dirname(__file__)
    ),
    'data',
    'queen.wav'
)
encoded_audio = base64.b64encode(open(audio_path, 'rb').read())
audio, sr = read_audio(audio_path)


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(
    __name__,
    update_title=None,
    external_stylesheets=external_stylesheets
)
app.title = 'Laughter Synthesis'
app.layout = html.Div(
    [
        html.Div(
            className='container-fluid',
            children=[
                html.H1(
                    app.title,
                    style={
                        'color': '#CECECE',
                        'textAlign': 'center'
                    }
                ),
                html.Div(
                    dcc.Graph(
                        id='soundwave-fig',
                        animate=True
                    )
                ),
                html.Audio(
                    id='player',
                    src='data:audio/mpeg;base64,{}'.format(encoded_audio.decode()),
                    controls=True,
                    autoPlay=False,
                    style={'width': '100%'}
                ),





                html.Table(
                    [
                        html.Tr(
                            [
                                html.Td(
                                    html.H5('Number of Person'),
                                    style={'width': '15%'}
                                ),
                                html.Td(
                                    dcc.Input(
                                        id='n-person',
                                        type='number',
                                        placeholder='Number of Person',
                                    ),
                                    style={'width': '18%'}
                                ),
                                html.Td(
                                    html.H5('Female/Male Ratio'),
                                    style={'width': '15%'}
                                ),
                                html.Td(
                                    dcc.Slider(
                                        id='female-male-slider',
                                        min=0,
                                        max=100,
                                        step=0.5,
                                        value=50,
                                        marks={
                                            0: '0',
                                            50: '50',
                                            100: '100'
                                        }
                                    ),
                                    style={'width': '18%'}
                                ),
                                html.Td(
                                    html.H5('Clapping Intensity'),
                                    style={'width': '15%'}
                                ),
                                html.Td(
                                    dcc.Slider(
                                        id='clapping',
                                        min=0,
                                        max=100,
                                        step=0.5,
                                        value=50,
                                        marks={
                                            0: '0',
                                            50: '50',
                                            100: '100'
                                        }
                                    ),
                                    style={'width': '18%'}
                                )
                            ]
                        ),
                        html.Tr(
                            [
                                html.Td(
                                    html.H5('Whistling Intensity'),
                                    style={'width': '15%'}
                                ),
                                html.Td(
                                    dcc.Slider(
                                        id='whistling',
                                        min=0,
                                        max=100,
                                        step=0.5,
                                        value=50,
                                        marks={
                                            0: '0',
                                            50: '50',
                                            100: '100'
                                        }
                                    ),
                                    style={'width': '18%'}
                                ),
                                html.Td(
                                    html.H5('Laughter Intensity'),
                                    style={'width': '15%'}
                                ),
                                html.Td(
                                    dcc.Slider(
                                        id='laughter',
                                        min=0,
                                        max=100,
                                        step=0.5,
                                        value=50,
                                        marks={
                                            0: '0',
                                            50: '50',
                                            100: '100'
                                        }
                                    ),
                                    style={'width': '18%'}
                                ),
                                html.Td(
                                    html.H5('Enthusiasm'),
                                    style={'width': '15%'}
                                ),
                                html.Td(
                                    dcc.Slider(
                                        id='enthusiasm',
                                        min=0,
                                        max=100,
                                        step=0.5,
                                        value=50,
                                        marks={
                                            0: '0',
                                            50: '50',
                                            100: '100'
                                        }
                                    ),
                                    style={'width': '18%'}
                                )
                            ]
                        )
                    ],
                    style={'width': '100%'}
                )
            ]
        ),
        dcc.Interval(
            id='update',
            interval=1000*300,
            n_intervals=0
        ),
    ], style={"fontFamily": "Calibri, sans-serif"}
)


@app.callback(
    Output('soundwave-fig', 'figure'),
    Input('soundwave-fig', 'clickData')
)
def update_soundwave_fig(clickData):
    bin_size = 20
    x = np.array_split(audio, len(audio) // sr * bin_size)
    x = list(map(lambda x: np.mean(x), x))

    current_x = 0
    if clickData is not None:
        current_x = clickData['points'][0]['x']

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.arange(current_x),
            y=x[:current_x],
            showlegend=False,
            mode='lines',
            hoverinfo='none',
            marker={'color': 'red'}
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(current_x, len(x)),
            y=x[current_x:],
            showlegend=False,
            mode='lines',
            hoverinfo='none',
            marker={'color': 'blue'}
        )
    )

    fig.update_layout(
        title_text='Soundwave',
        xaxis=dict(
            title_text='Seconds',
            tickvals=[sec for sec in range(0, len(x), bin_size)],
            ticktext=[sec for sec in range(0, len(x) // bin_size)]
        ),
        yaxis=dict(
            showticklabels=False
        )
    )
    return fig


if __name__ == '__main__':
    # TODO: debug=False
    app.run_server(
        host='0.0.0.0',
        port=os.getenv('PORT', 8500),
        debug=True
    )