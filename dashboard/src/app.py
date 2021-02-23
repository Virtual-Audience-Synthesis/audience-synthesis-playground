import os
import dash
import base64
import numpy as np
import utils.audio as audio
from .audio import read_audio
import plotly.graph_objs as go
import dash_core_components as dcc
import scipy.io.wavfile as wavfile
import dash_html_components as html
from dash.dependencies import Input, Output, State


audio_path = os.path.join(
    os.path.dirname(
        os.path.dirname(__file__)
    ),
    'data',
    'test.wav'
)
encoded_audio = base64.b64encode(open(audio_path, 'rb').read())
AUDIO, SR = read_audio(audio_path)


SOUNDWAVE_BIN_SIZE = 20
# AUDIO = None
# SR = 22050
DURATION_IN_SEC = 10


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
                
                # Audio soundwave
                html.Div(
                    dcc.Graph(
                        id='soundwave-fig',
                        animate=True
                    )
                ),
                
                # Audio
                html.Audio(
                    id='audio-player',
                    src='data:audio/mpeg;base64,{}'.format(encoded_audio.decode()),
                    controls=True,
                    autoPlay=False,
                    style={'width': '100%'}
                ),
                
                # Panel
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
                                        id='clapping-intensity',
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
                                        id='whistling-intensity',
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
                                        id='laughter-intensity',
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
                ),
                
                # Spawn button
                html.Button('Spawn Audio', id='spawn-audio')
            ]
        ),
        dcc.Interval(
            id='update',
            interval=100*5,
            n_intervals=0
        ),
    ], style={'fontFamily': 'Calibri, sans-serif'}
)


@app.callback(
    Output('soundwave-fig', 'figure'),
    [
        Input('spawn-audio', 'n_clicks'),
        Input('soundwave-fig', 'clickData'),
    ]
)
def update_soundwave_fig(n:int, clickData:dict) -> go.Figure:
    '''
    Updates the soundwave figure in the dashboard. The soundwave is colored as red until the click
    position and the rest stays blue.
    
    Args:
        n (int): Number of button clicks. Just to invoke the function.
        clickData (dict): Mouse click data.
        
    Returns:
        plotly.graph_objects.Figure: Updated soundwave figure.
    '''
    ctx = dash.callback_context
    input_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    current_x = 0
    if input_id == 'soundwave-fig':
        current_x = 0
        if clickData is not None:
            current_x = clickData['points'][0]['x']
            
    fig = plot_soundwave(current_x)
    
    return fig


def plot_soundwave(current_x:int) -> go.Figure:
    '''
    Plots the soundwave figure for given waveform. The soundwave is colored as red until the click
    position and the rest stays blue.
    
    Args:
        current_x (int): Clicked point in the figure.
        
    Returns:
        plotly.graph_objects.Figure: Soundwave figure.
    '''
    x = np.array_split(AUDIO, len(AUDIO) // SR * SOUNDWAVE_BIN_SIZE)
    x = list(map(lambda x: np.mean(x), x))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.arange(current_x + 1),
            y=x[:current_x + 1],
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
            tickvals=[sec for sec in range(0, len(x), SOUNDWAVE_BIN_SIZE)],
            ticktext=[sec for sec in range(0, len(x) // SOUNDWAVE_BIN_SIZE)]
        ),
        yaxis=dict(
            showticklabels=False
        )
    )
    
    return fig


@app.callback(
    Output('audio-player', 'src'),
    Input('spawn-audio', 'n_clicks'),
    [
        State('n-person', 'value'),
        State('female-male-slider', 'value'),
        State('clapping-intensity', 'value'),
        State('whistling-intensity', 'value'),
        State('laughter-intensity', 'value'),
        State('enthusiasm', 'value')
    ]
)
def spawn_audio(n:int, n_person:int, female_to_male_ratio:float, 
                clapping_intensity:float, whistling_intensity:float, 
                laughter_intensity:float, enthusiasm:float) -> (np.ndarray, go.Figure):
    '''
    Generates an audio based on different modules clapping, whistling, etc.
    
    Args:
        n (int): Number of button clicks. Just to invoke the function.
        n_person (int): Number of people.
        female_to_male_ratio (float): Gender ratio of the people as female / male.
        clapping_intensity (float): Intensity of clappings as percentage.
        whistling_intensity (float): Intensity of whistles as percentage.
        laughter_intensity (float): Intensity of luaghters as percentage.
        Enthusiasm (float): Enthusiasm percentage.
        
    Returns:
        numpy.ndarray, plotly.graph_objects.Figure: Audio generated. Updated soundwave figure.
    '''
    # Check None inputs
    if (n_person is not None
        and female_to_male_ratio is not None
        and clapping_intensity is not None
        and whistling_intensity is not None
        and laughter_intensity is not None
        and enthusiasm is not None):
        # Spawn clapping
        clapping = audio.spawnClaps(n_person, SR, DURATION_IN_SEC)
        
        # Combine modules
        AUDIO = clapping
        
        # Save AUDIO for playing
        location = os.path.join(
            os.path.dirname(
                os.path.dirname(__file__)
            ),
            'data',
            'test.wav'
        )
        wavfile.write(location, SR, AUDIO.T.astype(np.float32))
        
        return 'data:audio/mpeg;base64,{}'.format(
            base64.b64encode(open(location, 'rb').read()).decode())
        
        
if __name__ == '__main__':
    # TODO: debug=False
    app.run_server(
        host='0.0.0.0',
        port=os.getenv('PORT', 8500),
        debug=True
    )