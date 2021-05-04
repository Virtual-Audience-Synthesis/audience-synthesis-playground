import os
import dash
import json
import time
import queue
import numpy as np
from scipy import signal
import sounddevice as sd
import plotly.graph_objs as go
import utils.audio as utils_audio
import dash_core_components as dcc
import scipy.io.wavfile as wavfile
import dash_html_components as html
from dash.dependencies import Input, Output, State


AUDIO_BLOCKSIZE = 50
SR = 44100
DURATION_IN_SEC = 10


STREAM = None
STREAM_START_TIME = None


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(
    __name__,
    update_title=None,
    external_stylesheets=external_stylesheets
)
app.title = 'Audio Synthesis'
app.layout = html.Div(
    [
        html.Div(
            className='container-fluid',
            children=[
                html.H3(
                    app.title,
                    style={
                        'color': '#CECECE',
                        'textAlign': 'center'
                    }
                ),
                
                # Audio soundwave
                dcc.Loading(
                    id='soundwave-loading',
                    type='circle',
                    children=html.Div(
                        dcc.Graph(
                            id='soundwave-fig',
                            animate=True
                        ),
                        style={'height': '200px'}
                    )
                ),
                
                # Audience Panel
                html.Table(
                    [
                        html.Tr(
                            [
                                html.Td(
                                    html.H5('Number of Persons'),
                                    style={'width': '15%'}
                                ),
                                html.Td(
                                    dcc.Input(
                                        id='n-person',
                                        type='number',
                                        value=10,
                                        placeholder='Number of Persons'
                                    ),
                                    style={'width': '15%'}
                                ),
                                html.Td(
                                    html.H5('Female Ratio (%)'),
                                    style={'width': '15%'}
                                ),
                                html.Td(
                                    dcc.Slider(
                                        id='female-slider',
                                        min=0,
                                        max=100,
                                        step=0.5,
                                        value=50,
                                        marks={
                                            0: '0',
                                            100: '100'
                                        },
                                        tooltip={
                                            'always_visible': True
                                        }
                                    ),
                                    style={'width': '25%'}
                                ),
                                html.Td(
                                    # Spawn audio
                                    html.Button('Spawn Audio', id='spawn-audio'),
                                    style={'width': '10%'}
                                ),
                                html.Td(
                                    # Start stream button
                                    html.Button('Start Stream', id='start-stream'),
                                    style={'width': '10%'}
                                ),
                                html.Td(
                                    # Stop stream button
                                    html.Button('Stop Stream', id='stop-stream'),
                                    style={'width': '10%'}
                                )
                            ]
                        )
                    ],
                    style={'width': '100%'}
                ),
                
                # Panel
                html.Table(
                    [
                        html.Tr(
                            [
                                html.Td(
                                    [
                                        dcc.Loading(
                                            id='clapping-loading',
                                            type='circle',
                                            children=html.Div(
                                                dcc.Graph(
                                                    id='claps-fig',
                                                    animate=True
                                                )
                                            )
                                        ),
                                        dcc.Slider(
                                            id='clapping-intensity',
                                            min=0,
                                            max=100,
                                            step=0.5,
                                            value=50,
                                            marks={
                                                0: '0',
                                                100: '100 '
                                            },
                                            tooltip={
                                                'always_visible': True
                                            }
                                        )
                                    ],
                                    style={'width': '50%'}
                                ),
                                html.Td(
                                    [
                                        dcc.Loading(
                                            id='whistling-loading',
                                            type='circle',
                                            children=html.Div(
                                                dcc.Graph(
                                                    id='whistles-fig',
                                                    animate=True
                                                )
                                            )
                                        ),
                                        dcc.Slider(
                                            id='whistling-intensity',
                                            min=0,
                                            max=100,
                                            step=0.5,
                                            value=50,
                                            marks={
                                                0: '0',
                                                100: '100'
                                            },
                                            tooltip={
                                                'always_visible': True
                                            }
                                        )
                                    ],
                                    style={'width': '50%'}
                                )
                            ],
                            style={'height': '160px'}
                        ),
                        html.Tr(
                            [
                                html.Td(
                                    [
                                        dcc.Loading(
                                            id='laughter-loading',
                                            type='circle',
                                            children=html.Div(
                                                dcc.Graph(
                                                    id='laughters-fig',
                                                    animate=True
                                                )
                                            )
                                        ),
                                        dcc.Slider(
                                            id='laughter-intensity',
                                            min=0,
                                            max=100,
                                            step=0.5,
                                            value=50,
                                            marks={
                                                0: '0',
                                                100: '100'
                                            },
                                            tooltip={
                                                'always_visible': True
                                            }
                                        )
                                    ],
                                    style={'width': '50%'}
                                ),
                                html.Td(
                                    [
                                        dcc.Loading(
                                            id='booing-loading',
                                            type='circle',
                                            children=html.Div(
                                                dcc.Graph(
                                                    id='boos-fig',
                                                    animate=True
                                                )
                                            )
                                        ),
                                        dcc.Slider(
                                            id='booing-intensity',
                                            min=0,
                                            max=100,
                                            step=0.5,
                                            value=50,
                                            marks={
                                                0: '0',
                                                100: '100'
                                            },
                                            tooltip={
                                                'always_visible': True
                                            }
                                        )
                                    ],
                                    style={'width': '50%'}
                                )
                            ],
                            style={'height': '160px'}
                        ),
                    ],
                    style={'width': '100%'}
                ),
            ]
        ),
        dcc.Store(id='claps'),
        dcc.Store(id='whistles'),
        dcc.Store(id='boos'),
        dcc.Store(id='laughters'),
        dcc.Store(id='audio'),
        dcc.Store(id='stream-counter', storage_type='session'),
        dcc.Interval(
            id='update',
            interval=100*10,
            n_intervals=0
        ),
    ], style={
        'fontFamily': 'Calibri, sans-serif',
        'fontSize': '16'
    }
)


@app.callback(
    [
        Output('claps', 'data'),
        Output('whistles', 'data'),
        Output('laughters', 'data'),
        Output('boos', 'data'),
        Output('audio', 'data'),
    ],
    Input('spawn-audio', 'n_clicks'),
    [
        State('n-person', 'value'),
        State('female-slider', 'value'),
        State('clapping-intensity', 'value'),
        State('whistling-intensity', 'value'),
        State('laughter-intensity', 'value'),
        State('booing-intensity', 'value')
    ]
)
def update_audio(n_clicks:int, n_person:int, female_ratio:float, 
                clapping_intensity:float, whistling_intensity:float, 
                laughter_intensity:float, booing_intensity:float):
    n_female = n_person * (female_ratio / 100)
    n_male = n_person - n_female
    
    # Spawn Claps
    n_person_clapping = int(n_person * clapping_intensity / 100)
    claps = utils_audio.spawnClaps(n_person_clapping, SR, DURATION_IN_SEC)
    
    claps = min_max_normalize(claps)
    #####
    
    # Spawn Whistles
    root_whistle = utils_audio.spawnWhistles(
        SR, 
        can_radius=200, 
        pea_radius=50, 
        bump_radius=150, 
        fipple_freq_mod=.35, 
        fipple_gain_mod=.35, 
        noise_gain=.2, 
        base_freq=5000, 
        sine_rate=4000, 
        pole=.95, 
        norm_can_loss=.97,
        load=True, 
        load_path=os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data', 
            'whistle.wav'
        )
    )
    
    whistles = np.zeros_like(claps)
    n_person_whistling = int(n_person * whistling_intensity / 100)
    for _ in range(n_person_whistling):
        start = int(np.random.uniform(0, DURATION_IN_SEC * SR - root_whistle.shape[0]))
        direction = np.random.randint(0, 2)
        whistle = utils_audio.changePitch(root_whistle, SR, np.random.uniform(-3, 3))
        
        whistles[direction, start:start + len(whistle)] += whistle
        
    whistles = min_max_normalize(whistles)
    #####
    
    # Spawn Laughters
    n_person_laughing = int(n_person * laughter_intensity / 100)
    laughters = utils_audio.spawnLaughter(
        n_person=n_person_laughing, 
        ratio_female=(female_ratio / 100), 
        female_laughter_path=os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data', 
            'female_laughter.npy'
        ), 
        male_laughter_path=os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data', 
            'male_laughter.npy'
        ), 
        stereo=True, 
        fs=SR, #BUG Not working, it uses 22050 default
        t_len=DURATION_IN_SEC #BUG Not working, it uses 10 default
    )
    
    laughters_resampled = np.array(
        [
            signal.resample(laughters[0], SR * DURATION_IN_SEC),
            signal.resample(laughters[1], SR * DURATION_IN_SEC)
        ]
    )
    
    laughters_resampled = min_max_normalize(laughters_resampled)
    #####
    
    # Spawn Female Boos
    root_female_boo, root_female_boo_sr = utils_audio.loadAudio(
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data', 
            'female_boo.m4a'
        ),
        sr=SR
    )
    root_female_boo = root_female_boo[118000:-107000] # Manual cropping
    
    female_boos = np.zeros_like(claps)
    n_female_booing = int(n_female * booing_intensity / 100)
    for _ in range(n_female_booing):
        start = int(np.random.uniform(0, DURATION_IN_SEC * SR - root_female_boo.shape[0]))
        direction = np.random.randint(0, 2)
        boo = utils_audio.changePitch(root_female_boo, root_female_boo_sr, np.random.uniform(-3, 3))

        female_boos[direction, start:start + len(boo)] += boo
        
    female_boos = min_max_normalize(female_boos)
    #####
    
    # Spawn Male Boos
    root_male_boo, root_male_boo_sr = utils_audio.loadAudio(
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data', 
            'male_boo.mp3'
        ),
        sr=SR
    )
    
    male_boos = np.zeros_like(claps)
    n_male_booing = int(n_male * booing_intensity / 100)
    for _ in range(n_male_booing):
        start = int(np.random.uniform(0, DURATION_IN_SEC * SR - root_male_boo.shape[0]))
        direction = np.random.randint(0, 2)
        boo = utils_audio.changePitch(root_male_boo, root_male_boo_sr, np.random.uniform(-3, 3))

        male_boos[direction, start:start + len(boo)] += boo
        
    male_boos = min_max_normalize(male_boos)
    #####
    
    # Combine Boos
    boos = female_boos + male_boos
    
    # Combine audios
    audio = (
        claps
        + whistles
        + laughters_resampled
        + boos
    )
    #####
    
    return claps.T, whistles.T, laughters_resampled.T, boos.T, audio.T


@app.callback(
    Output('soundwave-fig', 'figure'),
    [
        Input('audio', 'data'), 
        Input('stream-counter', 'data')
    ]
)
def update_figure(audio:np.ndarray, stream_counter:int):
    ctx = dash.callback_context
    input_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if input_id == 'audio':
        return plot_soundwave(audio, 0, 'Soundwave')
    elif input_id == 'stream-counter':
        return plot_soundwave(audio, stream_counter / SR * AUDIO_BLOCKSIZE, 'Soundwave')


@app.callback(
    Output('claps-fig', 'figure'),
    Input('claps', 'data')
)
def update_claps_figure(claps:np.ndarray):
    return plot_soundwave(claps, 0, 'Clapping Intensity (%)')


@app.callback(
    Output('whistles-fig', 'figure'),
    Input('whistles', 'data')
)
def update_whistles_figure(whistles:np.ndarray):
    return plot_soundwave(whistles, 0, 'Whistle Intensity (%)')


@app.callback(
    Output('laughters-fig', 'figure'),
    Input('laughters', 'data')
)
def update_laughters_figure(laughters:np.ndarray):
    return plot_soundwave(laughters, 0, 'Laughter Intensity (%)')


@app.callback(
    Output('boos-fig', 'figure'),
    Input('boos', 'data')
)
def update_boos_figure(boos:np.ndarray):
    return plot_soundwave(boos, 0, 'Booing Intensity (%)')


@app.callback(
    Output('stream-counter', 'data'),
    [
        Input('audio', 'data'),
        Input('start-stream', 'n_clicks'), 
        Input('stop-stream', 'n_clicks'), 
        Input('soundwave-fig', 'clickData')
    ],
    [
        State('stream-counter', 'data')
    ]
)
def update_stream_counter(
    audio:np.ndarray, 
    start_stream_click:int, 
    stop_stream_click:int, 
    clickData:dict, 
    stream_counter:int
):
    global STREAM, STREAM_START_TIME
    
    ctx = dash.callback_context
    input_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if input_id == 'audio':
        stream_counter = 0
    elif input_id == 'start-stream':
        def callback(outdata, frames, time, status):
            if status:
                print(status)
                
            try:
                data = q.get_nowait()
            except queue.Empty:
                print('Buffer is empty: increase buffersize?')
                raise sd.CallbackAbort
            if len(data) < len(outdata):
                outdata[:len(data)] = data
                outdata[len(data):] = b'\x00' * (len(outdata) - len(data))
                raise sd.CallbackStop
            else:
                outdata[:] = data
                
        if STREAM is None or not STREAM.active:
            q = queue.Queue(maxsize=AUDIO_BLOCKSIZE)
            
            STREAM_START_TIME = time.time()
            
            STREAM = sd.OutputStream(
                samplerate=SR, 
                channels=2, 
                callback=callback, 
                blocksize=AUDIO_BLOCKSIZE,
                dtype='float32'
            )
            
            with STREAM:
                while stream_counter < len(audio):
                    data = audio[stream_counter:stream_counter + AUDIO_BLOCKSIZE]
                    q.put(data)
                    stream_counter += AUDIO_BLOCKSIZE
    elif input_id == 'stop-stream':
        STREAM.stop()
        stream_counter = int((time.time() - STREAM_START_TIME) * SR)
    elif input_id == 'soundwave-fig':
        if clickData is not None:
            stream_counter = clickData['points'][0]['x'] / AUDIO_BLOCKSIZE * SR
    
    return stream_counter


def min_max_normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-15)


def plot_soundwave(audio:np.ndarray, current_x:int, title:str) -> go.Figure:
    x = np.array(audio).mean(axis=1)
    x = np.array_split(x, len(x) // SR * AUDIO_BLOCKSIZE)
    x = list(map(lambda x: np.mean(x), x))
    
    eps = 1e-15
    
    current_x = int(current_x)
    
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
    if current_x + 1 < len(x):
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
        title_text=title,
        height=200 if title == 'Soundwave' else 160,
        title_x=0.5,
        xaxis=dict(
            range=[
                0, 
                len(x)
            ],
            tickvals=[sec for sec in range(0, len(x) + 1, AUDIO_BLOCKSIZE)],
            ticktext=[sec for sec in range(0, (len(x) // AUDIO_BLOCKSIZE) + 1)]
        ),
        yaxis=dict(
            range=[
                min(x) * (1 - (0.001 * min(x) / abs(min(x) + eps))), 
                max(x) * (1 + (0.001 * max(x) / abs(max(x) + eps)))
            ],
            showticklabels=False
        ),
        margin=dict(
            l=10,
            r=10,
            b=50,
            t=0 if title == 'Soundwave' else 25,
            pad=0
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