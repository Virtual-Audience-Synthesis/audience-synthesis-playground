# laughter-synthesis

[![Actions Status](https://github.com/tamaykut/laughter-synthesis/workflows/CI/badge.svg)](https://github.com/tamaykut/laughter-synthesis)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Tailored laughter synthesis

## Development

```shell
# Install development requirements
pip3 install -r requirements/dev.txt

# Automatically apply code format
make format
```

### Virtual Environment

```shell
# Create venv
make init_venv

# Install dependencies
sudo apt install portaudio19-dev python3-pyaudio
pip3 install -r requirements.txt
```

### Docker

```shell
# Build the development environment
docker-compose build

# Run the jupyter server
docker-compose up
```

### Dashboard
Plotly dashboard is integrated as a playground for the audio synthesis. In the dashboard, the following options can be configured and new audio synthesized:
- Number of people in the crowd
- Female vs. male ratio of the crowd
- Individual intensity level of clapping, laughing, whistling and booing

```shell
# Change directory to the project root
cd audience-synthesis-playground

# Install requirements
pip install -r dashboard/requirements.txt

# Run the dashboard server
python -m dashboard.src.app
```

## Links and Resources

### Unsupervised acoustic unit discovery for speech synthesis using discrete latent-variable neural networks

* Github Author (Benjamin van Niekerk): https://github.com/bshall?tab=repositories
* Vector-Quantized Contrastive Predictive Coding: https://github.com/bshall/VectorQuantizedCPC
* VQ-VAE for Acoustic Unit Discovery and Voice Conversion:
  * https://github.com/bshall/ZeroSpeech
  * https://bshall.github.io/ZeroSpeech/
