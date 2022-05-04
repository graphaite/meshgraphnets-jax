# meshgraphnets-jax
JAX implementation of `meshgraphnets` (https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets). NB: this is currently work in progress.

## Setup
```
virtualenv --python=python3.6 "${ENV}"
${ENV}/bin/activate
pip install -r meshgraphnets-jax/requirements.txt
```

## Running the model
From within the `meshgraphnets-jax` directory, run:
`python run_model.py`.

## Datasets
`flame_minimal` : a minimal (single simulation trajectory) combustion CFD dataset of a jet-engine afterburner.
