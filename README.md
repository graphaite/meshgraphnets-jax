# meshgraphnets-jax
JAX implementation of MeshGraphNets.

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
