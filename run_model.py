"""Runs the learner/evaluator."""

from config import get_cfg_defaults
import dataset
import flame_model
import core_model


def main():
    # Load configs
    cfg = get_cfg_defaults()
    cfg.merge_from_file("flame_minimal.yaml")
    cfg.freeze()

    # Load dataset
    ds = dataset.load_dataset(cfg)

    # Add targets
    ds = dataset.add_targets(ds, cfg)

    # Pick a single snapshot from the trajectory as input to build the graph
    # for now
    inputs = {key: value[0] for key, value in ds.items()}

    # Build model
    is_training = True
    model = flame_model.Model()
    graph = model._build_graph(inputs, is_training)

    # Debugging
    gnb = core_model.GraphNetBlock(cfg)
    print(gnb)


if __name__ == "__main__":
    main()
