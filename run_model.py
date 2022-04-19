"""Runs the learner/evaluator."""

from config import get_cfg_defaults
import dataset
import flame_model


def main():
	# Load configs
	cfg = get_cfg_defaults()
	cfg.merge_from_file("flame_minimal.yaml")
	cfg.freeze()

	# Load dataset
	ds = dataset.load_dataset(cfg.DATASET_PATH)
	print(ds)	

	# Build model
	model = flame_model.Model()
	model._build_graph(ds)

if __name__ == "__main__":
	main()
