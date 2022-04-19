"""Runs the learner/evaluator."""

from config import get_cfg_defaults
import dataset


def main():
	# Load configs
	cfg = get_cfg_defaults()
	cfg.merge_from_file("flame_minimal.yaml")
	cfg.freeze()

	ds = dataset.load_dataset(cfg.DATASET_PATH)
	print(ds)	

if __name__ == "__main__":
	main()
