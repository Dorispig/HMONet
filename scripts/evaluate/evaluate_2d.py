"""Run selected-method 2-D parametric evaluation from YAML or stored samples."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.configuration import load_configuration
from scripts.evaluate.evaluation_common import run_evaluation


def main(argv=None):
    args, config = load_configuration(2, argv)
    return run_evaluation(2, args, config)


if __name__ == "__main__":
    main()
