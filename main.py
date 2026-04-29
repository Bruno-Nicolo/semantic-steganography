from semantic_stego.config.defaults import build_default_debug_config
from semantic_stego.experiments.runner import ExperimentRunner


def main() -> None:
    config = build_default_debug_config()
    runner = ExperimentRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
