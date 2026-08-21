from pathlib import Path
from src.pipeline import Config, run

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    run(
        Config(
            data_path=ROOT / "data" / "mobile_app_interactions.csv",
            results_dir=ROOT / "results",
            figures_dir=ROOT / "figures" / "model_evaluation",
        )
    )
    print("Project pipeline completed.")
