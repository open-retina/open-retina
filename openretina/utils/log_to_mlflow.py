import logging
import mlflow
import platform
import lightning.pytorch
from omegaconf import OmegaConf
import sys

log = logging.getLogger(__name__)


def log_to_mlflow(logger, model, cfg, data_info, valid_loader):
    try:
        # Get the existing run from the Lightning MLflow logger
        run_id = logger.run_id

        # Use the active run context
        with mlflow.start_run(run_id=run_id, nested=True) as run:
            # Log the model as an artifact
            mlflow.pytorch.log_model(model, "model")
            log.info("Logged model to MLflow")

            # Log dataset information
            mlflow.log_dict(data_info, "data_info.json")
            log.info("Logged data information to MLflow")

            # Log the full configuration
            mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config.json")

            # Log system information
            sys_info = {
                "python_version": sys.version,
                "platform": platform.platform(),
                "lightning_version": lightning.__version__,
            }
            mlflow.log_dict(sys_info, "system_info.json")

            # Log model summary if available
            if hasattr(model, "summarize"):
                mlflow.log_text(str(model.summarize()), "model_summary.txt")
    except Exception as e:
        log.warning(f"Failed to log artifacts to MLflow: {str(e)}")
