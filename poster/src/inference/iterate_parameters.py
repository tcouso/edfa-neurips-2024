from src.config import get_logger
from src.inference.change_detection.run_change_detection_param_iter import (
    run_change_detection_param_iter,
)

def main():
    logger = get_logger()

    message = "Param iteration over all"
    logger.info(message)
    run_change_detection_param_iter("all")

    message = "Done"
    logger.info(message)

if __name__ == "__main__":
    main()
