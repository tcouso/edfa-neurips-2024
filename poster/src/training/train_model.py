from src.training.train_esn import train_esn
from src.config import get_logger


def main():
    logger = get_logger()

    try:
        message = "Starting model training"
        logger.info(message)

        train_esn()

        message = "Model training done"
        logger.info(message)

    except Exception as e:
        message = f"ESN training failed: {e}"
        logger.error(message)


if __name__ == "__main__":
    main()
