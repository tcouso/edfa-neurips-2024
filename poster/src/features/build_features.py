from src.config import get_logger
from src.features.run_signal_denoising import run_signal_denoising
from src.features.run_signal_slicing import run_signal_slicing


def main():
    logger = get_logger()

    message = "Starting features building"
    logger.info(message)

    message = "Starting training dataset denoising"
    logger.info(message)
    run_signal_denoising()

    message = "Starting training dataset slicing"
    logger.info(message)
    run_signal_slicing()

if __name__ == "__main__":
    main()
