from src.config import get_logger
from src.features.run_signal_denoising import run_signal_denoising


def main():
    logger = get_logger()

    try:
        message = "Starting signal denoising"
        logger.info(message)

        logger.info("Denoising ESN signal")
        run_signal_denoising()

        message = "Signal denoising completed successfully"
        logger.info(message)

    except Exception as e:
        message = f"Signal denoising failed: {e}"
        logger.error(message)


if __name__ == "__main__":
    main()
