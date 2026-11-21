from src.config import get_logger
from src.inference.change_detection.run_change_detection import run_change_detection

def main():
    logger = get_logger()

    message = "Starting change detection"
    logger.info(message)
    run_change_detection()

if __name__ == "__main__":
    main()
