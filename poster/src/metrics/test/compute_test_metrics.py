from src.config import get_logger
from src.metrics.test.compute_test_classification_scores import compute_test_classification_scores

def main():
    logger = get_logger()
    logger.info("Computing test metrics")
    compute_test_classification_scores()

    logger.info("Done")


if __name__ == "__main__":
    main()
