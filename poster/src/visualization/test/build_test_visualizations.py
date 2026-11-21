from src.config import get_logger
from src.visualization.test.plot_test_classification_scores import plot_test_classification_scores, plot_test_classification_scores_by_event_type


def main():
    logger = get_logger()
    logger.info("Building test visualizations")
    plot_test_classification_scores()
    plot_test_classification_scores_by_event_type()


    logger.info("Done")


if __name__ == "__main__":
    main()
