from src.config import get_logger
from src.visualization.param_iter.plot_param_iter_scores import plot_clasification_scores_over_parameters
from src.visualization.param_iter.plot_detailed_classification_scores import plot_detailed_param_iter_scores

def main():
    logger = get_logger()

    logger.info("Building param iter visualiazations for all")
    plot_clasification_scores_over_parameters("all")
    plot_detailed_param_iter_scores("all")

    logger.info("Done")

if __name__ == "__main__":
    main()
