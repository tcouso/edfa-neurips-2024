from src.config import get_logger
from src.metrics.param_iter.compute_param_iter_score_matrices import compute_param_iter_scores
from src.metrics.param_iter.compute_param_iter_classification_scores import compute_param_iter_classification_scores

def main():
    logger = get_logger()

    logger.info("Computing param iteration metrics for all")
    compute_param_iter_scores("all")
    compute_param_iter_classification_scores("all")

    logger.info("Done")

if __name__ == "__main__":
    main()
