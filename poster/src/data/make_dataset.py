from src.config import get_logger
from src.data.make_train_test_datasets import make_train_and_test_datasets
from src.data.annotate import annotate
from src.data.filter import filter


def main() -> None:
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = get_logger()
    logger.info("Making dataset")

    annotate()
    filter()
    make_train_and_test_datasets()


if __name__ == "__main__":
    main()
