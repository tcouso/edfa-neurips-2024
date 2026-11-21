from src.config import get_logger
from src.models.run_simulations import run_simulations
from src.models.predict_over_pixels import predict_over_pixels
from src.models.predict_over_polygons import predict_over_polygons


def main():
    try:
        logger = get_logger()

        message = "Starting signal simulations"
        logger.info(message)
        run_simulations()

        message = "Starting pixel predictions"
        logger.info(message)
        predict_over_pixels()

        message = "Starting polygon predictions"
        logger.info(message)
        predict_over_polygons()

        message = "Done"
        logger.info(message)

    except KeyboardInterrupt:
        message = "Execution interrupted by keyboard"
        logger.info(message)

    except Exception as e:
        message = f"Execution crashed: {e}"
        logger.error(message)


if __name__ == '__main__':
    main()
