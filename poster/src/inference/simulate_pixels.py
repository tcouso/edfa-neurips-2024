from src.config import get_logger
from src.inference.simulation.run_simulations import run_simulations


def main():
    logger = get_logger()

    message = "Starting signal simulations"
    logger.info(message)
    run_simulations()

if __name__ == "__main__":
    main()
