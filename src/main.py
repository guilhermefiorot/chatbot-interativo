import sys
import os
import logging
from dotenv import load_dotenv


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


load_dotenv()


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for the application."""
    try:
        from src.ui.app import main as run_app
        logger.info("Starting Adaptive Learning Chatbot application")
        run_app()
    except Exception as e:
        logger.error(f"Error starting application: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
