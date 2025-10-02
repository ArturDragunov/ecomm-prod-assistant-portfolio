# logger/__init__.py
from .custom_logger import CustomLogger
# Create a single shared logger instance
# We don't need to create a new logger instance for each module
# We can import the logger instance instead
GLOBAL_LOGGER = CustomLogger().get_logger("prod_assistant")