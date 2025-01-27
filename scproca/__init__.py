import logging
from scproca.settings import settings

package_name = "scProca"
__version__ = "0.1"

settings.verbosity = logging.INFO

logger = logging.getLogger("scProca")
logger.propagate = False
logger.info(f"Last run with {package_name} version: {__version__}")

