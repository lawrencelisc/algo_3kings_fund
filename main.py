import logging
import sys
from pathlib import Path

_HYPE_BOT = Path(__file__).resolve().parent / "hype_bot"
_hype_path = str(_HYPE_BOT)
if _hype_path not in sys.path:
    sys.path.insert(0, _hype_path)

from config import CFG
from master_bot import MasterBot


def setup_logging():
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    handlers = [logging.StreamHandler(sys.stdout),
                logging.FileHandler(CFG.LOG_FILE, encoding="utf-8")]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def main():
    setup_logging()
    bot = MasterBot()
    bot.run()


if __name__ == "__main__":
    main()