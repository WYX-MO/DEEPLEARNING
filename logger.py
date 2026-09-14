import logging
import os

# 日志固定落在本文件旁边，不受启动时 cwd 影响
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.log")
FMT = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")


class OnlyInfoWarn(logging.Filter):
    """控制台只要 INFO 和 WARNING，其余级别一律挡掉。"""

    def filter(self, record):
        return record.levelno in (logging.INFO, logging.WARNING)


def get_logger(name="train",path=LOG_PATH):
    """拿到配置好的 logger；重复调用返回同一个，不会把 handler 挂两遍。"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    # 文件：什么都留，带时间戳
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(FMT)

    # 控制台：先放 INFO 及以上通过，再交给 filter 精确筛成 INFO/WARNING
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.addFilter(OnlyInfoWarn())
    ch.setFormatter(FMT)

    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


if __name__ == "__main__":
    logger = get_logger()
    logger.debug("DEBUG —— 只在文件里")
    logger.info("INFO  —— 文件 + 控制台")
    logger.warning("WARN  —— 文件 + 控制台")
    logger.error("ERROR —— 只在文件里")
    logger.critical("CRIT  —— 只在文件里")
