# !/usr/bin/python
# -*- coding:utf-8 -*-
import time
import logging
from constant import LOG_DIR, LOG_LEVEL


class Log(object):

    def __init__(self, logger=None):
        # 创建一个logger
        self.logger = logging.getLogger(logger)
        self.logger.setLevel(logging.DEBUG)
        # 创建一个handler，用于写入日志文件
        self.log_time = time.strftime("%y%m%d-%H%M")
        self.log_name = LOG_DIR / "log.log"
        fh = logging.FileHandler(self.log_name, 'a', encoding='utf-8')
        fh.setLevel(LOG_LEVEL)

        # 再创建一个handler，用于输出到控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # 定义handler的输出格式
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S", )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # 给logger添加handler
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        # 关闭打开的文件
        fh.close()
        ch.close()

    def get_logger(self):
        return self.logger
