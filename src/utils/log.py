import logging
import os
from datetime import datetime
from pathlib import Path

class Logger:
    """
    统一的日志记录器
    """
    def __init__(self, log_file=None, level=logging.INFO):
        """
        初始化日志记录器
        
        Args:
            log_file (str): 日志文件路径，如果为None则使用控制台输出
            level: 日志级别
        """
        self.logger = logging.getLogger('memeval')
        self.logger.setLevel(level)
        
        # 清除现有的处理器
        self.logger.handlers.clear()
        
        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # 添加控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 如果指定了日志文件，添加文件处理器
        if log_file:
            # 确保日志目录存在
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message):
        """记录信息日志"""
        self.logger.info(message)
    
    def warning(self, message):
        """记录警告日志"""
        self.logger.warning(message)
    
    def error(self, message):
        """记录错误日志"""
        self.logger.error(message)
    
    def debug(self, message):
        """记录调试日志"""
        self.logger.debug(message)
    
    def critical(self, message):
        """记录严重错误日志"""
        self.logger.critical(message)

# 全局日志记录器实例
_global_logger = None

def init_logger(log_file=None, level=logging.INFO):
    """
    初始化全局日志记录器
    
    Args:
        log_file (str): 日志文件路径
        level: 日志级别
    """
    global _global_logger
    _global_logger = Logger(log_file, level)

def get_logger():
    """
    获取全局日志记录器
    
    Returns:
        Logger: 日志记录器实例
    """
    global _global_logger
    if _global_logger is None:
        # 如果没有初始化，创建一个默认的控制台日志记录器
        _global_logger = Logger()
    return _global_logger

def log_info(message):
    """记录信息日志"""
    get_logger().info(message)

def log_warning(message):
    """记录警告日志"""
    get_logger().warning(message)

def log_error(message):
    """记录错误日志"""
    get_logger().error(message)

def log_debug(message):
    """记录调试日志"""
    get_logger().debug(message)

def log_critical(message):
    """记录严重错误日志"""
    get_logger().critical(message) 