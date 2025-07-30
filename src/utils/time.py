import datetime

def get_timestamp():
    """
    获取当前时间戳，格式为 [YYYY-MM-DD-HH:MM:SS]
    
    Returns:
        str: 格式化的时间戳字符串
    """
    return datetime.datetime.now().strftime("[%Y-%m-%d-%H:%M:%S]")

def get_duration(start_time, end_time=None):
    """
    计算时间间隔
    
    Args:
        start_time (datetime): 开始时间
        end_time (datetime, optional): 结束时间，如果为None则使用当前时间
    
    Returns:
        float: 时间间隔（秒）
    """
    if end_time is None:
        end_time = datetime.datetime.now()
    return (end_time - start_time).total_seconds()

def format_duration(seconds):
    """
    格式化时间间隔显示
    
    Args:
        seconds (float): 秒数
    
    Returns:
        str: 格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.2f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}小时"
