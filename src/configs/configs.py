import os
from dotenv import load_dotenv

load_dotenv(".env")

class Configs:
    def __init__(self):
        # 输出目录配置
        self.OUT_DIR = os.getenv("OUT_DIR", "./outputs")
        self.EMB_DIR = os.getenv("EMB_DIR", "./outputs")
        
        # 数据文件路径
        self.DATA_FILE_PATH = os.getenv("DATA_FILE_PATH", "./data/locomo10.json")
        
        # 输出文件名配置
        self.QA_OUTPUT_FILE = os.getenv("QA_OUTPUT_FILE", "locomo10_qa.json")
        self.OBS_OUTPUT_FILE = os.getenv("OBS_OUTPUT_FILE", "locomo10_observation.json")
        self.SESS_SUMM_OUTPUT_FILE = os.getenv("SESS_SUMM_OUTPUT_FILE", "locomo10_session_summary.json")
        
        # 提示词目录
        self.PROMPT_DIR = os.getenv("PROMPT_DIR", "./prompt_examples")
        
        # API密钥配置
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        self.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
        self.HF_TOKEN = os.getenv("HF_TOKEN")
        self.DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

        # 火山引擎模型配置
        # ARK API密钥配置
        self.ARK_API_KEY = os.getenv("ARK_API_KEY")
        
        # ARK模型ID配置
        self.ARK_DEEPSEEK_R1_MODEL = os.getenv("ARK_DEEPSEEK_R1_MODEL")
        self.ARK_DEEPSEEK_V3_MODEL = os.getenv("ARK_DEEPSEEK_V3_MODEL")

        # 嵌入模型配置
        self.EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL")

# 创建全局配置实例
configs = Configs()
        