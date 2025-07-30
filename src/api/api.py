import time
from openai import OpenAI
from src.configs.configs import Configs

configs = Configs()

def set_ark_key():
    return configs.ARK_API_KEY

def set_openai_key():
    return configs.OPENAI_API_KEY

def set_google_key():
    return configs.GOOGLE_API_KEY

def set_anthropic_key():
    return configs.ANTHROPIC_API_KEY

def set_gemini_key():
    return configs.GEMINI_API_KEY

def set_hf_key():
    return configs.HF_TOKEN

def set_deepseek_key():
    return configs.DEEPSEEK_API_KEY

def ark_api_model(query, num_gen=1, num_tokens_request=1000, 
             model='DeepSeekR1-Ark', temperature=1.0, wait_time=1):
    """
    运行火山引擎API的函数
    
    Args:
        query: 查询内容
        num_gen: 生成数量
        num_tokens_request: 请求的最大token数
        model: 模型名称 ('DeepSeekR1-Ark' 或 'DeepSeekV3-Ark')，实际模型ID从环境变量读取
        temperature: 温度参数
        wait_time: 等待时间
    
    Returns:
        生成的文本内容
    """
    
    # 模型映射 - 将简洁名称映射到实际的模型ID
    model_mapping = {
        'DeepSeekR1-Ark': configs.ARK_DEEPSEEK_R1_MODEL,
        'DeepSeekV3-Ark': configs.ARK_DEEPSEEK_V3_MODEL,
    }
    
    # 获取实际的模型名称
    actual_model = model_mapping.get(model, model)
    
    api_key = configs.ARK_API_KEY
    if not api_key:
        raise ValueError("ARK_API_KEY not found. Please source env.sh before running.")
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://ark.cn-beijing.volces.com/api/v3"
    )
    
    completion = None
    while completion is None:
        try:
            completion = client.chat.completions.create(
                model=actual_model,
                messages=[
                    {"role": "user", "content": query}
                ],
                temperature=temperature,
                max_tokens=num_tokens_request,
                n=num_gen,
                stream=False
            )
        except Exception as e:
            print(f"ARK API error: {e}; waiting for {wait_time} seconds")
            time.sleep(wait_time)
            wait_time = wait_time * 2
            if wait_time > 60:  # 最大等待60秒
                print("Exiting after too many retries")
                raise e
    
    return completion.choices[0].message.content