import sys
from pathlib import Path
# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

import random
import os, json
from tqdm import tqdm
import time
import datetime
from src.api.api import ark_api_model, set_ark_key
from src.utils.time import get_timestamp, get_duration, format_duration
from src.utils.log import log_info, log_warning, log_error

# 不同ARK模型的最大输入长度配置（单位：tokens）
MAX_LENGTH={'DeepSeekR1-Ark': 64000, 'DeepSeekV3-Ark': 128000}
# 每个问答的token预算
PER_QA_TOKEN_BUDGET = 50

# 标准问答提示模板
QA_PROMPT = """
Based on the above context, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {} Short answer:
"""

# 类别5问题的问答提示模板
QA_PROMPT_CAT_5 = """
Based on the above context, answer the following question.

Question: {} Short answer:
"""

# 批量问答提示模板
QA_PROMPT_BATCH = """
Based on the above conversations, write short answers for each of the following questions in a few words. Write the answers in the form of a json dictionary where each entry contains the string format of question number as 'key' and the short answer as value. Use single-quote characters for named entities. Answer with exact words from the conversations whenever possible.

"""

# 对话开始提示模板
CONV_START_PROMPT = "Below is a conversation between two people: {} and {}. The conversation takes place over multiple days and the date of each conversation is wriiten at the beginning of the conversation.\n\n"

def get_input_context(conversation, num_question_tokens, encoding, args):
    """
    获取输入上下文，确保不超过模型的最大长度限制
    
    Args:
        conversation (dict): 对话数据，包含多个会话
        num_question_tokens (int): 问题文本的token数量
        encoding: 编码器对象（当前未使用）
        args: 参数对象，包含模型配置信息
    
    Returns:
        str: 处理后的对话上下文文本，确保在模型长度限制内
    """
    query_conv = ''
    min_session = -1
    stop = False
    
    # 获取所有会话编号
    session_nums = [int(k.split('_')[-1]) for k in conversation.keys() if 'session' in k and 'date_time' not in k]
    
    for i in range(min(session_nums), max(session_nums) + 1):
        if 'session_%s' % i in conversation:
            query_conv += "\n\n"
            for dialog in conversation['session_%s' % i][::-1]:
                turn = ''
                turn = dialog['speaker'] + ' said, \"' + dialog['text'] + '\"' + '\n'
                if "blip_caption" in dialog:
                    turn += ' and shared %s.' % dialog["blip_caption"]
                turn += '\n'
            
                # 简化token计算，使用固定值
                num_tokens = 100  # 简化版本，使用固定值
                if (num_tokens + len(query_conv.encode()) + num_question_tokens) < 8000:  # 使用固定长度限制
                    query_conv = turn + query_conv
                else:
                    min_session = i
                    stop = True
                    break
            query_conv = 'DATE: ' + conversation['session_%s_date_time' % i] + '\n' + 'CONVERSATION:\n' + query_conv
        if stop:
            break
    
    return query_conv

def get_cat_5_answer(answer, cat_5_answer):
    """
    处理类别5问题的答案，解析模型输出的选项选择
    
    Args:
        answer (str): 模型的原始答案文本
        cat_5_answer (dict): 包含选项a和b的答案字典
    
    Returns:
        str: 根据模型选择解析出的正确答案
    """
    answer = answer.lower().strip()
    if '(a)' in answer or 'a)' in answer:
        return cat_5_answer['a']
    elif '(b)' in answer or 'b)' in answer:
        return cat_5_answer['b']
    else:
        return answer

def process_ouput(answer):
    """
    处理模型输出的答案格式，尝试解析为结构化数据
    
    Args:
        answer (str): 模型的原始输出文本
    
    Returns:
        dict: 解析后的答案字典，键为问题编号，值为答案内容
    """
    try:
        # 尝试解析JSON格式
        return json.loads(answer)
    except:
        # 如果不是JSON格式，尝试其他解析方式
        lines = answer.strip().split('\n')
        result = {}
        for i, line in enumerate(lines):
            if ':' in line:
                key, value = line.split(':', 1)
                result[key.strip()] = value.strip()
            else:
                result[str(i)] = line.strip()
        return result

def get_ark_answers(in_data, out_data, prediction_key, args):
    """
    使用ARK模型批量生成问答答案的主函数
    
    Args:
        in_data (dict): 输入数据，包含对话和问答对
        out_data (dict): 输出数据，用于存储预测结果
        prediction_key (str): 预测结果的键名
        args: 参数对象，包含模型配置、批处理大小等
    
    Returns:
        dict: 更新后的输出数据，包含模型预测的答案
    
    Raises:
        AssertionError: 当输入和输出数据长度不匹配时
        NotImplementedError: 当尝试使用未实现的RAG模式时
    """
    assert len(in_data['qa']) == len(out_data['qa']), (len(in_data['qa']), len(out_data['qa']))

    # 设置ARK API密钥
    set_ark_key()

    # 根据参数确定使用的模型名称
    if 'DeepSeekR1-Ark' in args.model:
        model_name = 'DeepSeekR1-Ark'
    elif 'DeepSeekV3-Ark' in args.model:
        model_name = 'DeepSeekV3-Ark'
    else:
        model_name = 'DeepSeekV3-Ark'  # 默认使用DeepSeekV3-Ark

    # 构建对话开始提示，获取说话者名称
    session_nums = [int(k.split('_')[-1]) for k in in_data['conversation'].keys() if 'session' in k and 'date_time' not in k]
    if session_nums:
        first_session = min(session_nums)
        if f'session_{first_session}' in in_data['conversation']:
            speakers_names = list(set([d['speaker'] for d in in_data['conversation'][f'session_{first_session}']]))
            if len(speakers_names) >= 2:
                start_prompt = CONV_START_PROMPT.format(speakers_names[0], speakers_names[1])
            else:
                start_prompt = CONV_START_PROMPT.format("Person A", "Person B")
        else:
            start_prompt = CONV_START_PROMPT.format("Person A", "Person B")
    else:
        start_prompt = CONV_START_PROMPT.format("Person A", "Person B")
    start_tokens = 100

    # RAG模式检查（当前未实现）
    if args.use_rag:
        raise NotImplementedError("RAG mode not implemented for ARK models yet")
    else:
        context_database, query_vectors = None, None

    log_info("开始批量处理问答对，共 %d 个问题" % len(in_data['qa']))
    # 批量处理问答对
    for batch_start_idx in tqdm(range(0, len(in_data['qa']), args.batch_size), desc='Generating answers'):

        questions = []
        include_idxs = []
        cat_5_idxs = []
        cat_5_answers = []
        
        # 收集当前批次的问答对
        for i in range(batch_start_idx, batch_start_idx + args.batch_size):

            if i>=len(in_data['qa']):
                break

            qa = in_data['qa'][i]
            
            # 检查是否需要生成答案
            if prediction_key not in out_data['qa'][i]:
                include_idxs.append(i)
            else:
                continue

            # 根据问题类别处理不同类型的问题
            if qa['category'] == 2:
                # 类别2：日期相关问题
                questions.append(qa['question'] + ' Use DATE of CONVERSATION to answer with an approximate date.')
            elif qa['category'] == 5:
                # 类别5：选择题，随机排列选项顺序
                question = qa['question'] + " Select the correct answer: (a) {} (b) {}. "
                adversarial_answer = qa.get('adversarial_answer', qa.get('answer', 'Not mentioned in the conversation'))
                if random.random() < 0.5:
                    question = question.format('Not mentioned in the conversation', adversarial_answer)
                    answer = {'a': 'Not mentioned in the conversation', 'b': adversarial_answer}
                else:
                    question = question.format(adversarial_answer, 'Not mentioned in the conversation')
                    answer = {'b': 'Not mentioned in the conversation', 'a': adversarial_answer}

                cat_5_idxs.append(len(questions))
                questions.append(question)
                cat_5_answers.append(answer)
            else:
                # 其他类别：标准问题
                questions.append(qa['question'])

        if questions == []:
            continue

        # 构建查询上下文
        context_ids = None
        if args.use_rag:
            raise NotImplementedError("RAG mode not implemented for ARK models yet")
        else:
            question_prompt = QA_PROMPT_BATCH + "\n".join(["%s: %s" % (k, q) for k, q in enumerate(questions)])
            num_question_tokens = 100
            query_conv = get_input_context(in_data['conversation'], num_question_tokens + start_tokens, None, args)
            query_conv = start_prompt + query_conv

        # 单批次处理
        if args.batch_size == 1:
            # 构建单个问题的查询
            query = query_conv + '\n\n' + QA_PROMPT.format(questions[0]) if len(cat_5_idxs) == 0 else query_conv + '\n\n' + QA_PROMPT_CAT_5.format(questions[0])
            
            # 调用ARK模型生成答案
            current_batch = batch_start_idx // args.batch_size + 1
            total_batches = (len(in_data['qa']) + args.batch_size - 1) // args.batch_size
            log_info("当前 %d/%d 批次调用ARK模型生成答案" % (current_batch, total_batches))
            api_start_time = datetime.datetime.now()
            answer = ark_api_model(query, num_gen=1, num_tokens_request=PER_QA_TOKEN_BUDGET, 
                           model=model_name, temperature=0, wait_time=2)
            api_duration = get_duration(api_start_time)
            log_info("ARK API调用完成，耗时 %s" % format_duration(api_duration))
            
            # 处理类别5问题的答案
            if len(cat_5_idxs) > 0:
                answer = get_cat_5_answer(answer, cat_5_answers[0])

            # 保存预测结果
            out_data['qa'][include_idxs[0]][prediction_key] = answer.strip()
            if args.use_rag:
                out_data['qa'][include_idxs[0]][prediction_key + '_context'] = context_ids

        # 批量处理
        else:
            query = query_conv + '\n' + question_prompt
            
            # 重试机制，最多尝试5次
            trials = 0
            while trials < 5:
                try:
                    trials += 1
                    current_batch = batch_start_idx // args.batch_size + 1
                    total_batches = (len(in_data['qa']) + args.batch_size - 1) // args.batch_size
                    log_info("当前 %d/%d 批次调用ARK模型生成答案 (第 %d 次尝试)" % (current_batch, total_batches, trials))
                    api_start_time = datetime.datetime.now()
                    answer = ark_api_model(query, num_gen=1, num_tokens_request=args.batch_size*PER_QA_TOKEN_BUDGET, 
                                   model=model_name, temperature=0, wait_time=2)
                    api_duration = get_duration(api_start_time)
                    log_info("ARK API调用完成，耗时 %s" % format_duration(api_duration))
                    answer = answer.replace('\\"', "'").replace('json','').replace('`','').strip()
                    answers = process_ouput(answer.strip())
                    break
                except Exception as e:
                    log_error(f'Error at trial {trials}/5: {e}')
                    if trials == 5:
                        raise e
            
            # 处理批量答案并保存结果
            for k, idx in enumerate(include_idxs):
                try:
                    if k in cat_5_idxs:
                        # 处理类别5问题的答案
                        predicted_answer = get_cat_5_answer(answers[str(k)], cat_5_answers[cat_5_idxs.index(k)])
                        out_data['qa'][idx][prediction_key] = predicted_answer
                    else:
                        try:
                            # 处理标准问题的答案
                            out_data['qa'][idx][prediction_key] = str(answers[str(k)]).replace('(a)', '').replace('(b)', '').strip()
                        except:
                            # 备用处理方式
                            out_data['qa'][idx][prediction_key] = ', '.join([str(n) for n in list(answers[str(k)].values())])
                except:
                    # 异常处理：尝试其他解析方式
                    try:
                        answers = json.loads(answer.strip())
                        if k in cat_5_idxs:
                            predicted_answer = get_cat_5_answer(answers[k], cat_5_answers[cat_5_idxs.index(k)])
                            out_data['qa'][idx][prediction_key] = predicted_answer
                        else:
                            out_data['qa'][idx][prediction_key] = answers[k].replace('(a)', '').replace('(b)', '').strip()
                    except:
                        # 最后的备用处理方式
                        if k in cat_5_idxs:
                            predicted_answer = get_cat_5_answer(answer.strip(), cat_5_answers[cat_5_idxs.index(k)])
                            out_data['qa'][idx][prediction_key] = predicted_answer
                        else:
                            out_data['qa'][idx][prediction_key] = json.loads(answer.strip().replace('(a)', '').replace('(b)', '').split('\n')[k])[0]

    return out_data 