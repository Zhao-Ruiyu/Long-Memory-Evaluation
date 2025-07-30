import regex
import json
import string
import unicodedata
from typing import List
import numpy as np
from collections import Counter
import os
from bert_score import score
from nltk.stem import PorterStemmer
ps = PorterStemmer()

LENGTH_THRESHOLD = 5

class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        """
        对文本进行分词处理
        
        Args:
            text (str): 需要分词的文本
            uncased (bool, optional): 是否将结果转为小写。默认为False
        
        Returns:
            list: 分词后的token列表
        """
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens


def check_answer(example, tokenizer) -> List[bool]:
    """
    检查所有文档中是否包含答案
    
    Args:
        example (dict): 包含答案和上下文的样本字典
        tokenizer: 用于文本标记化的分词器
    
    Returns:
        List[bool]: 每个文档是否包含答案的布尔值列表
    """
    answers = example['answers']
    ctxs = example['ctxs']

    hits = []

    for _, doc in enumerate(ctxs):
        text = doc['text']

        if text is None:  # cannot find the document for some reason
            hits.append(False)
            continue

        hits.append(has_answer(answers, text, tokenizer))

    return hits


def has_answer(answers, text, tokenizer=SimpleTokenizer()) -> bool:
    """
    检查文档中是否包含答案字符串
    
    Args:
        answers (list): 可能的答案列表
        text (str): 要检查的文档文本
        tokenizer (SimpleTokenizer, optional): 用于分词的标记器。默认为SimpleTokenizer()
    
    Returns:
        bool: 如果文档包含任何答案则返回True，否则返回False
    """
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)

    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncased=True)
        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    return False


def _normalize(text):
    """
    对文本进行Unicode标准化处理
    
    Args:
        text (str): 需要标准化的文本
    
    Returns:
        str: 标准化后的文本
    """
    return unicodedata.normalize('NFD', text)


def normalize_answer(s):
    """
    规范化答案字符串，包括移除标点符号、冠词、多余空格并转为小写
    
    Args:
        s (str): 要规范化的答案字符串
    
    Returns:
        str: 规范化处理后的字符串
    """
    s = s.replace(',', "")
    def remove_articles(text):
        # return regex.sub(r'\b(a|an|the)\b', ' ', text)
        return regex.sub(r'\b(a|an|the|and)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    """
    计算预测与真实答案的精确匹配得分
    
    Args:
        prediction (str): 预测答案
        ground_truth (str): 真实答案
    
    Returns:
        bool: 如果规范化后的预测答案与真实答案的词集合相同则返回True，否则返回False
    """
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    # print('# EM #', prediction, ' | ', ground_truth, ' #', set(prediction.split()) == set(ground_truth.split()))
    # return normalize_answer(prediction) == normalize_answer(ground_truth)
    return set(prediction.split()) == set(ground_truth.split())
    
# def bert_score(prediction, ground_truths):
#     prediction = normalize_answer(prediction)
#     values = []
#     for ground_truth in ground_truths:
#         ground_truth = normalize_answer(ground_truth)
#         P, R, F1 = score([prediction], [ground_truth], lang='en', verbose=False, rescale_with_baseline=True)
#         values.append(R[0].item())
#     print('# BERT # ', normalize_answer(prediction), ' | ', normalize_answer(ground_truth), ' #', P, R, F1)
#     return max(0, max(values))


def bert_score(prediction, ground_truth):
    """
    使用BERT计算预测答案和真实答案之间的相似度得分
    
    Args:
        prediction (str): 预测答案
        ground_truth (str): 真实答案
    
    Returns:
        float: BERT评分的F1值，范围为[0, 1]
    """
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    P, R, F1 = score([prediction], [ground_truth], lang='en', verbose=False, rescale_with_baseline=True)
    # print('# BERT # ', normalize_answer(prediction), ' | ', normalize_answer(ground_truth), ' #', P, R, F1)
    return max(0, F1[0].item())


def ems(prediction, ground_truths):
    """
    计算预测答案与多个参考答案之间的最大精确匹配得分
    
    Args:
        prediction (str): 预测答案
        ground_truths (list): 参考答案列表
    
    Returns:
        float: 与任一参考答案匹配的最高精确匹配得分
    """
    return max([exact_match_score(prediction, gt) for gt in ground_truths])


def f1_score(prediction, ground_truth):
    """
    计算预测答案与参考答案之间的F1得分
    
    Args:
        prediction (str): 预测答案
        ground_truth (str): 参考答案
    
    Returns:
        float: F1得分，范围为[0, 1]，基于词干化后的词汇重叠计算
    """
    prediction_tokens = [ps.stem(w) for w in normalize_answer(prediction).split()]
    ground_truth_tokens = [ps.stem(w) for w in normalize_answer(ground_truth).split()]
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    # print('# F1 #', prediction, ' | ', ground_truth, ' #', precision, recall, f1)
    # return recall
    return f1


def f1(prediction, ground_truth):
    """
    计算多个预测答案与多个参考答案之间的平均F1得分
    
    Args:
        prediction (str): 逗号分隔的多个预测答案
        ground_truth (str): 逗号分隔的多个参考答案
    
    Returns:
        float: 平均F1得分，对每个参考答案，先找出与之最匹配的预测答案，再对所有参考答案取平均
    """
    predictions = [p.strip() for p in prediction.split(',')]
    ground_truths = [g.strip() for g in ground_truth.split(',')]
    # print('# F1 [multi-answer]#', predictions, ' | ', ground_truths, ' #', np.mean([max([f1_score(prediction, gt) for prediction in predictions]) for gt in ground_truths]))
    return np.mean([max([f1_score(prediction, gt) for prediction in predictions]) for gt in ground_truths])


def rougel_score(prediction, ground_truth):
    """
    计算预测答案与参考答案之间的ROUGE-L得分
    
    Args:
        prediction (str): 预测答案
        ground_truth (str): 参考答案
    
    Returns:
        float: ROUGE-1 F1得分，基于词干化后的规范化文本
    """
    from rouge import Rouge
    rouge = Rouge()
    prediction = ' '.join([ps.stem(w) for w in normalize_answer(prediction).split()])
    ground_truth = ' '.join([ps.stem(w) for w in normalize_answer(ground_truth).split()])
    # no normalization
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-1"]["f"]


def rl(prediction, ground_truths):
    """
    计算预测答案与多个参考答案之间的最大ROUGE-L得分
    
    Args:
        prediction (str): 预测答案
        ground_truths (list): 参考答案列表
    
    Returns:
        float: 与任一参考答案匹配的最高ROUGE-L得分
    """
    return max([rougel_score(prediction, gt) for gt in ground_truths])


## file-level evaluation ... ### 
def eval_recall(infile):
    """
    评估输入文件中问题-答案对的召回率
    
    Args:
        infile (str): 输入文件路径，包含问题和生成的答案
    
    Returns:
        tuple: (召回率, 平均答案长度)，召回率表示答案中包含正确答案的比例
    """
    tokenizer = SimpleTokenizer()
    lines = open(infile, 'r').readlines()[1:]

    has_answer_count = 0
    answer_lengths = []
    for line in lines:
        line = json.loads(line)
        answer = line['answer']
        output = ' || '.join(line['output'])

        if has_answer(answer, output, tokenizer):
            has_answer_count += 1

        answer_lengths.append(len(output.split()))

    recall = round(has_answer_count/len(lines), 4)
    lens = round(np.mean(answer_lengths), 4)

    return recall, lens


def eval_question_answering(qas, eval_key='prediction', metric='f1'):
    """
    评估问答系统的性能
    
    Args:
        qas (list): 包含问题和答案的字典列表
        eval_key (str, optional): 评估使用的预测键。默认为'prediction'
        metric (str, optional): 评估使用的度量标准。默认为'f1'
    
    Returns:
        tuple: (评分列表, 平均长度, 召回准确率列表)，其中评分根据不同问题类别使用不同的度量方法
    """
    all_ems = []
    all_recall = []
    exact_match_count = 0
    f1_count = 0
    answer_lengths = []
    for i, line in enumerate(qas):
        # line = json.loads(line)
        # 处理不同类别的答案字段
        if line['category'] == 5:
            # 类别5使用 adversarial_answer 字段
            answer = line.get('adversarial_answer', line.get('answer', ''))
        else:
            # 其他类别使用 answer 字段
            if type(line[eval_key]) == list:
                answer = line['answer']
            else:
                answer = str(line['answer'])
        
        if line['category'] == 3:
            answer = answer.split(';')[0].strip()
        
        output = line[eval_key]
        
        # single-hop, temporal, open-domain eval without splitting for sub-answers 
        if line['category'] in [2, 3, 4]:
            all_ems.append(f1_score(output, answer))
        
        # multi-hop eval by splitting entire phrase into sub-answers and computing partial F1 for each
        elif line['category'] in [1]:
            all_ems.append(f1(output, answer))

        # adversarial eval --> check for selection of correct option
        elif line['category'] in [5]:
            if 'no information available' in output.lower() or 'not mentioned' in output.lower():
                all_ems.append(1)
            else:
                all_ems.append(0)
        else:
            print(line)
            raise ValueError
        
        assert i+1 == len(all_ems), all_ems

        if eval_key + '_context' in line and len(line['evidence']) > 0:
            # recall_acc for dialog
            if line[eval_key + '_context'][0].startswith('S'):
                sessions = [e[1:] for e in line[eval_key + '_context']]
                recall_acc = float(sum([ev.split(':')[0][1:] in sessions for ev in line["evidence"]]))/len(line['evidence'])
            else:
                recall_acc = float(sum([ev in line[eval_key + '_context'] for ev in line["evidence"]]))/len(line['evidence'])
            all_recall.append(recall_acc)
        else:
            all_recall.append(1)

    print("{} QA samples evaluated; {} accuracy values".format(len(qas), len(all_ems)))
    lens = 0.0
    return all_ems, lens, all_recall


def eval_fact_checking(infile):
    """
    评估事实检查系统的性能
    
    Args:
        infile (str): 包含事实检查数据的输入文件路径
    
    Returns:
        tuple: (准确率, 平均答案长度)，准确率表示正确预测的比例
    """
    tokenizer = SimpleTokenizer()
    lines = open(infile, 'r').readlines()[1:]

    exact_match_count = 0
    answer_lengths = []
    for line in lines:
        line = json.loads(line)
        answer = line['answer']
        output = line['output'][0]

        if answer == ["refutes"]:
            answer = ["refutes", "no", "false"]
        if answer == ["supports"]:
            answer = ["supports", "yes", "true"]

        if has_answer(answer, output, tokenizer):
            exact_match_count += 1
        
        answer_lengths.append(len(output.split()))

    em = round(exact_match_count/len(lines), 4)
    lens = round(np.mean(answer_lengths), 4)

    return em, lens


def eval_dialogue_system(infile):
    """
    评估对话系统的性能
    
    Args:
        infile (str): 包含对话数据的输入文件路径
    
    Returns:
        tuple: (F1得分, ROUGE-L得分, 平均答案长度)，F1和ROUGE-L表示系统生成回复与参考回复的相似度
    """
    lines = open(infile, 'r').readlines()[1:]

    f1_scores = []
    rl_scores = []
    answer_lengths = []
    for line in lines:
        line = json.loads(line)
        answer = line['answer']
        output = line['output'][0]

        f1_scores.append(f1(output, answer))
        rl_scores.append(rl(output, answer))
        answer_lengths.append(len(output.split()))

    F1 = round(np.mean(f1_scores), 4)
    RL = round(np.mean(rl_scores), 4)
    lens = round(np.mean(answer_lengths), 4)

    return F1, RL, lens

