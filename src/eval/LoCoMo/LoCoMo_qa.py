import sys
from pathlib import Path
import datetime

import os, json
from tqdm import tqdm
import argparse
from src.api.api import set_ark_key
from src.eval.LoCoMo.evaluation import eval_question_answering
from src.eval.LoCoMo.evaluation_stats import analyze_aggr_acc
from src.eval.LoCoMo.utils.ark_utils import get_ark_answers
from src.utils.time import get_duration, format_duration
from src.utils.log import init_logger, log_info

import numpy as np

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', type=str, default="data/eval/LoCoMo/locomo10.json", 
                       help="数据文件路径, 默认在data/eval/LoCoMo/locomo10.json")   
    parser.add_argument('--model', type=str, default="DeepSeekV3-Ark", 
                       choices=["DeepSeekV3-Ark", "DeepSeekR1-Ark"],
                       help="模型选择: DeepSeekV3-Ark (默认) 或 DeepSeekR1-Ark") 
    parser.add_argument('--temperature', type=float, default=0.0,
                       help="模型温度参数")
    parser.add_argument('--batch-size', default=1, type=int,
                       help="批量处理大小, 默认1")        
    parser.add_argument('--use-4bit', action="store_true",
                       help="是否使用4bit量化, 默认不使用")
    parser.add_argument('--emb-dir', type=str, default="",
                       help="嵌入目录, 默认不使用") 
    parser.add_argument('--use-rag', action="store_true",
                       help="是否使用RAG模式, 默认不使用")   
    parser.add_argument('--rag-mode', type=str, default="",
                       help="RAG模式, 默认不使用")          
    parser.add_argument('--retriever', type=str, default="contriever",
                       help="检索器, 默认使用contriever")
    parser.add_argument('--top-k', type=int, default=5,
                       help="RAG模式Top-K结果数量, 默认5")                   
    parser.add_argument('--out-file',
                       default="data/.output/LoCoMo", type=str,
                       help="输出目录路径, 默认在data/.output/LoCoMo")

    args = parser.parse_args()
    return args


def main():

    # get arguments
    args = parse_args()

    # 创建时间戳文件夹
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = f"{args.out_file}/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置输出文件路径
    args.out_file = f"{output_dir}/locomo10_qa_output.json"
    # 设置默认日志文件路径
    args.log_file = f"{output_dir}/locomo10_qa_log.log"

    # 初始化日志记录器
    init_logger(args.log_file)

    log_info("******************  Evaluating Model %s ***************" % args.model)

    # set ark API key
    set_ark_key()
    log_info("ARK API密钥已设置")

    # load conversations
    log_info("开始加载数据文件: %s" % args.data_file)
    samples = json.load(open(args.data_file))
    log_info("数据加载完成，共 %d 个样本" % len(samples))
    
    prediction_key = "%s_prediction" % args.model if not args.use_rag else "%s_%s_top_%s_prediction" % (args.model, args.rag_mode, args.top_k)
    model_key = "%s" % args.model if not args.use_rag else "%s_%s_top_%s" % (args.model, args.rag_mode, args.top_k)
    
    # 初始化输出数据
    log_info("创建新的输出文件")
    out_samples = {}


    log_info("开始处理样本...")
    for i, data in enumerate(samples):
        log_info("处理样本 %d/%d: %s" % (i+1, len(samples), data['sample_id']))
        
        out_data = {'sample_id': data['sample_id']}
        if data['sample_id'] in out_samples:
            out_data['qa'] = out_samples[data['sample_id']]['qa'].copy()
        else:
            out_data['qa'] = data['qa'].copy()

        # get answers for each sample
        sample_start_time = datetime.datetime.now()
        answers = get_ark_answers(data, out_data, prediction_key, args)
        sample_duration = get_duration(sample_start_time)
        log_info("样本 %s 处理完成，耗时 %s" % (data['sample_id'], format_duration(sample_duration)))

        # evaluate individual QA samples and save the score
        exact_matches, lengths, recall = eval_question_answering(answers['qa'], prediction_key)
        for j in range(0, len(answers['qa'])):
            answers['qa'][j][model_key + '_f1'] = round(exact_matches[j], 3)
            if args.use_rag and len(recall) > 0:
                answers['qa'][j][model_key + '_recall'] = round(recall[j], 3)

        out_samples[data['sample_id']] = answers


    log_info("开始保存结果到文件: %s" % args.out_file)
    with open(args.out_file, 'w') as f:
        json.dump(list(out_samples.values()), f, indent=2)
    log_info("结果保存完成")

    log_info("开始生成统计报告...")
    analyze_aggr_acc(args.data_file, args.out_file, args.out_file.replace('.json', '_stats.json'),
                model_key, model_key + '_f1', rag=args.use_rag)
    log_info("统计报告生成完成")
    log_info("所有任务完成！")
    # encoder=tiktoken.encoding_for_model(args.model))


main()