import json
import jsonlines
from collections import defaultdict
import argparse
import os

ans_dict = {}

def compile_answers(vid_path: str = 'videos', answer_type: str = 'answers_reformatted'):
    nums = os.listdir(vid_path)
    for num in nums:
        ans_path = f"./{vid_path}/{num}/{num}_{answer_type}.json"
        if os.path.exists(ans_path):
            with open(ans_path) as f:
                ans_file = json.load(f)
            for q_dict in ans_file:
                ans_dict[q_dict["uid"]] = q_dict["answer"]
    output_file = f'./{vid_path}_compiled_answers.json'
    with open(output_file, 'w') as f:
        json.dump(ans_dict, f)
    
    return output_file

def compute_accuracy(answer_file: str, video_meta_file: str):
    total_qa_num = 0
    right_num = 0
    category_right = defaultdict(int)
    category_total = defaultdict(int)
    category_acc = defaultdict(int)

    with open(answer_file) as f:
        model_answers = json.load(f)
    
    wrong = []

    with jsonlines.open(video_meta_file) as reader:
        video_meta = list(reader)
        for meta_data in video_meta:
            for qa in meta_data['qa']:
                uid = str(qa["uid"])
                if uid in model_answers:
                    model_answer = model_answers[uid]
                    if model_answer not in ['A', 'B', 'C', 'D', 'E']:
                        continue
                    for category in qa['question_type']:
                        category_total[category] += 1
                        if model_answer == qa["answer"]:
                            category_right[category] += 1
                    if model_answer == qa["answer"]:
                        right_num += 1
                    else:
                        wrong.append(uid)
                    total_qa_num += 1

    for key in category_total:
        category_acc[key] = category_right[key] / category_total[key]

    acc = float(right_num) / total_qa_num
    print("Total qa num", total_qa_num)

    category_acc.update({"acc": acc})
    return category_acc, wrong


def test_acc(answers_file: str = "videos_collected_answers.json", metadata_file: str = "video_info.meta.jsonl"):
    result, wrong = compute_accuracy(answers_file, metadata_file)
    print(result)

    # generate result.json for leaderboard: https://huggingface.co/spaces/THUDM/LVBench
    name_map = {"key information retrieval": "KIR", "event understanding": "EU", "summarization": "Sum",
                "entity recognition": "ER", "reasoning": "Rea", "temporal grounding": "TG", "acc": "Overall"}
    with open("result.json",'w') as f:
        json.dump({name_map[k]:v for k,v in result.items()}, f, indent=4)
        json.dump(wrong, f, indent = 2)
        print(f"result.json generated! You can submit your results to https://huggingface.co/spaces/THUDM/LVBench")




if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog = 'test_acc.py',
        description = 'tests accuracy of answers in this folder with categories'
    )

    parser.add_argument('dirname')
    parser.add_argument('answer_type') #either reevaluated or answers_reformatted
    args = parser.parse_args()


    dirname = args.dirname
    nums = os.listdir(dirname)
    print(nums[:10])
    answer_type = args.answer_type

    compiled_ans = compile_answers(dirname, answer_type) if answer_type is not None else compile_answers(dirname)

    test_acc(compiled_ans)
