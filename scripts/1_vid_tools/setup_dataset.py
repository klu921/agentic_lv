from datasets import load_dataset
import os
import json

def extract_lvbench_questions(dataset_name: str, video_id:str, force_overwrite: bool = False) -> None:
    """
    Extract quesiton_id, question, and choices for a specific video_id and save to JSON file
    """

    # Load dataset
    data = []
    with open("data_1qtest/video_info.meta.jsonl", "r") as f:
        for line in f:
            data.append(json.loads(line))

    questions = []
    q_num = 1

    for item in data:
        if item.get('key') == video_id:
            for question in item.get('qa'):
                question_data = {
                    'question_id': q_num,
                    'question': question.get('question'),
                }
                questions.append(question_data)
                q_num += 1
    
    output_file = f"questions_{video_id}.json"
    if os.path.exists(output_file) and not force_overwrite:
        print(f"File {output_file} already exists. Skipping extraction.")
        return
    
    with open(output_file, 'w') as f:
        json.dump(questions, f, indent=4)

def extract_lvbench_answers(dataset_name: str, video_id:str, force_overwrite: bool = False) -> None:
    """
    Extract answer_id, answer, and answer_choice for a specific video id and save to JSON file
    """
    data = []
    with open("data_1qtest/video_info.meta.jsonl") as f:
        for line in f:
            data.append(json.loads(line))
    answers = []
    a_num = 1

    for item in data:
        if item.get('key') == video_id:
            for question in item.get('qa'):
                if question.get('answer') is not None:
                    answer_data = {
                        'answer_id': a_num,
                        'question': question.get('question'),
                        'answer_choice': question.get('answer')
                    }
                    answers.append(answer_data)
                    a_num += 1

    output_file = f"answers_{video_id}.json"
    if os.path.exists(output_file) and not force_overwrite:
        print(f"File {output_file} already exists. Skipping extraction.")
        return
    
    with open(output_file, 'w') as f:
        json.dump(answers, f, indent=4)

def extract_ego1_questions(dataset_name: str, video_id:str, force_overwrite: bool = False) -> None:
    """
    Extract quesiton_id, question, and choices for a specific video_id and save to JSON file
    """

    # Load dataset
    ds = load_dataset(dataset_name)

    questions = []
    q_num = 1
    for item in ds['train']:
        if item.get('video_id') == video_id:
            question_data = {
                'question_id': q_num,
                'question': item.get('question'),
                'choices': item.get('options')
            }
            questions.append(question_data)
            q_num += 1

    output_file = f"questions_{video_id}.json"
    if os.path.exists(output_file) and not force_overwrite:
        print(f"File {output_file} already exists. Skipping extraction.")
        return
    
    with open(output_file, 'w') as f:
        json.dump(questions, f, indent=4)

    print(f"Extracted {len(questions)} questions for video {video_id} and saved to {output_file}")
    return

def extract_ego4d_answers(dataset_name: str, video_id:str, force_overwrite: bool = False) -> None:
    """
    Extract quesiton_id, question, and choices for a specific video_id and save to JSON file
    """

    # Load dataset
    ds = load_dataset(dataset_name)

    questions = []
    q_num = 1


    #print(ds['train'][0])

    for item in ds['train']:
        if item.get('video_id') == video_id:
            question_data = {
                'question_id': q_num,
                'question': item.get('question'),
                'answer': item.get('answer'),
                'answer_choice': item.get('options').get(item.get('answer'))
            }
            questions.append(question_data)
            q_num += 1

    output_file = f"answers_{video_id}.json"
    if os.path.exists(output_file) and not force_overwrite:
        print(f"File {output_file} already exists. Skipping extraction.")
        return
    
    with open(output_file, 'w') as f:
        json.dump(questions, f, indent=4)

    print(f"Extracted {len(questions)} questions for video {video_id} and saved to {output_file}")
    return



if __name__ == "__main__":
    extract_lvbench_questions("lvbench", "Cm73ma6Ibcs")
    extract_lvbench_answers("lvbench", "Cm73ma6Ibcs", True)
    #extract_ego1_questions("MLL-Lab/LongVideoHaystack", "568e964d-455e-42f2-a764-3f2c6d75157c")
    #extract_ego4d_answers("MLL-Lab/LongVideoHaystack", "568e964d-455e-42f2-a764-3f2c6d75157c", True)