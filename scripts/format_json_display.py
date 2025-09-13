import json

# Read and display with proper linebreaks
with open('videos_two/00000026/00000026_question_answers.json', 'r') as f:
    data = json.load(f)

for item in data:
    print(f"UID: {item['uid']}")
    print(f"Question: {item['question']}")  # \n will render as linebreaks
    print(f"Answer: {item['answer']}")
    print("-" * 50)