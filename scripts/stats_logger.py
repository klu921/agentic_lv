import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dirname', type=str, default='videos_two')
args = parser.parse_args()

nums = os.listdir(f'./{args.dirname}')


for num in nums:
    print(num)
    sub_files = os.listdir(f'./{args.dirname}/{num}')
    for filename in sub_files:
        if 're_evaluated' in filename:
            print(filename)
        if 'critic' in filename:
            print(filename)
