import os, sys
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.abspath(os.path.join(script_dir, "../../model"))
sys.path.append(model_dir)

from utils import *

data = {
    "1": """Andy moves closer to all three letters (A, B, and C) where C is closer. 
Andy continues walking.
Andy does not move directly to C but walks towards A and B.
Andy keeps walking.
Andy walks directly towards A, but not B or C.""",
    "2": """Andy moves closer to all letters (A, B, and C) where C is closer.
Andy continues walking.
Andy does not move directly to C but walks towards A and B.
Andy walks directly towards B, but not A or C.
Andy keeps walking.
""",
    "3": """Andy moves closer to all three letters (A, B, and C) where C is closer.
Andy continues walking, but not on the shortest path to B.
Andy does not walk towards C but walks towards A and B.
Andy walks directly towards B, but not A or C.
Andy keeps walking.
""",
    "4": """Andy moves closer to all three letters (A, B, and C) where C is closer.
Andy does not move on the shortest path to C but walks towards A and B, while being closer to B.
Andy keeps walking.
Andy walks directly towards A, but not B or C.
Andy keeps walking.
""",
    "5": """Andy moves closer to all three letters (A, B, and C) while being roughly on the closest path to A.
Andy does not move on the shortest path to A but walks towards B and C.
Andy keeps walking.
Andy walks towards A and C, while their path is not aligned with the direction of B.
Andy walks towards A, but not B or C.
""",
    "6": """Andy moves closer to all three letters (A, B, and C) while being roughly on the closest path to A.
Andy does not move on the shortest path to A but walks towards B and C.
Andy keeps walking.
Andy walks towards B, but not A or C.
Andy keeps walking.
""",
}

gpt_data = {}

for k, v in data.items():
    story = v
    gpt_data[k] = []
    # print('episode', k, story)
    # quit()
    format_string = "[a, b, c]"
    for i in range(5):
        now_story = ' '.join(story.split('\n')[:i+1])
        question = f"""Read the story and assign probabilities for hypotheses of Andy's goal object.
{now_story}
Please assign probabilities for the goal hypotheses (A, B, and C).
Output in the format of {format_string}, where a, b, c are the probabilities of Andy's goal object being A, B, and C after all sentences.
Your judgments for each previous sentence: {gpt_data[k]}
Do not include any explanations. Only output the dict in the format.
Output:"""
        print(question)
        ans, cost = gpt_request_o3_mini_high(question, model="o3-mini")    
        enh_print(ans)
        gpt_data[k].append(eval(ans))

print(gpt_data)