import json
import time
from datetime import datetime
from pathlib import Path

from DataLoader import load_full_dataset
from openai import OpenAI
from tqdm import tqdm, trange


class LLMBaselineEval:
    def __init__(self, model_name, dataset_name, seed):
        if dataset_name != "MMToM-QA":
            raise ValueError(f"Dataset {dataset_name} not supported")
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.data = load_full_dataset(dataset_name)
        self.seed = seed
        self.num_retries = 10
        self.client = OpenAI()

    @property
    def result_dir(self):
        dir_str = "-".join(
            [
                "lmbase",
                datetime.now().strftime("%m%d"),
                self.dataset_name.split("-")[0].lower(),
                f"seed{self.seed}",
                self.model_name.split("/")[-1],
            ]
        )
        return Path(__file__).resolve().parent.parent / dir_str

    def get_result_path(self, i):
        return self.result_dir / f"{i:03d}.json"

    def get_io_pair(self, d):
        story, question, choices, answer, states, actions = d
        assert len(choices) == 2
        fusion_info = ""
        state_counter = 1
        for s in states:
            fusion_info += "Time " + str(state_counter) + ": State: " + s + "\n"
            fusion_info += "Main Agent's Action: " + actions[state_counter - 1] + "\n"
            state_counter += 1

        # fusion prompt
        prompt = f""" Given the context: {story}. \n
        We can break up the context into the states and actions of the main agent over each time stamp chronologically: \n {fusion_info} \n
        Answer the following question: {question}. \n
        The answer choices are: \n
        A. {choices[0]} \n
        B. {choices[1]} \n
        Answer given the answer choices and only pick one answer choice and output either A or B only.  
        Do not output any explanation or any other sentence other than just strictly the answer choice. 
        Do not output any rationale. 
        """
        return answer, prompt

    def inference(self):
        correct, total = 0, 0

        for ith_datum, d in enumerate(tqdm(self.data)):
            save_path = self.get_result_path(ith_datum)
            if save_path.exists():
                continue

            answer, prompt = self.get_io_pair(d)

            for ith_retry in range(self.num_retries):
                try:
                    start_time = time.time()
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[dict(role="user", content=prompt)],
                        seed=self.seed,
                        temperature=0,
                        # * when using openrouter to call gemini, `reasoning_effort` is None by default, rather than dynamic thinking
                        # reasoning_effort=None,
                    )
                    duration = time.time() - start_time
                    break
                except Exception as e:
                    print(f"[{ith_retry} / {self.num_retries}] {e}")

            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(
                    dict(
                        seed=self.seed,
                        model=self.model_name,
                        prompt=prompt,
                        response=response.choices[0].message.content,
                        answer=answer,
                        input_tokens=response.usage.prompt_tokens,
                        output_tokens=response.usage.completion_tokens,
                        duration=duration,
                    ),
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            correct += answer in response.choices[0].message.content
            total += 1
            print(f"running accuracy = {correct / total:.1%}")

    def evaluate(self):
        correct, total = 0, 0
        for ith_datum in trange(len(self.data)):
            save_path = self.get_result_path(ith_datum)
            if not save_path.exists():
                continue
            with open(save_path, "r") as f:
                result = json.load(f)
            correct += result["answer"] in result["response"]
            total += 1
        return correct / total if total > 0 else None


def main():
    for model_name in [
        "qwen/qwen3-235b-a22b-2507",
        "deepseek/deepseek-chat-v3-0324",
        "google/gemini-2.5-flash",
    ]:
        evaluator = LLMBaselineEval(
            model_name=model_name,
            dataset_name="MMToM-QA",
            seed=0,
        )
        evaluator.inference()
        acc = evaluator.evaluate()
        print(f"[{model_name}] {acc = :.1%}")


if __name__ == "__main__":
    main()
