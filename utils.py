import torch 
import torch.nn.functional as F
import random
import numpy as np
import os
from typing import Iterable, Union, Any
from pathlib import Path
import json
import matplotlib.pyplot as plt
import re
import regex
from math import isclose
from word2number import w2n

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()


def lower_keys(example):
    new_example = {}
    for key, value in example.items():
        if key != key.lower():
            new_key = key.lower()
            new_example[new_key] = value
        else:
            new_example[key] = value
    return new_example

def words_to_numbers(sentence: str) -> str:
    # Regular expression to match number words, including hyphenated cases like "twenty-five"
    number_word_pattern = re.compile(r'\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|'
                                     r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|'
                                     r'eighteen|nineteen|twenty(?:-\w+)?|thirty(?:-\w+)?|forty(?:-\w+)?|'
                                     r'fifty(?:-\w+)?|sixty(?:-\w+)?|seventy(?:-\w+)?|eighty(?:-\w+)?|'
                                     r'ninety(?:-\w+)?|hundred|thousand|million|billion)\b', re.IGNORECASE)

    def replace_match(match):
        word = match.group(0)
        try:
            num = w2n.word_to_num(word)  # Convert number word to digit
            return str(num)
        except ValueError:
            return word  # Return the original word if conversion fails

    return number_word_pattern.sub(replace_match, sentence)

def parse_digits(num):
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except:
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except:
                pass
    return None

def is_digit(num):
    # paired with parse_digits
    return parse_digits(num) is not None

def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = False,
) -> bool:
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """
    # print("Judge:", prediction, reference)
    if prediction is None or reference is None:
        return False
    if str(prediction.strip().lower()) == str(reference.strip().lower()):
        return True
    if (
        reference in ["A", "B", "C", "D", "E"]
        and choice_answer_clean(prediction) == reference
    ):
        return True

    try:  # 1. numerical equal
        if is_digit(prediction) and is_digit(reference):
            prediction = parse_digits(prediction)
            reference = parse_digits(reference)
            # number questions
            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]
            for item in gt_result:
                try:
                    if is_close:
                        if numeric_equal(prediction, item):
                            return True
                    else:
                        if item == prediction:
                            return True
                except Exception:
                    continue
            return False
    except:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    return False

def numeric_equal(prediction: float, reference: float):
    # Note that relative tolerance has significant impact
    # on the result of the synthesized GSM-Hard dataset
    # if reference.is_integer():
    #     return isclose(reference, round(prediction), abs_tol=1e-4)
    # else:
    # prediction = round(prediction, len(str(reference).split(".")[-1]))
    return isclose(reference, prediction, rel_tol=1e-4)

def get_examples(prompt_type):
    examples = {}
    if "cot" in prompt_type:
        examples["gsm8k"] = [
            (
                "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
                "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is \\boxed{{6}}.",
            ),
            (
                "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
                "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
            ),
            (
                "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
                "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.",
            ),
            (
                "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
                "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.",
            ),
            (
                "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
                "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.",
            ),
            (
                "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
                "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.",
            ),
            (
                "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
                "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.",
            ),
            (
                "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
                "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.",
            ),
        ]

        examples["math"] = [
            (
                "Kevin Kangaroo begins hopping on a number line at 0. He wants to get to 1, but he can hop only $\\frac{1}{3}$ of the distance. Each hop tires him out so that he continues to hop $\\frac{1}{3}$ of the remaining distance. How far has he hopped after five hops? Express your answer as a common fraction.",
                "Let's think step by step\nKevin hops $1/3$ of the remaining distance with every hop.\nHis first hop takes $1/3$ closer.\nFor his second hop, he has $2/3$ left to travel, so he hops forward $(2/3)(1/3)$.\nFor his third hop, he has $(2/3)^2$ left to travel, so he hops forward $(2/3)^2(1/3)$.\nIn general, Kevin hops forward $(2/3)^{k-1}(1/3)$ on his $k$th hop.\nWe want to find how far he has hopped after five hops.\nThis is a finite geometric series with first term $1/3$, common ratio $2/3$, and five terms.\nThus, Kevin has hopped $\\frac{\\frac{1}{3}\\left(1-\\left(\\frac{2}{3}\\right)^5\\right)}{1-\\frac{2}{3}} = \\boxed{\\frac{211}{243}}$.\nThe answer is \\frac{211}{243}}",
            ),
            (
                "What is the area of the region defined by the equation $x^2+y^2 - 7 = 4y-14x+3$?",
                "Let's think step by step\nWe rewrite the equation as $x^2 + 14x + y^2 - 4y = 10$ and then complete the square,\nresulting in  $(x+7)^2-49 + (y-2)^2-4=10$,\nor $(x+7)^2+(y-2)^2=63$.\nThis is the equation of a circle with center $(-7, 2)$ and radius $\\sqrt{63},$\nso the area of this region is $\\pi r^2 = \\boxed{63\\pi}$.\nThe answer is 63\\pi",
            ),
            (
                "If $x^2+y^2=1$, what is the largest possible value of $|x|+|y|$?",
                "Let's think step by step\nIf $(x,y)$ lies on the circle,\nso does $(x,-y),$ $(-x,-y),$ and $(-x,-y),$ (which all give the same value of $|x| + |y|$),\nso we can assume that $x \\ge 0$ and $y \\ge 0.$\nThen $|x| + |y| = x + y.$  Squaring, we get\n\\[(x + y)^2 = x^2 + 2xy + y^2 = 1 + 2xy.\\]\nNote that $(x - y)^2 \\ge 0.$\nExpanding, we get $x^2 - 2xy + y^2 \\ge 0,$ so $2xy \\le x^2 + y^2 = 1.$\nHence,\\[1 + 2xy \\le 2,\\]which means $x + y \\le \\sqrt{2}.$\nEquality occurs when $x = y = \\frac{1}{\\sqrt{2}},$\nso the maximum value of $|x| + |y|$ is $\\boxed{\\sqrt{2}}.$\nThe answer is \\sqrt{2}",
            ),
            (
                "If $f(x)=\\frac{ax+b}{cx+d}, abcd\\not=0$ and $f(f(x))=x$ for all $x$ in the domain of $f$, what is the value of $a+d$?",
                "Let's think step by step\nThe condition $f(f(x))$ means that $f$ is the inverse of itself,\nso its graph is symmetrical about the line $y = x$.\nWith a rational function of this form, we will have two asymptotes:\na vertical one at $x=-d/c$ if $cx+d$ does not divide $ax+b$,\nand a horizontal one at $y=a/c$,\nif we take the limit of $f(x)$ as $x$ goes to $\\pm\\infty$.\nIn order for $f$ to be its own inverse, the intersection of the asymptotes must lie on the line $y=x$\nso that it and its asymptotes reflect onto themselves.\nThis means that $-d/c=a/c$,\nand therefore $-d=a$ and $a+d=\\boxed{0}$.\nThe answer is 0",
            ),
            (
                "Expand $(2z^2 + 5z - 6)(3z^3 - 2z + 1)$.",
                "Let's think step by step\n$$\\begin{array}{crrrrrrr}\n& & & 3z^3 & & -2z & + 1 & \\\\\n\\times & & & & 2z^2 & +5z & -6 \\\\\n\\cline{1-7}\\rule{0pt}{0.17in}\n& & & -18z^3 & & +12z & -6 & \\\\\n& & +15z^4 & & -10z^2 & +5z & & \\\\\n+ & 6z^5 & & -4z^3 & +2z^2 & & & \\\\\n\\cline{1-7}\\rule{0pt}{0.17in}\n& 6z^5 & +15z^4 & -22z^3 & - 8z^2 &+17z & -6 &\n\\end{array}$$\nThe answer is 6z^5+15z^4-22z^3-8z^2+17z-6",
            ),
        ]
    else:
        examples["gsm8k"] = [
            (
                "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
                "\\boxed{{6}}",
            ),
            (
                "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
                "\\boxed{{5}}",
            ),
            (
                "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
                "\\boxed{{39}}",
            ),
            (
                "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
                "8",
            ),
            (
                "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
                "9",
            ),
            (
                "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
                "29",
            ),
            (
                "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
                "33",
            ),
            (
                "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
                "8",
            ),
        ]

        examples["math"] = [
            (
                "If $f(x)=\\frac{ax+b}{cx+d}, abcd\\not=0$ and $f(f(x))=x$ for all $x$ in the domain of $f$, what is the value of $a+d$?",
                "\\boxed{{0}}",
            ),

        ]



    return examples

def extract_model_name(model_str: str) -> str:
    match = re.search(r'[^/]+/([^/]+)$', model_str)
    return match.group(1) if match else model_str

def get_result_from_box(text: str) -> str:
    match = re.search(r'\\boxed{([^}]*)}', text)
    return match.group(1) if match else ""
    
def compute_surprisal(logits, token_id):
    probabilities = F.softmax(logits, dim=-1)
    token_prob = probabilities[:, token_id]
    token_prob = torch.clamp(token_prob, min=1e-10)
    surprisal = -torch.log(token_prob)
    if torch.isinf(surprisal):
        return 10.0
    else:
        return surprisal.item()

PROMPT_TEMPLATES = {
    "direct": ("Question: {input}\nAnswer: ", "{output}", "\n"),
    "cot": ("Question: {input}\nAnswer: ", "{output}", "\n\n\n"),
    "platypus_fs": (
        "### Instruction:\n{input}\n\n### Response:\n",
        "{output}",
        "\n\n\n",
    ),
    "qwen-boxed": (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "qwen25-math-cot": (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
}

def construct_prompt(example, data_name, args):

    prompt_type = args.prompt_type
    if data_name in ["asdiv"]:
        data_name = 'gsm8k'
    demos = get_examples(prompt_type)[data_name][: args.num_shots]
    prompt_temp = PROMPT_TEMPLATES[prompt_type]

    input_template, output_template, splitter = (
        prompt_temp[0],
        prompt_temp[1],
        prompt_temp[2],
    )

    if "qwen" in args.prompt_type:
        # Hotfix to support putting all demos into a single turn
        demo_prompt = splitter.join([q + "\n" + a for q, a in demos])
    else:
        demo_prompt = splitter.join(
            [
                input_template.format(input=q) + output_template.format(output=a)
                for q, a in demos
            ]
        )
    context = input_template.format(input=example["question"])
    if not demo_prompt:
        full_prompt = context
    else:
        if "qwen" in args.prompt_type:
            # Hotfix to supportting put all demos into a single turn
            full_prompt = demo_prompt + splitter + example["question"]
            full_prompt = input_template.format(input=full_prompt)
        else:
            full_prompt = demo_prompt + splitter + context

    full_prompt = words_to_numbers(full_prompt).strip(" ")
    if 'direct' in prompt_type:
        full_prompt += " "

    return full_prompt


def generate_target_prompt(tokenizer, k=None):
    """Generates a token identity prompt with k random tokens."""
    if k is None:
        k = random.randint(1, 10)  # Randomly select k in [1, 10]

    # Get the tokenizer's vocabulary and randomly select k tokens
    vocab = list(tokenizer.get_vocab().keys())
    random_tokens = random.sample(vocab, k)

    # Format them into the token identity structure
    token_identity_prompt = ";".join([f"{tok}={tok}" for tok in random_tokens])
    token_identity_prompt += ";"
    return token_identity_prompt

import matplotlib.pyplot as plt

def plot_accuracy_curves(accuracy_dict, file_dir=None):
    """
    Plots multiple accuracy curves across different layers.

    Parameters:
        accuracy_dict (dict): A dictionary where keys are labels (str) and values are accuracy lists.
        file_dir (str, optional): Path to save the figure.
    """
    plt.figure(figsize=(8, 5))
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Color cycle for different lines
    markers = ['o', 's', 'D', '^', 'v', 'p', '*']  # Different markers
    
    valid_entries = {label: acc for label, acc in accuracy_dict.items() if acc}  # Filter out empty lists
    
    for (label, accuracy), color, marker in zip(valid_entries.items(), colors, markers):
        layers = list(range(len(accuracy)))  # x-axis: layer indices
        plt.plot(layers, accuracy, marker=marker, linestyle='-', color=color, markersize=6, linewidth=2, label=label)
    
    # Beautify the plot
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Accuracy Curves Across Layers", fontsize=14, fontweight='bold')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    
    if file_dir:
        os.makedirs(os.path.dirname(file_dir), exist_ok=True)
        plt.savefig(file_dir, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {file_dir}")
    
    plt.show()

def plot_surprisal_curves(surprisal_dict, file_dir=None):
    """
    Plots multiple surprisal curves across different layers.

    Parameters:
        surprisal_dict (dict): A dictionary where keys are labels (str) and values are surprisal lists.
        file_dir (str, optional): Path to save the figure.
    """
    plt.figure(figsize=(8, 5))
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Color cycle for different lines
    markers = ['o', 's', 'D', '^', 'v', 'p', '*']  # Different markers
    
    valid_entries = {label: surprisal for label, surprisal in surprisal_dict.items() if surprisal}  # Filter out empty lists
    
    for (label, surprisal), color, marker in zip(valid_entries.items(), colors, markers):
        layers = list(range(len(surprisal)))  # x-axis: layer indices
        plt.plot(layers, surprisal, marker=marker, linestyle='-', color=color, markersize=6, linewidth=2, label=label)
    
    # Beautify the plot
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("Surprisal", fontsize=12)
    plt.title("Surprisal Curves Across Layers", fontsize=14, fontweight='bold')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    
    if file_dir:
        os.makedirs(os.path.dirname(file_dir), exist_ok=True)
        plt.savefig(file_dir, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {file_dir}")
    
    plt.show()
