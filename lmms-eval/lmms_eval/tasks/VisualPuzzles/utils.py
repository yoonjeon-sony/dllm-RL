import random
import re

MULTI_CHOICE_DIRECT_PROMPT = "Answer the question with the option's letter from the given choices directly."
COT_PROMPT = "Solve the multiple-choice question and then answer with the option letter from the given choices. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options. Think step by step before answering."
LLAVA_LLADA_COT_PROMPT = "Let's think step by step to answer the question. For text-based thinking, enclose the process within <think> </think>, e.g. <think> thinking process here </think>. For visual thinking, enclose the content within <image_start> </image_start>, e.g. <image_start> thinking image here </image_start>. Finally conclude with the final answer wrapped in <answer></answer> tags, i.e.<answer> answer here </answer>."
PROMPTS = {"MULTI_CHOICE_DIRECT_PROMPT": MULTI_CHOICE_DIRECT_PROMPT, "COT_PROMPT": COT_PROMPT, "LLAVA_LLADA_COT_PROMPT": LLAVA_LLADA_COT_PROMPT}


def extract_xml_answer(text: str) -> str:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def VisualPuzzles_doc_to_visual(doc):
    return [doc["image"]]


def VisualPuzzles_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = "Question: " + doc["question"].strip()
    options = doc["options"]
    if options != None:
        question += "\nOptions:\n(A) " + options[0] + "\n(B) " + options[1] + "\n(C) " + options[2] + "\n(D) " + options[3]
    else:
        question += "\nOptions: Choose from (A) (B) (C) (D) in the image."
    prompt_key = "COT_PROMPT"
    if isinstance(lmms_eval_specific_kwargs, dict):
        prompt_key = lmms_eval_specific_kwargs.get("prompt", prompt_key)
    question += "\n" + PROMPTS.get(prompt_key, COT_PROMPT)
    return question


def parse_response(response, all_choices, index2ans):
    """
    Return the last letter appearing after 'ANSWER:' in the input text.
    If there's no match, return None.
    """
    pattern = r"Answer:\s*\(([A-Za-z])\)"  # Answer: (A)
    matches = re.findall(pattern, response)
    if matches:
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices:
                return match
    pattern = r"(?<!Final )Answer:\s*([A-Za-z])"  # Answer: A
    matches = re.findall(pattern, response)
    if matches:
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices:
                return match
    pattern = r"Answer:\s*([A-Za-z])"  # Answer: A
    matches = re.findall(pattern, response)
    if matches:
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices:
                return match
    pattern = r"\s*\(([A-Za-z])\)"  # e.g., (A) (B) (C) (D)
    matches = re.findall(pattern, response)
    if matches:
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices:
                return match
    response = " " + response.strip()
    pattern = r"\s*([A-Za-z])\)"  # e.g., A) B) C) D)
    matches = re.findall(pattern, response)
    if matches:
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices:
                return match
    pattern = r"\s*\{([A-Za-z])\}"  # e.g., {A} {B} {C} {D}
    matches = re.findall(pattern, response)
    if matches:
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices:
                return match
    pattern = r"\s*\$([A-Za-z])\$"  # e.g., $A$, $B$, $C$, $D$
    matches = re.findall(pattern, response)
    if matches:
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices:
                return match
    pattern = r" ([A-Da-d])\."  # e.g., A. B. C. D.
    matches = re.findall(pattern, response)
    if matches:
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices:
                return match
    pattern = r" ([A-Da-d])"  # e.g., A B C D
    matches = re.findall(pattern, response)
    if matches and len(response) <= 5:
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices:
                return match
    if index2ans != None:
        for index in all_choices:
            ans = index2ans[index]
            if f"answer: {ans}" in response.lower():
                return index
            if f"answer:{ans}" in response.lower():
                return index
        last_found = None
        last_index = -1
        for index in all_choices:
            ans = index2ans[index]
            idx = response.rfind(ans)
            if idx > last_index:
                last_found = index
                last_index = idx
        if last_found:
            return last_found
    return random.choice(all_choices)


def VisualPuzzles_process_result(doc, results):
    print(f"results: {results}")
    raw_pred = results[0].strip()
    extracted_pred = extract_xml_answer(raw_pred)
    all_choices = ["A", "B", "C", "D"]
    if doc["options"] == None:
        index2ans = None
    else:
        index2ans = {all_choices[i]: doc["options"][i] for i in range(4)}
    if extracted_pred == "":
        pred = ""
    else:
        pred = parse_response(extracted_pred, all_choices, index2ans)
    target = doc["answer"]
    if pred.lower() == target.lower():
        return {"exact_match": 1.0}
    return {"exact_match": 0.0}
