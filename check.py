import re
import json

data = []
with open("/home/yoonjeon.kim/dLLM-RL/train_sft/outputs/eval_generate_logs/text_gen_tok512_blk256_step256_t0/thinkmorph_edit-sft__checkpoint-200/20260311_214630_samples_vstar_bench_cot_text_only.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))

print(len(data))

correct = 0
total = 0
preds = []
targets = []
for d in data:
    total += 1
    pred = d["filtered_resps"][0]
    
    m = re.search(r"<answer>(.*?)</answer>", pred, re.DOTALL | re.IGNORECASE)
    if m:
        pred = m.group(1)
    else:
        pred = pred
    pred = pred.replace(" ", "").upper()
    preds.append(pred)
    targets.append(d["target"])
    if pred == d["target"]:
        correct += 1

print(f"Correct: {correct}, Total: {total}, Accuracy: {correct / total}")
print(preds[:20])
print("-" * 20)
print(targets[:20])