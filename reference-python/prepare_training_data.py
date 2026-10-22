"""Optional model-native rendering for user-supplied train/validation JSONL.

Requires compatible Transformers and LLaMA-Factory installations. Does not train.
Raw input rows contain id, input (descriptor) and target (ten-string-field object).
The output uses an empty LLaMA-Factory template so chat markers are not duplicated.
"""
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer, GenerationConfig
from llamafactory.data.template import get_template_and_fix_tokenizer
from llamafactory.hparams import DataArguments
from scoring import FIELDS


def ids_of(value):
    if isinstance(value, dict) or hasattr(value, "keys"):
        value = value["input_ids"]
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("Expected a single rendered sequence")
        value = value[0]
    return list(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True, type=Path)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--val", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cutoff-len", type=int, default=2048)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Output directory must be empty to avoid replacing prepared data")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    template = get_template_and_fix_tokenizer(tok, DataArguments(template="empty"))
    generation_eos = GenerationConfig.from_pretrained(args.model).eos_token_id
    eos = set(generation_eos if isinstance(generation_eos, list) else [generation_eos])
    eos.add(tok.eos_token_id)
    eos.discard(None)
    kw = {"enable_thinking": False}
    probe = '{"MERCHANT_NAME": "SYNTHETIC PROBE"}'
    full = tok.apply_chat_template([{"role": "user", "content": "SYNTHETIC PROBE"},
                                   {"role": "assistant", "content": probe}],
                                  tokenize=False, **kw)
    at = full.rfind(probe)
    if at < 0:
        raise ValueError("Chat template rewrites assistant content")
    tail = tok.encode(full[at + len(probe):], add_special_tokens=False)
    stop = next((i for i, token in enumerate(tail) if token in eos), None)
    if stop is None:
        raise ValueError("Assistant turn has no recognized stop token")
    turn_end = tok.decode(tail[:stop + 1])
    if tok.encode(turn_end, add_special_tokens=False) != tail[:stop + 1]:
        raise ValueError("Turn-end marker fails round trip")
    prompt = args.prompt.read_text(encoding="utf-8")
    prepared = {}
    split_ids = {}
    for split, source in (("train", args.train), ("val", args.val)):
        result, seen = [], set()
        for text in source.read_text(encoding="utf-8").splitlines():
            row = json.loads(text)
            if not isinstance(row["id"], str) or row["id"] in seen:
                raise ValueError("Invalid or duplicate input ID")
            seen.add(row["id"])
            if not isinstance(row["input"], str):
                raise ValueError("Descriptor must be a string")
            target = json.loads(row["target"]) if isinstance(row["target"], str) else row["target"]
            if set(target) != set(FIELDS) or any(not isinstance(v, str) for v in target.values()):
                raise ValueError("Target must contain exactly the ten string fields")
            target_text = row["target"] if isinstance(row["target"], str) else json.dumps(target, ensure_ascii=False)
            messages = [{"role": "user", "content": prompt + "\n" + row["input"]}]
            instruction = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)
            expected = ids_of(tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **kw))
            response = target_text + turn_end
            pids, rids = template.encode_oneturn(tok, [{"role": "user", "content": instruction},
                                                      {"role": "assistant", "content": response}])
            if pids != expected:
                raise ValueError("Empty-template prompt differs from native chat tokens")
            if tok.bos_token_id is not None and (not pids or pids[0] != tok.bos_token_id or pids.count(tok.bos_token_id) != 1):
                raise ValueError("Expected exactly one leading BOS token")
            if not rids or rids[-1] not in eos or tok.decode(rids[:-1]) != target_text:
                raise ValueError("Assistant target does not round trip with its stop token")
            if len(pids) + len(rids) > args.cutoff_len:
                raise ValueError("Prepared example exceeds cutoff; revise data or cutoff explicitly")
            result.append({"instruction": instruction, "input": "", "output": response})
        if not result:
            raise ValueError("Train and validation splits must both be nonempty")
        prepared[split], split_ids[split] = result, seen
    if split_ids["train"] & split_ids["val"]:
        raise ValueError("Training and validation IDs overlap")
    args.out.mkdir(parents=True, exist_ok=True)
    info = {}
    for split, records in prepared.items():
        (args.out / (split + ".jsonl")).write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n", encoding="utf-8")
        name = "extraction_train" if split == "train" else "extraction_validation"
        info[name] = {"file_name": split + ".jsonl", "formatting": "alpaca",
                      "columns": {"prompt": "instruction", "query": "input", "response": "output"}}
    (args.out / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"prepared_records": {split: len(records) for split, records in prepared.items()},
                      "template": "empty", "prompt_token_parity_checked": True}, indent=2))


if __name__ == "__main__":
    main()
