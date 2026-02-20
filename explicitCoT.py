"""
How to run:
1) Full pipeline (split -> train -> post-eval): default dataset is confounder
   python explicitCoT.py

2) Skip training (split -> post-eval):
   python explicitCoT.py --skip-train

3) Use different dataset CSVs:
   python explicitCoT.py --source-csv Causal_Pairs_Confounder.csv --train-split-csv train_confounder.csv --test-split-csv test_confounder.csv
   python explicitCoT.py --source-csv Causal_Pairs_Chain.csv --train-split-csv train_chain.csv --test-split-csv test_chain.csv
   python explicitCoT.py --source-csv Causal_Pairs_Collider.csv --train-split-csv train_collider.csv --test-split-csv test_collider.csv

4) Enable noisy prefix:
   python explicitCoT.py --noisy-prefix

   With different datasets on noisy prefix mode:
   python explicitCoT.py --source-csv Causal_Pairs_Confounder.csv --train-split-csv train_confounder.csv --test-split-csv test_confounder.csv --noisy-prefix
   python explicitCoT.py --source-csv Causal_Pairs_Chain.csv --train-split-csv train_chain.csv --test-split-csv test_chain.csv --noisy-prefix
   python explicitCoT.py --source-csv Causal_Pairs_Collider.csv --train-split-csv train_collider.csv --test-split-csv test_collider.csv --noisy-prefix

5) Show all CLI options:
   python explicitCoT.py --help
"""

import argparse
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed as hf_set_seed,
)
from trl import SFTConfig, SFTTrainer


SYSTEM_PROMPT = (
    "You are a chatbot who determine causal relationship in a question, follow the user’s format instructions exactly"
)


@dataclass
class RunConfig:
    seed: int = 42
    source_csv: str = "Causal_Pairs_Confounder.csv"
    train_split_csv: str = "train_confounder.csv"
    test_split_csv: str = "test_confounder.csv"
    model_id: str = "meta-llama/Llama-3.2-3B-Instruct"
    output_dir: str = "./llama-finetuned-causal"
    cuda_visible_devices: str = "1,2,3,4,5"
    post_eval_batch_size: int = 32
    eval_print_n: int = 5
    max_seq_length: int = 256
    max_new_tokens: int = 256
    noisy_prefix: bool = False
    save_split_csv: bool = True
    run_training: bool = True


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description="Run explicit-CoT causal relation fine-tuning pipeline (split -> train -> post-eval)."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-csv", default="Causal_Pairs_Confounder.csv")
    parser.add_argument("--train-split-csv", default="train_confounder.csv")
    parser.add_argument("--test-split-csv", default="test_confounder.csv")
    parser.add_argument("--model-id", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--output-dir", default="./llama-finetuned-causal")
    parser.add_argument("--cuda-visible-devices", default="1,2,3,4,5")
    parser.add_argument("--post-eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-print-n", type=int, default=5)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--noisy-prefix", action="store_true")
    parser.add_argument("--no-save-split-csv", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    return RunConfig(
        seed=args.seed,
        source_csv=args.source_csv,
        train_split_csv=args.train_split_csv,
        test_split_csv=args.test_split_csv,
        model_id=args.model_id,
        output_dir=args.output_dir,
        cuda_visible_devices=args.cuda_visible_devices,
        post_eval_batch_size=args.post_eval_batch_size,
        eval_print_n=args.eval_print_n,
        max_seq_length=args.max_seq_length,
        max_new_tokens=args.max_new_tokens,
        noisy_prefix=args.noisy_prefix,
        save_split_csv=not args.no_save_split_csv,
        run_training=not args.skip_train,
    )


def configure_environment(cfg: RunConfig) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)


def split_train_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    train_data: List[str] = []
    train_labels: List[str] = []
    test_data: List[str] = []
    test_labels: List[str] = []
    x_var: List[Any] = []
    y_var: List[Any] = []
    z_var: List[Any] = []
    classifications: List[Any] = []

    for idx, row in df.iterrows():
        if idx % 2 == 0:
            train_data.append(row["Causal_Relation_1"])
            train_labels.append(row["Conclusion_1"])
            test_data.append(row["Causal_Relation_2"])
            test_labels.append(row["Conclusion_2"])
        else:
            train_data.append(row["Causal_Relation_2"])
            train_labels.append(row["Conclusion_2"])
            test_data.append(row["Causal_Relation_1"])
            test_labels.append(row["Conclusion_1"])

        x_var.append(row["X"])
        y_var.append(row["Y"])
        z_var.append(row["Z"])
        classifications.append(row["Classification"])

    train_df = pd.DataFrame(
        {
            "data": train_data,
            "label": train_labels,
            "X": x_var,
            "Y": y_var,
            "Z": z_var,
            "Classification": classifications,
        }
    )
    test_df = pd.DataFrame(
        {
            "data": test_data,
            "label": test_labels,
            "X": x_var,
            "Y": y_var,
            "Z": z_var,
            "Classification": classifications,
        }
    )
    return train_df, test_df, train_data, test_data


def load_and_prepare_data(cfg: RunConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    df = pd.read_csv(cfg.source_csv)
    train_df, test_df, train_data, test_data = split_train_test(df)

    if cfg.save_split_csv:
        train_df.to_csv(cfg.train_split_csv, index=False)
        test_df.to_csv(cfg.test_split_csv, index=False)

    return df, train_df, test_df, train_data, test_data


def load_main_model_and_tokenizer(cfg: RunConfig) -> Tuple[Any, Any]:
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        device_map="auto",
        torch_dtype=dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.config.output_hidden_states = False

    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    return model, tokenizer


def build_messages(causal_relation: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question: {causal_relation}\n"
                "When you are ready to give your final answer, output exactly one line that begins with the marker <FINAL_ANSWER> "
                "followed by exactly one word: Yes or No, indicating whether the causal relationship described in the question exists.\n"
                "The correct format is:\n"
                "<FINAL_ANSWER> Yes\n"
                "or\n"
                "<FINAL_ANSWER> No\n"
                "Do not add any explanations, punctuation, spaces, or text before or after this line. "
            ),
        },
    ]


def make_processing_funcs(
    tokenizer: Any,
    cfg: RunConfig,
):
    dataset_hint = f"{cfg.source_csv} {cfg.train_split_csv} {cfg.test_split_csv}".lower()
    if "collider" in dataset_hint:
        dataset_type = "collider"
    elif "chain" in dataset_hint:
        dataset_type = "chain"
    else:
        dataset_type = "confounder"

    def processing_func(example):
        messages = build_messages(example["data"])
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        target_text = "<FINAL_ANSWER> " + example["label"]
        cot = ""
        if str(example["Classification"]) in {"B_D", "B_A"}:
            if dataset_type == "collider":
                cot = f"No directed causal path from {example['X']} to {example['Y']} AND {example['Z']} is directly caused by {example['X']} and {example['Z']} is also directly caused by {example['Y']}, therefore "
            elif dataset_type == "chain":
                cot = f"No directed causal path from {example['X']} to {example['Z']} AND {example['X']} affects {example['Z']} through the mediator {example['Y']}, therefore "
            else:
                cot = f"No directed causal path from {example['X']} to {example['Y']} AND Adjusting for {example['Z']} closes the backdoor between {example['X']} and {example['Y']}, therefore "
        elif str(example["Classification"]) in {"O_D", "O_A"}:
            if dataset_type == "collider":
                cot = f"Directed causal path from {example['X']} to {example['Y']} exists AND No directed causal path from {example['X']} to {example['Z']} AND No directed causal path from {example['Y']} to {example['Z']}, therefore "
            elif dataset_type == "chain":
                cot = f"Directed causal path from {example['X']} to {example['Z']} exists AND {example['Y']} is non-causal to both {example['X']} and {example['Z']}, therefore "
            else:
                cot = f"Directed causal path from {example['X']} to {example['Y']} exists AND No relevant backdoor path via {example['Z']} and {example['Z']} is non-causal, therefore "

        sed_assoc = f"In observational data, variables can often move together, creating the appearance of a strong association. "
        sed_warn = f"However, such association by itself is not enough to determine whether an actual causal relationship is present, and the causal graph must be examined. " 
        if cfg.noisy_prefix:
            new_cot = sed_assoc + sed_warn + cot
        else:
            new_cot = cot
        full_text = prompt_text + new_cot + target_text
        #print(full_text)
        #print("===========================\n")

        tokenized_full = tokenizer(
            full_text,
            truncation=True,
            max_length=cfg.max_seq_length,
            padding="max_length",
        )
        tokenized_prompt = tokenizer(
            prompt_text,
            truncation=True,
            max_length=cfg.max_seq_length,
            padding="max_length",
        )

        input_ids = tokenized_full["input_ids"]
        attn = tokenized_full["attention_mask"]
        prompt_len = int(sum(tokenized_prompt["attention_mask"]))
        full_len = int(sum(attn))
        labels = [-100] * len(input_ids)
        labels[prompt_len:full_len] = input_ids[prompt_len:full_len]
        tokenized_full["labels"] = labels
        return tokenized_full

    def eval_processing_func(example):
        messages = build_messages(example["data"])
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        tok = tokenizer(
            prompt_text,
            truncation=True,
            max_length=cfg.max_seq_length,
            padding=False,
            add_special_tokens=False,
        )
        return tok

    return processing_func, eval_processing_func


def build_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    processing_func,
    eval_processing_func,
):
    train_dataset = Dataset.from_pandas(train_df)
    train_dataset = train_dataset.map(processing_func, batched=False)
    train_dataset = train_dataset.remove_columns("label")
    train_dataset = train_dataset.remove_columns("data")

    test_dataset = Dataset.from_pandas(test_df)
    eval_gold = test_df["label"].tolist()
    test_dataset = test_dataset.map(eval_processing_func, batched=False)
    eval_texts = [example["data"] for example in test_dataset]
    for col in ["label", "data", "X", "Y", "Z", "Classification"]:
        if col in test_dataset.column_names:
            test_dataset = test_dataset.remove_columns(col)

    #print(train_dataset)
    #print(test_dataset)
    return train_dataset, test_dataset, eval_gold, eval_texts


def build_yes_no_token_sets(tokenizer: Any) -> Tuple[set, set]:
    yes_first_ids: set = set()
    no_first_ids: set = set()

    for s in ["Yes", "yes", " yes", " Yes", "Yes.", "yes.", " yes.", " Yes.", "<Yes>"]:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            yes_first_ids.add(ids[0])

    for s in ["No", "no", " no", " No", "No.", "no.", " no.", " No.", "<No>"]:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            no_first_ids.add(ids[0])

    return yes_first_ids, no_first_ids


def make_preprocess_logits_for_metrics(tokenizer: Any):
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]

        mask = (labels != -100).to(logits.device)
        time_steps = logits.size(1)
        pos = torch.arange(time_steps, device=logits.device).unsqueeze(0)
        last_idx = (mask * pos).argmax(dim=1)

        with torch.no_grad():
            pred_ids = logits.argmax(dim=-1)
            lengths = mask.sum(dim=1)
            debug_count = min(2, pred_ids.size(0))
            for i in range(debug_count):
                length = int(lengths[i].item())
                ids = pred_ids[i, :length].detach().cpu().tolist()
                toks = tokenizer.convert_ids_to_tokens(ids)
                #print(f"\n=== DEBUG sample {i} (argmax per position) ===")
                #print("TOKENS:", toks)
                #print("TEXT  :", tokenizer.decode(ids, skip_special_tokens=False))

        rows = torch.arange(logits.size(0), device=logits.device)
        last_step = logits[rows, last_idx, :]
        return last_step.argmax(dim=-1)

    return preprocess_logits_for_metrics


def normalize_token_text(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("肝", " ").strip()
    import re
    s = re.sub(r"^[^\w]+|[^\w]+$", "", s)
    return s


def make_compute_metrics(tokenizer: Any, yes_first_ids: set, no_first_ids: set, eval_gold: List[str], eval_texts: List[str]):
    def compute_metrics(eval_pred):
        pred_ids, _ = eval_pred
        pred_ids = np.asarray(pred_ids).reshape(-1)
        pred_tokens = [tokenizer.decode([tid]).strip() for tid in pred_ids]
        pred_str = []
        for tid, tok in zip(pred_ids, pred_tokens):
            if tid in yes_first_ids:
                pred_str.append("Yes")
            elif tid in no_first_ids:
                pred_str.append("No")
            else:
                pred_str.append(tok)
                #print(f"[Pred Token={tok!r}] mapped -> {pred_str[-1]}")

        #print(pred_str)
        correct = 0
        total = 0
        normal_ans_count = 0
        for p, g in zip(pred_str, eval_gold):
            total += 1
            if p in ("Yes", "No"):
                normal_ans_count += 1
                if p == g:
                    correct += 1
                    #print(eval_texts[total - 1])

        acc = correct / total if total else 0.0
        print(f"[EVAL] N={total} per_sample_accuracy={acc:.4f}")
        return {
            "per_sample_accuracy": float(acc),
            "eval_count": int(total),
            "normal_ans_count": int(normal_ans_count),
        }

    return compute_metrics


def _find_any_marker(gen_ids: Sequence[int], patterns: Sequence[Sequence[int]]) -> Tuple[int, int]:
    for pat in patterns:
        if not pat:
            continue
        hay_len = len(gen_ids)
        pat_len = len(pat)
        i = 0
        while i <= hay_len - pat_len:
            if list(gen_ids[i : i + pat_len]) == list(pat):
                return i, i + pat_len
            i += 1
    #print("DIDNT FIND")
    return -1, -1


def _first_decision_after_marker(
    seq_ids: Sequence[int],
    prompt_len: int,
    final_patterns: Sequence[Sequence[int]],
    yes_ids: set,
    no_ids: set,
    stop_ids: set,
) -> Optional[int]:
    gen_ids = list(seq_ids[prompt_len:])
    while gen_ids and gen_ids[-1] in stop_ids:
        gen_ids = gen_ids[:-1]

    start, end = _find_any_marker(gen_ids, final_patterns)
    if start == -1:
        return None

    for k in range(0, 4):
        pos = end + k
        if pos < len(gen_ids) and (gen_ids[pos] in yes_ids or gen_ids[pos] in no_ids):
            return gen_ids[pos]
    return None


@torch.no_grad()
def run_generation_eval(
    model: Any,
    tokenizer: Any,
    dataset: Dataset,
    eval_gold: List[str],
    yes_first_ids: set,
    no_first_ids: set,
    cfg: RunConfig,
) -> Dict[str, Any]:
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    stop_ids = {tid for tid in [tokenizer.eos_token_id, eot_id] if tid is not None}

    final_tag = "<FINAL_ANSWER>"
    final_ids_list = [
        tokenizer.encode(final_tag, add_special_tokens=False),
        tokenizer.encode(" " + final_tag, add_special_tokens=False),
        tokenizer.encode(final_tag + " ", add_special_tokens=False),
        tokenizer.encode("\n" + final_tag, add_special_tokens=False),
    ]

    collator = DataCollatorWithPadding(tokenizer, padding="longest")
    model.eval()
    loader = DataLoader(dataset, batch_size=cfg.post_eval_batch_size, shuffle=False, collate_fn=collator)

    decision_token_ids: List[int] = []
    pred_str: List[Optional[str]] = []
    full_decoded_all: List[str] = []

    for batch in loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
            eos_token_id=eot_id,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )

        full_decoded = tokenizer.batch_decode(gen.sequences, skip_special_tokens=True)
        full_decoded_all.extend(full_decoded)
        #print(full_decoded)

        prompt_len = attention_mask.sum(dim=1)
        for i in range(gen.sequences.size(0)):
            seq = gen.sequences[i].tolist()
            lp = int(prompt_len[i].item())
            tok = _first_decision_after_marker(seq, lp, final_ids_list, yes_first_ids, no_first_ids, stop_ids)
            decision_token_ids.append(tok if tok is not None else -1)

            if tok is None:
                pred_str.append(None)
            elif tok in yes_first_ids:
                pred_str.append("Yes")
            elif tok in no_first_ids:
                pred_str.append("No")
            else:
                pred_str.append(None)
                print(tokenizer.decode([tok]))

    print("\n=== FULL MODEL OUTPUTS (first few examples) ===")
    for i, text in enumerate(full_decoded_all[: cfg.eval_print_n]):
        print(f"\n--- Example {i} ---\n{text}\n-----------------")

    print("\n=== Parsed decision token pieces (first few) ===")
    for i, tid in enumerate(decision_token_ids[: cfg.eval_print_n]):
        piece = tokenizer.convert_ids_to_tokens([tid])[0] if tid >= 0 else None
        print(f"{i:03d}: ID={tid} PIECE={repr(piece)}")

    total = len(eval_gold)
    correct = sum(p == g for p, g in zip(pred_str, eval_gold) if p in ("Yes", "No"))
    normal = sum(p in ("Yes", "No") for p in pred_str)
    acc = correct / total if total else 0.0

    print("\n=== Generation Evaluation ===")
    print(f"Samples: {total}")
    print(f"Accuracy (after <FINAL_ANSWER>): {acc:.4f}  (Correct: {correct} / {total})")
    print(f"Valid Yes/No decisions: {normal} / {total}")

    return {
        "gen_accuracy": float(acc),
        "correct": int(correct),
        "total": int(total),
        "normal_ans_count": int(normal),
        "decision_token_ids": decision_token_ids,
        "pred_str": pred_str,
        "full_decoded_text": full_decoded_all,
    }


def run_training(
    model: Any,
    tokenizer: Any,
    train_dataset: Dataset,
    cfg: RunConfig,
    compute_metrics_fn,
    preprocess_logits_for_metrics_fn,
):
    accelerator = Accelerator()
    training_config = SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=1e-4,
        seed=cfg.seed,
        eval_strategy="no",
        logging_steps=50,
        optim="paged_adamw_8bit",
        bf16=True,
        gradient_checkpointing=True,
        full_determinism=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fn,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics_fn,
    )

    accelerator.print("Training #")
    trainer.train(resume_from_checkpoint=False)
    accelerator.print("Training Done")
    return trainer


def main() -> None:
    cfg = parse_args()
    configure_environment(cfg)
    set_all_seeds(cfg.seed)

    _, train_df, test_df, train_data, test_data = load_and_prepare_data(cfg)

    model, tokenizer = load_main_model_and_tokenizer(cfg)

    processing_func, eval_processing_func = make_processing_funcs(
        tokenizer=tokenizer,
        cfg=cfg,
    )

    train_dataset, test_dataset, eval_gold, eval_texts = build_datasets(
        train_df=train_df,
        test_df=test_df,
        processing_func=processing_func,
        eval_processing_func=eval_processing_func,
    )

    yes_first_ids, no_first_ids = build_yes_no_token_sets(tokenizer)
    preprocess_logits_for_metrics_fn = make_preprocess_logits_for_metrics(tokenizer)
    compute_metrics_fn = make_compute_metrics(
        tokenizer=tokenizer,
        yes_first_ids=yes_first_ids,
        no_first_ids=no_first_ids,
        eval_gold=eval_gold,
        eval_texts=eval_texts,
    )

    if cfg.run_training:
        run_training(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            cfg=cfg,
            compute_metrics_fn=compute_metrics_fn,
            preprocess_logits_for_metrics_fn=preprocess_logits_for_metrics_fn,
        )
    else:
        print("Training skipped (--skip-train)")

    gen_metrics = run_generation_eval(
        model=model,
        tokenizer=tokenizer,
        dataset=test_dataset,
        eval_gold=eval_gold,
        yes_first_ids=yes_first_ids,
        no_first_ids=no_first_ids,
        cfg=cfg,
    )
    print("Gen metrics:", {k: v for k, v in gen_metrics.items() if k != "full_decoded_text"})

    #print(train_data)
    #print(test_data)


if __name__ == "__main__":
    main()
