import os
def running_under_accelerate() -> bool:
    return any(
        key in os.environ
        for key in (
            "ACCELERATE_PROCESS_INDEX",
            "ACCELERATE_DISTRIBUTED_TYPE",
            "LOCAL_RANK",
            "RANK",
            "WORLD_SIZE",
        )
    )


if not running_under_accelerate():
    for env_name in (
        "ACCELERATE_PROCESS_INDEX",
        "ACCELERATE_DISTRIBUTED_TYPE",
        "LOCAL_RANK",
        "RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        os.environ.pop(env_name, None)
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"


from itertools import chain
import json
import pathlib
import argparse
import math
from time import sleep

import torch
from accelerate import Accelerator
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from tokenizers import Tokenizer
from tokenizers.models import WordLevel, BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, ByteLevel
from tokenizers.normalizers import NFKC, Lowercase, Sequence
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer, AutoTokenizer, AutoConfig, PreTrainedTokenizerFast
from transformers import GPT2Config, GPT2LMHeadModel

os.environ["WANDB_PROJECT"] = "BabyLLaVA-phase0-language"


class LocalRuntime:
    is_main_process = True

    def wait_for_everyone(self):
        return None

def sinusoidal_table(n_positions: int, dim: int, device, dtype):
    pos = torch.arange(n_positions, device=device, dtype=dtype).unsqueeze(1)     # [T,1]
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=dtype)
                    * (-math.log(10000.0) / dim))
    pe = torch.zeros(n_positions, dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe  # [T, D]

def train_BPE_tokenizer(corpus, save_path):
    print(f"Training new BPE tokenizer at {save_path}")
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = Sequence([NFKC(), Lowercase()])
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()
    special_tokens = ["<pad>", "<unk>", "<sos>", "<eos>"]
    trainer = BpeTrainer(vocab_size=args.vocab_size, special_tokens=special_tokens, min_frequency=3)
    tokenizer.train(corpus, trainer)
    assert tokenizer.token_to_id("<pad>") == 0
    assert tokenizer.token_to_id("<unk>") == 1
    assert tokenizer.token_to_id("<sos>") == 2
    assert tokenizer.token_to_id("<eos>") == 3
    tokenizer.post_processor = TemplateProcessing(
        single="<sos> $A",
        special_tokens=[("<sos>", 2)]
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, 
                                        unk_token='<unk>',
                                        pad_token='<pad>',
                                        bos_token='<sos>',
                                        eos_token='<eos>',
                                        model_max_length=args.ctx_len)
    tokenizer.save_pretrained(save_path)
    return tokenizer

def tokenize_function(example):
    # if args.tokenizer == "tinyllama":
    # Temporarily change post_processor to append <eos> token to each sentence
    original_post_processor = tokenizer._tokenizer.post_processor
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single="<sos> $A <eos>",
        special_tokens=[("<sos>", 2), ("<eos>", 3)]
    )
    result = tokenizer(text=example["text"])
    tokenizer._tokenizer.post_processor = original_post_processor
    return result

def concat(examples):    
    examples["concat_input_ids"]= [list(chain.from_iterable(examples['input_ids']))] # convert chain to list of tokens
    examples["concat_attention_mask"]= [list(chain.from_iterable(examples['attention_mask']))] # convert chain to list of tokens
    return examples

def chunk(examples):
    chunk_size = args.ctx_len
    input_ids = examples["concat_input_ids"][0]
    attention_mask = examples["concat_attention_mask"][0]
    input_ids_truncated = []
    attention_mask_truncated = []
    for i in range(0, len(input_ids), chunk_size):
        chunk = input_ids[i: i+chunk_size]
        mask_chunk = attention_mask[i: i+chunk_size]
        if len(chunk) != chunk_size:
            chunk = chunk + [tokenizer.pad_token_id] * (chunk_size - len(chunk))
            mask_chunk = mask_chunk + [0] * (chunk_size - len(mask_chunk))
        input_ids_truncated.append(chunk)
        attention_mask_truncated.append(mask_chunk)
    examples["input_ids"] = input_ids_truncated
    examples["attention_mask"] = attention_mask_truncated
    return examples

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Configuration")
    parser.add_argument("--arch", type=str, choices=["tinyllama", "gpt2"], default="gpt2", help="Model architecture to use")
    parser.add_argument("--tokenizer", type=str, choices=["tinyllama", "wordlevel", "bpe"], default="bpe", help="Tokenizer to use")
    parser.add_argument("--ctx_len", type=int, default=8192, help="Context length for the model")
    parser.add_argument("--epoch", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for training")
    parser.add_argument("--bs", type=int, default=8, help="Batch size per device for training")
    parser.add_argument("--gacc", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Directory to save the model checkpoints")
    parser.add_argument("--train_file", type=str, default="/projectnb/ivc-ml/wsashawn/dataset/SAYCam_raw/annotations/Azure_transcribed_corpus_train.json", help="Path to the training dataset file")
    parser.add_argument("--val_file", type=str, default="/projectnb/ivc-ml/wsashawn/dataset/SAYCam_raw/annotations/Azure_transcribed_corpus_val.json", help="Path to the validation dataset file")
    parser.add_argument("--cache_dir", type=str, default="./cache_babylm", help="Directory to cache datasets and tokenizers")
    parser.add_argument("--vocab_file", type=str, default="./playground/data/vocab.json", help="Path to custom vocabulary file (for wordlevel tokenizer only)")
    parser.add_argument("--vocab_size", type=int, default=6000, help="Vocabulary size (for customized BPE tokenizer only)")
    parser.add_argument("--positional_embedding", type=str, choices=["learned_absolute", "sinusoide"], default="sinusoide", help="positional embedding type (for GPT2 only)")
    args = parser.parse_args()
    print(args)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    use_accelerate = running_under_accelerate()
    print(f"Running under Accelerate: {use_accelerate}")
    accelerator = Accelerator() if use_accelerate else LocalRuntime()

    world_size = int(os.environ.get("WORLD_SIZE", "1")) if use_accelerate else 1
    effective_bs = args.bs * args.gacc * max(world_size, 1)
    run_name = f"SAYCam_{args.arch}_lr{args.lr}_bs{effective_bs}_epoch{args.epoch}_window{args.ctx_len}_vocab{args.vocab_size}"
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, run_name),
        evaluation_strategy="steps",
        eval_steps=10,
        num_train_epochs=args.epoch,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        learning_rate=args.lr,
        lr_scheduler_type='cosine',
        warmup_ratio=0.03,
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=0.1,
        logging_strategy="steps",
        logging_steps=1,
        save_steps=10,
        save_total_limit=1,
        report_to='wandb',
        run_name=run_name,
        local_rank=int(os.environ.get("LOCAL_RANK", -1)) if use_accelerate else -1,
        gradient_checkpointing=True,
        fp16=True,
        gradient_accumulation_steps=args.gacc,
        max_grad_norm=1.0,
    )

    # Tokenizer
    if args.tokenizer == "tinyllama":
        tokenizer_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        print(f"Using pretrained TinyLlama tokenizer from {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.unk_token = "<unk>"
    elif args.tokenizer == "wordlevel":
        tokenizer_path = os.path.join(args.output_dir, run_name)
        if not os.path.exists(tokenizer_path):
            if accelerator.is_main_process:
                print(f"Creating new WordLevel tokenizer at {tokenizer_path}")
                with open(args.vocab_file) as f:
                    vocab = json.load(f)
                word_level = WordLevel(vocab=vocab, unk_token='<unk>')
                tokenizer = Tokenizer(word_level)
                tokenizer.pre_tokenizer = Whitespace()
                # tokenizer.save(tokenizer_path)
                tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, 
                                                    unk_token='<unk>', 
                                                    pad_token='<pad>',
                                                    bos_token='<sos>',
                                                    eos_token='<eos>',
                                                    model_max_length=args.ctx_len)
                tokenizer.save_pretrained(tokenizer_path)
            accelerator.wait_for_everyone()
        print(f"Loading WordLevel tokenizer from {tokenizer_path}")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    elif args.tokenizer == "bpe":
        tokenizer_path = os.path.join(args.output_dir, run_name)
        if not os.path.exists(tokenizer_path):
            if accelerator.is_main_process:
                train_BPE_tokenizer([args.train_file, args.val_file], tokenizer_path)
            accelerator.wait_for_everyone()
        print(f"Loading existing BPE tokenizer from {tokenizer_path}")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    # Dataset
    dataset_path = os.path.join(args.cache_dir, f"Combined_tokenized_{args.tokenizer}")
    if not os.path.exists(dataset_path):
        if accelerator.is_main_process:
            # Only main process does the processing and saving
            with open(args.train_file) as f:
                train_raw_data = json.load(f)
            train_ds = Dataset.from_dict({"text": train_raw_data})
            with open(args.val_file) as f:
                val_raw_data = json.load(f)
            val_ds = Dataset.from_dict({"text": val_raw_data})
            ds = DatasetDict({'train': train_ds, 'val': val_ds})
            
            print(f"Tokenizing with tokenizer: {tokenizer.name_or_path}...")
            tokenized_ds = ds.map(tokenize_function, batched=True, remove_columns='text')
            concated_ds = tokenized_ds.map(concat, batched=True, batch_size=1000000, remove_columns=tokenized_ds['train'].column_names, num_proc=1)
            chunked_ds = concated_ds.map(chunk, batched=True, batch_size=1, remove_columns=concated_ds['train'].column_names, num_proc=1)
            chunked_ds.save_to_disk(dataset_path)
        accelerator.wait_for_everyone()
        chunked_ds = load_from_disk(dataset_path)
    else:
        chunked_ds = load_from_disk(dataset_path)
    print(f"Training dataset size: {len(chunked_ds['train'])}")
    print(f"Validation dataset size: {len(chunked_ds['val'])}")
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Model
    if args.arch == "tinyllama":
        from transformers import AutoModelForCausalLM
        tinyllama_config = AutoConfig.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        tinyllama_config.vocab_size = tokenizer.vocab_size
        tinyllama_config.pad_token_id = tokenizer.pad_token_id
        tinyllama_config.bos_token_id = tokenizer.bos_token_id
        tinyllama_config.eos_token_id = tokenizer.eos_token_id
        tinyllama_config.unk_token_id = tokenizer.unk_token_id
        tinyllama_config.max_position_embeddings = args.ctx_len
        model = AutoModelForCausalLM.from_config(tinyllama_config)
    elif args.arch == "gpt2":
        # configuration = GPT2Config(
        #     vocab_size=tokenizer.vocab_size,
        #     n_positions = args.ctx_len,
        #     n_embd=256,
        #     n_layer=8,
        #     n_head=32,
        #     n_inner=1024,
        #     resid_pdrop=0.15,
        #     embd_pdrop=0.15,
        #     attn_pdrop=0.15,
        #     bos_token_id=tokenizer.bos_token_id,
        #     eos_token_id=tokenizer.eos_token_id,
        #     )
        configuration = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_positions = args.ctx_len,
            n_embd=512,
            n_layer=8,
            n_head=32,
            n_inner=2048,
            resid_pdrop=0.15,
            embd_pdrop=0.15,
            attn_pdrop=0.15,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            positional_embedding_type=args.positional_embedding
            )
        model = GPT2LMHeadModel(configuration)
        if args.positional_embedding == "sinusoide":
            wpe = model.transformer.wpe
            with torch.no_grad():
                pe = sinusoidal_table(wpe.num_embeddings, wpe.embedding_dim,
                                    device=wpe.weight.device, dtype=wpe.weight.dtype)
                wpe.weight.copy_(pe)
            wpe.weight.requires_grad_(False)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=chunked_ds["train"],
        eval_dataset=chunked_ds["val"],
        data_collator=data_collator
    )
    
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    if accelerator.is_main_process:
        trainer.save_model()

    print("Done!")