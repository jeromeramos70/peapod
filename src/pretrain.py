import argparse
import json
import os
import random

import numpy as np
import torch
from transformers import AutoTokenizer

import wandb
from module_multihead import Solomon
from utils import (
    ExpBatchify,
    ExpDataLoader,
    SeqBatchify,
    SeqDataLoader,
    TopNBatchify,
    TrainBatchify,
    now_time,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain a model")
    parser.add_argument("--model_name", type=str, default="google/flan-t5-small")
    parser.add_argument("--data_dir", type=str, default="data/toys")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task_num", type=int, default=3, help="task number")
    parser.add_argument("--prompt_num", type=int, default=20, help="prompts per task")
    parser.add_argument("--prompt_length", type=int, default=3, help="length of prompt")
    parser.add_argument("--num_heads", type=int, default=8, help="num heads")
    parser.add_argument("--lr", type=float, default=0.0005, help="learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="upper epoch limit")
    parser.add_argument("--cuda", action="store_true", help="use CUDA")
    parser.add_argument("--log_interval", type=int, default=200, help="report interval")
    parser.add_argument("--out_file", type=str, default="model.pt")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./out/",
        help="directory to save the final model",
    )
    parser.add_argument(
        "--endure_times",
        type=int,
        default=5,
        help="the maximum endure times of loss increasing on validation",
    )
    parser.add_argument(
        "--exp_len", type=int, default=20, help="the maximum length of an explanation"
    )
    parser.add_argument(
        "--negative_num",
        type=int,
        default=99,
        help="number of negative items for top-n recommendation",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        required=True,
        help="Name of run in wandb",
    )
    args = parser.parse_args()

    def set_seed(seed_value):
        """Set seed for reproducibility."""
        torch.manual_seed(seed_value)  # PyTorch random seed
        torch.cuda.manual_seed(seed_value)  # CUDA random seed
        torch.cuda.manual_seed_all(
            seed_value
        )  # CUDA random seed for all GPUs (if you are using multi-GPU)
        np.random.seed(seed_value)  # NumPy random seed
        random.seed(seed_value)  # Python random module
        torch.backends.cudnn.deterministic = True  # Deterministic algorithm
        torch.backends.cudnn.benchmark = (
            False  # if benchmark=True, deterministic will not be guaranteed
        )

    set_seed(args.seed)
    wandb.init(project="prompt-rec", name=args.wandb_name, config=args)

    if torch.cuda.is_available():
        if not args.cuda:
            print(
                now_time()
                + "WARNING: You have a CUDA device, so you should probably run with --cuda"
            )
    device = torch.device("cuda" if args.cuda else "cpu")
    print("Using", device)
    with open(os.path.join(args.data_dir, "user_to_embeddings.json"), "r") as f:
        user_to_embeddings = json.load(f)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # Prepare dataset
    exp_corpus = ExpDataLoader(args.data_dir)
    seq_corpus = SeqDataLoader(args.data_dir)
    nitem = len(seq_corpus.id2item)
    num_users = len(seq_corpus.id2user)
    all_iterator = TrainBatchify(
        exp_corpus.train,
        seq_corpus.user2items_positive,
        args.negative_num,
        nitem,
        tokenizer,
        args.exp_len,
        args.batch_size,
        args.data_dir,
    )
    exp_iterator = ExpBatchify(
        exp_corpus.valid, seq_corpus.user2id, tokenizer, args.exp_len, args.batch_size
    )
    seq_iterator = SeqBatchify(
        seq_corpus.user2items_positive,
        tokenizer,
        args.batch_size,
    )
    topn_iterator = TopNBatchify(
        seq_corpus.user2items_positive,
        seq_corpus.user2items_negative,
        args.negative_num,
        nitem,
        tokenizer,
        args.batch_size,
    )

    ###############################################################################
    # Build the model
    ###############################################################################

    model = Solomon.from_pretrained(args.model_name)
    # model.freeze_base_model_parameters()
    model.init_prompt(args.prompt_num, args.prompt_length, args.num_heads, device, user_to_embeddings)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ###############################################################################
    # Training code
    ###############################################################################

    def train():
        # Turn on training mode which enables dropout.
        model.train()
        text_loss = 0.0
        total_sample = 0
        while True:
            (
                task,
                users,
                source,
                source_mask,
                whole_word,
                target,
            ) = all_iterator.next_batch()

            task = task.to(device)
            source = source.to(device)  # (batch_size, seq_len)
            source_mask = source_mask.to(device)
            whole_word = whole_word.to(device)
            target = target.to(device)

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            optimizer.zero_grad()
            outputs = model(
                task_id=task,
                user_ids=users,
                input_ids=source,
                whole_word_ids=whole_word,
                attention_mask=source_mask,
                labels=target,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            batch_size = task.size(0)
            text_loss += batch_size * loss.item()
            total_sample += batch_size

            if (
                all_iterator.batch_index % args.log_interval == 0
                or all_iterator.batch_index % all_iterator.batch_num == 0
            ):
                cur_t_loss = text_loss / total_sample
                print(
                    now_time()
                    + "text loss {:4.4f} | {:5d}/{:5d} batches".format(
                        cur_t_loss, all_iterator.batch_index, all_iterator.batch_num
                    )
                )
                wandb.log({"batch_loss": cur_t_loss, "step": all_iterator.batch_index})
                text_loss = 0.0
                total_sample = 0
            if all_iterator.batch_index % all_iterator.batch_num == 0:
                break

    def evaluate(iterator):
        # Turn on evaluation mode which disables dropout.
        model.eval()
        text_loss = 0.0
        total_sample = 0
        with torch.no_grad():
            while True:
                (
                    task,
                    users,
                    source,
                    source_mask,
                    whole_word,
                    target,
                ) = iterator.next_batch_valid()
                task = task.to(device)
                source = source.to(device)  # (batch_size, seq_len)
                source_mask = source_mask.to(device)
                whole_word = whole_word.to(device)
                target = target.to(device)
                outputs = model(
                    task_id=task,
                    user_ids=users,
                    input_ids=source,
                    whole_word_ids=whole_word,
                    attention_mask=source_mask,
                    labels=target,
                )
                loss = outputs.loss

                batch_size = task.size(0)
                text_loss += batch_size * loss.item()
                total_sample += batch_size

                if iterator.step == iterator.total_step:
                    break
        return text_loss / total_sample

    output_dir = os.path.join(args.checkpoint, args.out_file)
    with open(output_dir, "wb") as f:
        torch.save(model, f)
    print(now_time() + "Start training")
    # Loop over epochs.
    best_val_loss = float("inf")
    endure_count = 0
    for epoch in range(1, args.epochs + 1):
        print(now_time() + "epoch {}".format(epoch))
        train()
        print(now_time() + "validation")
        exp_loss = evaluate(exp_iterator)
        print(now_time() + "explanation loss {:4.4f}".format(exp_loss))
        wandb.log({"exp_loss": exp_loss, "epoch": epoch})
        seq_loss = evaluate(seq_iterator)
        print(now_time() + "sequential loss {:4.4f}".format(seq_loss))
        wandb.log({"seq_loss": seq_loss, "epoch": epoch})
        topn_loss = evaluate(topn_iterator)
        print(now_time() + "top-N loss {:4.4f}".format(topn_loss))
        wandb.log({"topn_loss": topn_loss, "epoch": epoch})
        val_loss = (topn_loss + seq_loss + exp_loss) / 3
        print(now_time() + "total loss {:4.4f}".format(val_loss))
        wandb.log({"average_eval_loss": val_loss, "epoch": epoch})
        # Save the model if the validation loss is the best we've seen so far.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            with open(output_dir, "wb") as f:
                torch.save(model, f)
        else:
            endure_count += 1
            print(now_time() + "Endured {} time(s)".format(endure_count))
            if endure_count == args.endure_times:
                print(now_time() + "Cannot endure it anymore | Exiting from early stop")
                break
    wandb.finish()