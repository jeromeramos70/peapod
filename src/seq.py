import argparse
import json
import os
import random

import torch
from tqdm import tqdm
from transformers import T5Tokenizer

from utils import SeqBatchify, SeqDataLoader, evaluate_hr, evaluate_ndcg, now_time

parser = argparse.ArgumentParser(description="POD (PrOmpt Distillation)")
parser.add_argument(
    "--data_dir", type=str, default=None, help="directory for loading the data"
)
parser.add_argument(
    "--model_version",
    type=int,
    default=0,
    help="1: t5-base; 2: t5-large; 3: t5-3b; 4: t5-11b; otherwise: t5-small",
)
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--cuda", action="store_true", help="use CUDA")
parser.add_argument(
    "--checkpoint", type=str, default="./pod/", help="directory to load the final model"
)
parser.add_argument("--num_beams", type=int, default=20, help="number of beams")
parser.add_argument("--top_n", type=int, default=10, help="number of items to predict")
parser.add_argument("--model_path", type=str, required=True, help="saved model")
args = parser.parse_args()


model_version = "t5-small"

print("-" * 40 + "ARGUMENTS" + "-" * 40)
for arg in vars(args):
    print("{:40} {}".format(arg, getattr(args, arg)))
print("-" * 40 + "ARGUMENTS" + "-" * 40)

if torch.cuda.is_available():
    if not args.cuda:
        print(
            now_time()
            + "WARNING: You have a CUDA device, so you should probably run with --cuda"
        )
device = torch.device("cuda" if args.cuda else "cpu")

if not os.path.exists(args.checkpoint):
    os.makedirs(args.checkpoint)
model_path = os.path.join(args.checkpoint, args.model_path)

###############################################################################
# Load data
###############################################################################

print(now_time() + "Loading data")
tokenizer = T5Tokenizer.from_pretrained("t5-small")
seq_corpus = SeqDataLoader(args.data_dir)
nitem = len(seq_corpus.id2item)
seq_iterator = SeqBatchify(seq_corpus.user2items_positive, tokenizer, args.batch_size)

###############################################################################
# Test the model
###############################################################################


def generate():
    # Turn on evaluation mode which disables dropout.
    model.eval()
    idss_predict = []
    with torch.no_grad():
        while True:
            task, users, source, source_mask, whole_word, _ = (
                seq_iterator.next_batch_test()
            )
            task = task.to(device)  # (batch_size,)
            source = source.to(device)  # (batch_size, seq_len)
            source_mask = source_mask.to(device)
            whole_word = whole_word.to(device)

            beam_outputs = model.beam_search(
                task_id=task,
                user_ids=users,
                input_ids=source,
                whole_word_ids=whole_word,
                attention_mask=source_mask,
                num_beams=args.num_beams,
                num_return_sequences=args.top_n,
            )

            output_tensor = beam_outputs.view(task.size(0), args.top_n, -1)
            for i in range(task.size(0)):
                results = tokenizer.batch_decode(
                    output_tensor[i], skip_special_tokens=True
                )
                idss_predict.append(results)

            if seq_iterator.step == seq_iterator.total_step:
                break
    return idss_predict


# Load the best saved model.
with open(model_path, "rb") as f:
    model = torch.load(f).to(device)

# Run on test data.
print(now_time() + "Generating recommendations")
idss_predicted = generate()
print(now_time() + "Evaluation")
user2item_test = {}
for user, item_list in seq_corpus.user2items_positive.items():
    user2item_test[user] = [int(item_list[-1])]
user2rank_list = {}
for predictions, user in zip(idss_predicted, seq_iterator.user_list):
    prediction_list = []
    for p in predictions:
        try:
            prediction_list.append(
                int(p.split(" ")[0])
            )  # use the id before white space
        except:
            prediction_list.append(
                random.randint(1, nitem)
            )  # randomly generate a recommendation
    user2rank_list[user] = prediction_list

top_ns = [1]
if args.top_n >= 5:
    for i in range(1, (args.top_n // 5) + 1):
        top_ns.append(i * 5)
for top_n in top_ns:
    hr = evaluate_hr(user2item_test, user2rank_list, top_n)
    print(now_time() + "HR@{} {:7.4f}".format(top_n, hr))
    ndcg = evaluate_ndcg(user2item_test, user2rank_list, top_n)
    print(now_time() + "NDCG@{} {:7.4f}".format(top_n, ndcg))
