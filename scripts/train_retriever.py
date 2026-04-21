"""Fine-tune the bi-encoder retriever on FinQA gold_inds evidence.

Each FinQA train example has `gold_inds`: a dict mapping evidence keys
("text_3", "table_2", ...) to their text. We unfold this into one positive
pair per evidence span — so a question with 3 evidence spans yields 3 train
pairs. This trains the retriever to find the *specific* evidence chunk inside
the doc, not just the doc as a whole, which matches our chunk-level inference
architecture.

Loss: MultipleNegativesRankingLoss (in-batch negatives).
The BGE asymmetric query prefix is applied on the anchor side only, mirroring
what `Embedder.embed_query` does at inference time.

Output: a SentenceTransformer model directory plug-compatible with
`embedding.model_name` in config.yaml.
"""

import argparse
import os
import sys

import yaml
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.loader import load_finqa_file
from src.indexing.embedder import BGE_QUERY_PREFIX


def build_pairs_from_gold_inds(docs, min_chars: int = 20):
    """Unfold each (question, gold_inds) into one pair per evidence span.

    Falls back to the linearized table when gold_inds is empty/too sparse,
    so we don't drop those examples from training.
    """
    anchors, positives = [], []
    n_docs_with_evidence = 0
    n_docs_fallback = 0
    n_docs_skipped = 0

    for doc in docs:
        if not doc.question:
            n_docs_skipped += 1
            continue

        anchor = BGE_QUERY_PREFIX + doc.question
        evidence_texts = [
            v.strip() for v in doc.gold_evidence.values()
            if isinstance(v, str) and len(v.strip()) >= min_chars
        ]

        if evidence_texts:
            for ev in evidence_texts:
                anchors.append(anchor)
                positives.append(ev)
            n_docs_with_evidence += 1
        elif doc.table_linearized and len(doc.table_linearized) >= min_chars:
            anchors.append(anchor)
            positives.append(doc.table_linearized)
            n_docs_fallback += 1
        else:
            n_docs_skipped += 1

    print(f"  docs with gold_inds evidence: {n_docs_with_evidence}")
    print(f"  docs falling back to table:   {n_docs_fallback}")
    print(f"  docs skipped (no signal):     {n_docs_skipped}")
    print(f"  total positive pairs:         {len(anchors)}")
    return anchors, positives


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--output-dir", default="models/bge-finqa-base")
    ap.add_argument("--base-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-seq-length", type=int, default=384)
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_path = os.path.join(config["data"]["raw_dir"], config["data"]["train_file"])
    print(f"=== Loading train data: {train_path} ===")
    train_docs = load_finqa_file(train_path)
    print(f"train docs: {len(train_docs)}")

    print("\n=== Building (question, gold_evidence_span) pairs ===")
    anchors, positives = build_pairs_from_gold_inds(train_docs)

    n_eval = min(300, len(anchors) // 20)
    train_dataset = Dataset.from_dict({
        "anchor": anchors[:-n_eval],
        "positive": positives[:-n_eval],
    })
    eval_dataset = Dataset.from_dict({
        "anchor": anchors[-n_eval:],
        "positive": positives[-n_eval:],
    })
    print(f"train pairs: {len(train_dataset)}, held-out eval pairs: {len(eval_dataset)}")

    print(f"\n=== Loading base model: {args.base_model} ===")
    model = SentenceTransformer(args.base_model)
    model.max_seq_length = args.max_seq_length

    loss = MultipleNegativesRankingLoss(model)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        fp16=True,
        # In-batch negatives only work if the same positive doesn't appear
        # twice in a batch — NO_DUPLICATES enforces this.
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=20,
        report_to=[],
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
    )
    trainer.train()

    print(f"\n=== Saving fine-tuned model to {args.output_dir} ===")
    model.save_pretrained(args.output_dir)
    print("done.")
    print("\nNext steps:")
    print(f'  1) Update configs/config.yaml -> embedding.model_name: "{args.output_dir}"')
    print("  2) Rebuild index:        python -m src.indexing.build_index")
    print("  3) Re-measure recall:    python scripts/eval_retrieval.py --split dev --max-examples 200")


if __name__ == "__main__":
    main()
