"""
embedding_finetune.py

Fine-tunes the sentence transformer (all-MiniLM-L6-v2) on baseball-specific
sentence pairs.

Loss function evolution (documented for notebook):
    v1: CosineSimilarityLoss — minimizes MSE between cosine_sim(a,b) and
        target score. Collapsed immediately (train_loss → 0.01 in 2 epochs)
        because the model memorized score targets without learning generalizable
        similarity geometry. All post-training cosine sims dropped uniformly.

    v2: MultipleNegativesRankingLoss — treats every other item in the batch
        as a negative for each anchor. Forces the model to discriminate between
        similar and dissimilar pairs in a contrastive way. Standard loss for
        retrieval fine-tuning (used in SBERT, E5, etc.)
        Only uses positive pairs (score >= 0.8) — negatives come from the batch.

Why fine-tune:
    The pretrained model was trained on general English corpora.
    Baseball has domain-specific language where generic similarity
    scores are often wrong:
        - "walk-off win" and "late comeback" should be very close
        - "pitcher threw 12 Ks" and "team scored 11 runs" should be further
        - "blowout" and "close game" should be far apart despite
          sharing surface structure ("Team A beat Team B X-Y")

After fine-tuning, we measure retrieval quality improvement with
a held-out evaluation set — this is the key experimental result
for the notebook and pitch.

Architecture note:
    MultipleNegativesRankingLoss trains the model by:
        1. Encoding all (anchor, positive) pairs in the batch
        2. Computing cosine similarity matrix between all anchors and positives
        3. Treating diagonal as positive, off-diagonal as negatives
        4. Minimizing cross-entropy loss (i.e. the correct positive should
           have the highest similarity score for each anchor)

    The transformer weights are updated via backprop through the
    mean pooling and normalization layers all the way to the
    attention heads — this is real fine-tuning, not just a linear head.

Author: Parker Jackson
Course: CSCI 357 - AI and Neural Networks
"""

import os
import math
import torch
import numpy as np
import wandb
from collections import Counter
from typing import List, Tuple, Optional
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from src.mlb_rag.pair_generator import SentencePair, build_finetuning_dataset
from src.mlb_rag.embedder import MLBEmbedder, MLBVectorStore, build_vector_store, query_store
from src.mlb_rag.data_ingestion import get_mock_chunks


# ── Config ─────────────────────────────────────────────────────────────────────

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FINETUNED_MODEL_PATH = "/var/tmp/prj004/checkpoints/mlb-minilm-finetuned"

FINETUNE_CONFIG = {
    "base_model": BASE_MODEL,
    "num_epochs": 10,
    "batch_size": 32,
    "warmup_ratio": 0.1,
    "learning_rate": 5e-6,        # reduced from 2e-5 — prevents pretrained weight destruction
    "eval_split": 0.15,
    "seed": 42,
    "loss": "MultipleNegativesRankingLoss",
}


# ── Data Preparation ───────────────────────────────────────────────────────────

def pairs_to_input_examples(pairs: List[SentencePair]) -> List[InputExample]:
    """
    Convert SentencePairs to InputExample format with similarity scores.
    Used for the EmbeddingSimilarityEvaluator on the val set.
    """
    return [
        InputExample(texts=[p.sentence_a, p.sentence_b], label=p.score)
        for p in pairs
    ]


def pairs_to_ranking_examples(pairs: List[SentencePair]) -> List[InputExample]:
    """
    Convert only POSITIVE pairs to InputExample format for
    MultipleNegativesRankingLoss. Negatives come from the batch automatically.

    Only uses pairs with score >= 0.8 (paraphrase pairs).
    Hard negatives and true negatives are handled implicitly by the loss
    function — any two different items in the batch act as negatives for
    each other, which is more effective than explicit negative pairs.
    """
    positive_pairs = [p for p in pairs if p.score >= 0.8]
    return [
        InputExample(texts=[p.sentence_a, p.sentence_b])
        for p in positive_pairs
    ]


def train_val_split(
        pairs: List[SentencePair],
        val_ratio: float = 0.15,
        seed: int = 42
) -> Tuple[List[SentencePair], List[SentencePair]]:
    """Split pairs into train/val, stratified by pair_type."""
    from collections import defaultdict
    import random
    random.seed(seed)

    by_type = defaultdict(list)
    for p in pairs:
        by_type[p.pair_type].append(p)

    train, val = [], []
    for pair_type, type_pairs in by_type.items():
        random.shuffle(type_pairs)
        n_val = max(1, int(len(type_pairs) * val_ratio))
        val.extend(type_pairs[:n_val])
        train.extend(type_pairs[n_val:])

    random.shuffle(train)
    random.shuffle(val)
    return train, val


# ── Retrieval Evaluator ────────────────────────────────────────────────────────

class RetrievalEvaluator:
    """
    Measures retrieval quality before and after fine-tuning.

    This is the key experimental result for the notebook:
        - Build a small test corpus of MLB chunks
        - For each query, check if the top-1 retrieved chunk is correct
        - Compare pre-trained vs fine-tuned hit rates

    This is Precision@1 — the fraction of queries where the
    most similar chunk is actually the right one.
    """

    def __init__(self):
        self.test_cases = [
            {
                "query": "close one-run game late comeback",
                "relevant_keywords": ["close", "edge", "nail-biter", "tight", "one run"],
                "irrelevant_keywords": ["dominated", "cruised", "rout", "blowout"]
            },
            {
                "query": "dominant pitcher strikeouts",
                "relevant_keywords": ["striking out", "strikeouts", "overpowered", "punchouts"],
                "irrelevant_keywords": ["runs", "scored", "offense", "blowout"]
            },
            {
                "query": "extra innings walk off",
                "relevant_keywords": ["extra", "innings", "extras", "thriller"],
                "irrelevant_keywords": ["dominated", "standings", "division"]
            },
            {
                "query": "division standings leader",
                "relevant_keywords": ["standings", "lead", "division", "record", "atop"],
                "irrelevant_keywords": ["defeated", "strikeout", "innings"]
            },
            {
                "query": "offensive explosion runs scored",
                "relevant_keywords": ["erupted", "runs", "blowout", "dominated", "rout"],
                "irrelevant_keywords": ["strikeout", "standings", "extra innings"]
            },
        ]

    def evaluate(self, embedder: MLBEmbedder, chunks, label: str = "") -> float:
        """
        Compute Precision@1 over test queries.
        Returns fraction of queries where top result contains relevant keywords.
        """
        store = build_vector_store(chunks, embedder=embedder, save=False)
        hits = 0
        details = []

        for tc in self.test_cases:
            results = query_store(tc["query"], store, embedder, top_k=1)
            if not results:
                details.append((tc["query"], "NO RESULT", False))
                continue
            top_chunk, score = results[0]
            text_lower = top_chunk.text.lower()

            relevant_hit = any(kw in text_lower for kw in tc["relevant_keywords"])
            irrelevant_hit = any(kw in text_lower for kw in tc["irrelevant_keywords"])
            correct = relevant_hit and not irrelevant_hit

            if correct:
                hits += 1
            details.append((tc["query"], top_chunk.text[:60], correct))

        precision_at_1 = hits / len(self.test_cases)
        print(f"  [{label}] Precision@1: {precision_at_1:.2f} ({hits}/{len(self.test_cases)} queries)")
        for query, result, correct in details:
            mark = "✓" if correct else "✗"
            print(f"    {mark} '{query[:35]}' → '{result[:50]}'")
        return precision_at_1


# ── Fine-Tuning Pipeline ───────────────────────────────────────────────────────

def finetune_embedding_model(
        pairs: List[SentencePair] = None,
        config: dict = None,
        use_wandb: bool = True,
        chunks_for_eval=None,
) -> SentenceTransformer:
    """
    Fine-tune all-MiniLM-L6-v2 on baseball sentence pairs using
    MultipleNegativesRankingLoss with per-epoch W&B logging.

    Args:
        pairs: Training pairs. Generates default dataset if None.
        config: Hyperparameter config. Uses FINETUNE_CONFIG if None.
        use_wandb: Whether to log to W&B.
        chunks_for_eval: MLBChunks for retrieval evaluation.

    Returns:
        Fine-tuned SentenceTransformer model.
    """
    if config is None:
        config = FINETUNE_CONFIG.copy()
    if pairs is None:
        pairs = build_finetuning_dataset()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[FineTune] Device: {device}")
    print(f"[FineTune] Base model: {config['base_model']}")
    print(f"[FineTune] Loss: {config.get('loss', 'MultipleNegativesRankingLoss')}")
    print(f"[FineTune] Total pairs: {len(pairs)}")

    # ── Dataset stats ──────────────────────────────────────────────────────────
    type_counts = Counter(p.pair_type for p in pairs)
    n_positive = sum(1 for p in pairs if p.score >= 0.8)
    n_hard_neg = sum(1 for p in pairs if 0.1 <= p.score < 0.4)
    n_true_neg = sum(1 for p in pairs if p.score < 0.1)

    print(f"\n[FineTune] Dataset composition:")
    print(f"  Positive pairs (score >= 0.8): {n_positive}")
    print(f"  Hard negatives (0.1-0.4):      {n_hard_neg}")
    print(f"  True negatives (< 0.1):        {n_true_neg}")

    # ── W&B init ───────────────────────────────────────────────────────────────
    run = None
    if use_wandb:
        run = wandb.init(
            project="mlb-rag",
            name=f"mnrl-ep{config['num_epochs']}-lr{config['learning_rate']:.0e}-bs{config['batch_size']}",
            config=config
        )
        # Log dataset stats immediately so every run has them
        run.log({
            "data/total_pairs": len(pairs),
            "data/n_positive": n_positive,
            "data/n_hard_neg": n_hard_neg,
            "data/n_true_neg": n_true_neg,
            **{f"data/type_{k}": v for k, v in type_counts.items()},
        })

    # ── Baseline evaluation ────────────────────────────────────────────────────
    evaluator = RetrievalEvaluator()
    if chunks_for_eval is None:
        chunks_for_eval = get_mock_chunks()

    print("\n[FineTune] Baseline retrieval quality (pretrained):")
    baseline_embedder = MLBEmbedder(model_name=config["base_model"], device=device)
    baseline_score = evaluator.evaluate(baseline_embedder, chunks_for_eval, label="pretrained")

    if run:
        run.log({"retrieval/precision_at_1_pretrained": baseline_score, "epoch": 0})

    # ── Data preparation ───────────────────────────────────────────────────────
    train_pairs, val_pairs = train_val_split(pairs, val_ratio=config["eval_split"])
    print(f"\n[FineTune] Train: {len(train_pairs)} pairs, Val: {len(val_pairs)} pairs")

    # For training: only positive pairs (MNRL handles negatives from batch)
    train_examples = pairs_to_ranking_examples(train_pairs)
    # For validation scoring: all pairs with scores (Spearman correlation)
    val_examples = pairs_to_input_examples(val_pairs)

    print(f"[FineTune] Positive training examples: {len(train_examples)}")

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=config["batch_size"]
    )

    # ── Model + Loss ───────────────────────────────────────────────────────────
    model = SentenceTransformer(config["base_model"], device=device)

    # MultipleNegativesRankingLoss:
    # - Takes (anchor, positive) pairs
    # - Treats all other positives in the batch as negatives for each anchor
    # - Minimizes cross-entropy over cosine similarity matrix
    # - Effective batch size = batch_size^2 negative pairs
    train_loss = MultipleNegativesRankingLoss(model)

    # Val evaluator: Spearman correlation between predicted and target scores
    val_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        val_examples,
        name="mlb-val"
    )

    # ── W&B callback (per-epoch logging) ──────────────────────────────────────
    epoch_counter = [0]  # mutable container for closure

    def wandb_callback(val_score, epoch, steps):
        """Called by sentence-transformers after each epoch evaluation."""
        epoch_counter[0] = epoch

        log_dict = {
            "epoch": epoch,
            "val/spearman": val_score,
        }

        # Retrieval eval every 2 epochs (more expensive)
        if epoch % 2 == 0 or epoch == config["num_epochs"]:
            ft_embedder = MLBEmbedder(
                model_name=FINETUNED_MODEL_PATH,
                device=device
            )
            p_at_1 = evaluator.evaluate(
                ft_embedder, chunks_for_eval,
                label=f"epoch_{epoch}"
            )
            log_dict["retrieval/precision_at_1"] = p_at_1
            log_dict["retrieval/improvement_vs_baseline"] = p_at_1 - baseline_score

        if run:
            run.log(log_dict)

        print(f"  Epoch {epoch:2d} | val_spearman={val_score:.4f}"
              + (f" | P@1={log_dict.get('retrieval/precision_at_1', '—')}"
                 if "retrieval/precision_at_1" in log_dict else ""))

    # ── Training ───────────────────────────────────────────────────────────────
    warmup_steps = math.ceil(
        len(train_dataloader) * config["num_epochs"] * config["warmup_ratio"]
    )
    print(f"[FineTune] Warmup steps: {warmup_steps}")
    print(f"[FineTune] Steps per epoch: {len(train_dataloader)}")
    print(f"[FineTune] Starting fine-tuning for {config['num_epochs']} epochs...\n")

    os.makedirs(FINETUNED_MODEL_PATH, exist_ok=True)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=val_evaluator,
        epochs=config["num_epochs"],
        warmup_steps=warmup_steps,
        optimizer_params={"lr": config["learning_rate"]},
        output_path=FINETUNED_MODEL_PATH,
        save_best_model=True,
        show_progress_bar=True,
        callback=wandb_callback,
        evaluation_steps=len(train_dataloader),  # eval every epoch
    )

    # ── Final evaluation ───────────────────────────────────────────────────────
    print("\n[FineTune] Final retrieval quality (best checkpoint):")
    finetuned_embedder = MLBEmbedder(model_name=FINETUNED_MODEL_PATH, device=device)
    finetuned_score = evaluator.evaluate(
        finetuned_embedder, chunks_for_eval, label="finetuned_final"
    )

    improvement = finetuned_score - baseline_score
    print(f"\n[FineTune] Retrieval improvement: {improvement:+.2f} "
          f"({baseline_score:.2f} → {finetuned_score:.2f})")

    if run:
        run.log({
            "retrieval/precision_at_1_finetuned": finetuned_score,
            "retrieval/final_improvement": improvement,
        })
        run.finish()

    return model


# ── Comparison Utility ─────────────────────────────────────────────────────────

def compare_embeddings(
        query: str,
        sentences: List[str],
        pretrained_path: str = BASE_MODEL,
        finetuned_path: str = FINETUNED_MODEL_PATH,
        device: str = None
) -> None:
    """
    Visual comparison of cosine similarities before and after fine-tuning.
    Great for notebook demonstrations.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pre_model = MLBEmbedder(model_name=pretrained_path, device=device)
    ft_model = MLBEmbedder(model_name=finetuned_path, device=device)

    q_pre = pre_model.embed([query], show_progress=False)
    q_ft = ft_model.embed([query], show_progress=False)
    s_pre = pre_model.embed(sentences, show_progress=False)
    s_ft = ft_model.embed(sentences, show_progress=False)

    sim_pre = (q_pre @ s_pre.T)[0]
    sim_ft = (q_ft @ s_ft.T)[0]

    print(f"\nQuery: '{query}'")
    print(f"{'Sentence':<55} {'Pretrained':>12} {'Finetuned':>12} {'Delta':>8}")
    print("-" * 90)
    for sent, sp, sf in zip(sentences, sim_pre, sim_ft):
        delta = sf - sp
        marker = " ↑" if delta > 0.05 else (" ↓" if delta < -0.05 else "")
        print(f"{sent[:54]:<55} {sp:>12.3f} {sf:>12.3f} {delta:>+8.3f}{marker}")


# ── Quick Test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.mlb_rag.data_ingestion import ingest_mlb_data, get_mock_chunks

    pairs = build_finetuning_dataset(
        n_paraphrase=600,
        n_cross_type=300,
        n_hard_neg=500,
        n_true_neg=300,
    )

    # Use real chunks if available
    try:
        chunks = ingest_mlb_data(days_back=3)
        if not chunks:
            chunks = get_mock_chunks()
    except Exception:
        chunks = get_mock_chunks()

    model = finetune_embedding_model(
        pairs=pairs,
        use_wandb=True,
        chunks_for_eval=chunks,
        config={**FINETUNE_CONFIG, "num_epochs": 10, "learning_rate": 5e-6}
    )

    # Show before/after comparison
    compare_embeddings(
        query="close one-run game late comeback",
        sentences=[
            "Yankees edged Red Sox by 1 run, winning 3-2.",
            "Dodgers dominated Giants 12-1 in a blowout.",
            "Braves outlasted Mets 4-3 in 11 innings.",
            "AL East standings: Yankees lead at 88-74.",
            "Pitcher struck out 14 batters in dominant outing.",
        ]
    )