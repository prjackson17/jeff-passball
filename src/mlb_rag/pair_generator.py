"""
pair_generator.py

Generates sentence pairs for fine-tuning the sentence transformer
on baseball-specific language.

We use CosineSimilarityLoss which requires:
    (sentence_A, sentence_B, similarity_score)
    where similarity_score is in [0, 1]
        1.0 = semantically identical
        0.5 = loosely related
        0.0 = unrelated

Pair generation strategy:
    POSITIVE pairs (score ~0.8-1.0):
        - Two recaps of games with similar outcomes (both close, both blowouts)
        - Same team winning described different ways
        - Game recap + standings update for same team

    HARD NEGATIVE pairs (score ~0.1-0.3):
        - Close game recap vs blowout recap
        - Pitching-dominant game vs offensive explosion
        - Two different teams' outcomes on same day

    TRUE NEGATIVE pairs (score 0.0):
        - Game recap vs standings chunk from different division
        - Randomly sampled unrelated chunks

Why this matters:
    The pretrained model treats "walk-off win" and "late comeback"
    as moderately similar. After fine-tuning on baseball pairs,
    these should be very close in embedding space. Meanwhile
    "pitcher threw 12 Ks" and "team scored 11 runs" should be
    further apart than the generic model places them.

Author: Parker Jackson
Course: CSCI 357 - AI and Neural Networks
"""

import random
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from src.mlb_rag.data_ingestion import MLBChunk, ingest_mlb_data, get_mock_chunks
from src.mlb_rag.auto_labeler import label_game_with_reasons, RULES


# ── Data Structure ─────────────────────────────────────────────────────────────

@dataclass
class SentencePair:
    """
    A single training pair for CosineSimilarityLoss fine-tuning.

    sentence_a, sentence_b: the two text strings to compare
    score: float in [0, 1] — how semantically similar they should be
    pair_type: label for analysis/debugging
    """
    sentence_a: str
    sentence_b: str
    score: float
    pair_type: str = "unknown"


# ── Text Templates ─────────────────────────────────────────────────────────────
# These generate paraphrases of the same semantic content.
# The model should learn these map to the same region of embedding space.

def _close_game_templates(winner: str, loser: str, w_score: int, l_score: int) -> List[str]:
    """Multiple ways to describe a close game."""
    margin = w_score - l_score
    return [
        f"The {winner} defeated the {loser} {w_score}-{l_score} in a close game.",
        f"{winner} edged out {loser} by {margin} run{'s' if margin > 1 else ''}, winning {w_score}-{l_score}.",
        f"A tight contest ended with {winner} on top, {w_score} to {l_score} over {loser}.",
        f"{winner} held off {loser} in a {w_score}-{l_score} nail-biter.",
    ]


def _blowout_templates(winner: str, loser: str, w_score: int, l_score: int) -> List[str]:
    """Multiple ways to describe a blowout."""
    return [
        f"The {winner} dominated the {loser} {w_score}-{l_score}.",
        f"{winner} cruised to a {w_score}-{l_score} victory over {loser}.",
        f"A lopsided affair saw {winner} rout {loser} by a score of {w_score}-{l_score}.",
        f"{winner} had little trouble with {loser}, winning comfortably {w_score}-{l_score}.",
    ]


def _extra_innings_templates(winner: str, loser: str, w_score: int, l_score: int, innings: int) -> List[str]:
    """Multiple ways to describe an extra innings game."""
    return [
        f"The {winner} outlasted the {loser} {w_score}-{l_score} in {innings} innings.",
        f"It took {innings} innings but {winner} finally pulled ahead of {loser}, {w_score}-{l_score}.",
        f"An extra-inning thriller ended with {winner} defeating {loser} {w_score}-{l_score}.",
        f"{winner} and {loser} went the distance before {winner} won {w_score}-{l_score} in extras.",
    ]


def _dominant_pitching_templates(team: str, opponent: str, strikeouts: int) -> List[str]:
    """Multiple ways to describe a dominant pitching performance."""
    return [
        f"The {team} pitcher was dominant, striking out {strikeouts} batters against {opponent}.",
        f"{team}'s starter overpowered {opponent}, recording {strikeouts} strikeouts.",
        f"A stellar pitching performance from {team} saw {strikeouts} punchouts vs {opponent}.",
        f"{team} held {opponent} in check with {strikeouts} strikeouts from their starter.",
    ]


def _standings_templates(team: str, wins: int, losses: int, division: str) -> List[str]:
    """Multiple ways to describe standings position."""
    pct = wins / (wins + losses) if (wins + losses) > 0 else 0
    return [
        f"The {team} lead the {division} with a {wins}-{losses} record.",
        f"{team} are atop the {division} standings at {wins}-{losses}.",
        f"With a {wins}-{losses} mark, {team} sit first in the {division}.",
        f"{team} hold the top spot in the {division}, {wins} wins and {losses} losses.",
    ]


# ── Pair Generators ────────────────────────────────────────────────────────────

def generate_paraphrase_pairs(n: int = 500) -> List[SentencePair]:
    """
    POSITIVE pairs: different phrasings of the same semantic content.
    Score: 0.9 (not 1.0 — they're paraphrases, not identical)

    Strategy: for each game recap chunk, generate multiple template
    descriptions and pair them together.
    """
    pairs = []
    teams = [
        ("Yankees", "Red Sox"), ("Dodgers", "Giants"), ("Braves", "Mets"),
        ("Astros", "Rangers"), ("Phillies", "Nationals"), ("Cubs", "Cardinals"),
        ("Padres", "Diamondbacks"), ("Mariners", "Angels"), ("Orioles", "Blue Jays"),
        ("Guardians", "Tigers"), ("Twins", "White Sox"), ("Brewers", "Reds"),
    ]
    divisions = ["AL East", "AL Central", "AL West", "NL East", "NL Central", "NL West"]

    for _ in range(n // 4):
        winner, loser = random.choice(teams)
        game_type = random.choice(["close", "blowout", "extra", "pitching"])

        if game_type == "close":
            w_score = random.randint(2, 5)
            l_score = w_score - random.randint(1, 2)
            l_score = max(0, l_score)
            templates = _close_game_templates(winner, loser, w_score, l_score)

        elif game_type == "blowout":
            w_score = random.randint(9, 15)
            l_score = random.randint(0, 3)
            templates = _blowout_templates(winner, loser, w_score, l_score)

        elif game_type == "extra":
            w_score = random.randint(3, 7)
            l_score = w_score - 1
            innings = random.randint(10, 13)
            templates = _extra_innings_templates(winner, loser, w_score, l_score, innings)

        else:  # pitching
            so = random.randint(10, 16)
            templates = _dominant_pitching_templates(winner, loser, so)

        # Pair each template with every other template for this game type
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                pairs.append(SentencePair(
                    sentence_a=templates[i],
                    sentence_b=templates[j],
                    score=0.9,
                    pair_type=f"paraphrase_{game_type}"
                ))

    # Standings paraphrases
    for _ in range(n // 8):
        team, _ = random.choice(teams)
        wins = random.randint(60, 100)
        losses = random.randint(40, 80)
        division = random.choice(divisions)
        templates = _standings_templates(team, wins, losses, division)
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                pairs.append(SentencePair(
                    sentence_a=templates[i],
                    sentence_b=templates[j],
                    score=0.85,
                    pair_type="paraphrase_standings"
                ))

    random.shuffle(pairs)
    return pairs[:n]


def generate_cross_type_pairs(n: int = 300) -> List[SentencePair]:
    """
    RELATED pairs: different game types for same teams — loosely related.
    Score: 0.4-0.6

    A game recap and a standings entry for the same team are related
    but not semantically equivalent.
    """
    pairs = []
    teams = [
        ("Yankees", "AL East"), ("Dodgers", "NL West"), ("Braves", "NL East"),
        ("Astros", "AL West"), ("Phillies", "NL East"), ("Cubs", "NL Central"),
    ]

    for _ in range(n):
        team, division = random.choice(teams)
        opponent = random.choice([t for t, _ in teams if t != team])[0]

        w_score = random.randint(2, 8)
        l_score = random.randint(0, w_score - 1)
        wins = random.randint(60, 95)
        losses = random.randint(40, 80)

        recap = f"The {team} defeated the {opponent} {w_score}-{l_score}."
        standings = f"The {team} hold a {wins}-{losses} record in the {division}."

        pairs.append(SentencePair(
            sentence_a=recap,
            sentence_b=standings,
            score=0.5,
            pair_type="cross_type_related"
        ))

    return pairs


def generate_hard_negative_pairs(n: int = 400) -> List[SentencePair]:
    """
    HARD NEGATIVE pairs: superficially similar but semantically different.
    Score: 0.1-0.2

    These are the most valuable training examples — they teach the model
    that a blowout and a close game are NOT the same even though both
    are game recaps with similar surface structure.
    """
    pairs = []
    matchups = [
        ("Yankees", "Red Sox"), ("Dodgers", "Giants"),
        ("Braves", "Mets"), ("Astros", "Rangers"),
    ]

    for _ in range(n // 3):
        # Close game vs blowout — same teams, very different games
        winner, loser = random.choice(matchups)
        close_score_w = random.randint(2, 4)
        close_score_l = close_score_w - 1
        blowout_w = random.randint(10, 15)
        blowout_l = random.randint(0, 2)

        close = random.choice(_close_game_templates(winner, loser, close_score_w, close_score_l))
        blowout = random.choice(_blowout_templates(winner, loser, blowout_w, blowout_l))

        pairs.append(SentencePair(
            sentence_a=close,
            sentence_b=blowout,
            score=0.15,
            pair_type="hard_neg_close_vs_blowout"
        ))

    for _ in range(n // 3):
        # Pitching dominant vs offensive explosion — both notable, opposite reasons
        team_a, opp_a = random.choice(matchups)
        team_b, opp_b = random.choice([m for m in matchups if m != (team_a, opp_a)])
        so = random.randint(11, 16)
        runs = random.randint(12, 18)

        pitching = random.choice(_dominant_pitching_templates(team_a, opp_a, so))
        offense = f"The {team_b} erupted for {runs} runs in a blowout of {opp_b}."

        pairs.append(SentencePair(
            sentence_a=pitching,
            sentence_b=offense,
            score=0.1,
            pair_type="hard_neg_pitching_vs_offense"
        ))

    for _ in range(n // 3):
        # Extra innings game vs routine win — both wins, very different contexts
        winner, loser = random.choice(matchups)
        other_winner, other_loser = random.choice([m for m in matchups if m[0] != winner])
        innings = random.randint(10, 13)
        w = random.randint(3, 6)
        l = w - 1
        routine_w = random.randint(5, 8)
        routine_l = random.randint(1, 3)

        extra = random.choice(_extra_innings_templates(winner, loser, w, l, innings))
        routine = f"The {other_winner} beat the {other_loser} {routine_w}-{routine_l}."

        pairs.append(SentencePair(
            sentence_a=extra,
            sentence_b=routine,
            score=0.2,
            pair_type="hard_neg_extra_vs_routine"
        ))

    random.shuffle(pairs)
    return pairs[:n]


def generate_true_negative_pairs(n: int = 300) -> List[SentencePair]:
    """
    TRUE NEGATIVE pairs: completely unrelated content.
    Score: 0.0

    Game recaps paired with standings from completely different contexts.
    """
    pairs = []
    game_teams = [
        ("Yankees", "Red Sox", 7, 3),
        ("Dodgers", "Giants", 5, 2),
        ("Braves", "Mets", 4, 1),
        ("Astros", "Rangers", 6, 0),
        ("Cubs", "Cardinals", 3, 2),
    ]
    standings_entries = [
        ("Mariners", 88, 74, "AL West"),
        ("Brewers", 92, 70, "NL Central"),
        ("Orioles", 101, 61, "AL East"),
        ("Padres", 78, 84, "NL West"),
        ("Guardians", 76, 86, "AL Central"),
    ]

    for _ in range(n):
        winner, loser, w_score, l_score = random.choice(game_teams)
        team, wins, losses, division = random.choice(standings_entries)

        # Make sure they're not the same team
        if team in [winner, loser]:
            continue

        recap = random.choice(_close_game_templates(winner, loser, w_score, l_score))
        standings = random.choice(_standings_templates(team, wins, losses, division))

        pairs.append(SentencePair(
            sentence_a=recap,
            sentence_b=standings,
            score=0.0,
            pair_type="true_negative"
        ))

    return pairs[:n]


# ── Real-Data Pair Generation ──────────────────────────────────────────────────

def _rule_sig_similarity(sig_a: frozenset, sig_b: frozenset) -> float:
    """Jaccard-based similarity score for two rule-signature sets."""
    if not sig_a and not sig_b:
        return 0.5   # both routine — loosely related
    if not sig_a or not sig_b:
        return 0.1   # one notable, one routine — hard negative
    return 0.1 + 0.8 * (len(sig_a & sig_b) / len(sig_a | sig_b))


def build_real_data_pairs(
    features,                    # List[GameFeatures] with recap_text populated
    max_pairs: int = 2000,
    seed: int = 42,
) -> List[SentencePair]:
    """
    Build sentence pairs from real historical game recap text, using
    auto-labeler rule signatures to assign similarity scores.

    Positive pairs (score ~0.9):  two games with the same fired rules
    Hard negatives (score ~0.1):  two notable games with disjoint rule signatures
    True negatives (score  0.0):  one notable game vs one routine game
    """
    from collections import defaultdict
    from src.mlb_rag.auto_labeler import label_game_with_reasons

    random.seed(seed)

    with_text = [f for f in features if f.recap_text]
    if not with_text:
        print("[PairGen] No GameFeatures with recap_text — skipping real pairs.")
        return []

    # Build signature groups
    sig_groups: dict = defaultdict(list)
    routine: list = []
    for f in with_text:
        label, fired = label_game_with_reasons(f)
        sig = frozenset(fired)
        if label == 1:
            sig_groups[sig].append(f)
        else:
            routine.append(f)

    notable_all = [f for group in sig_groups.values() for f in group]
    sig_list = list(sig_groups.keys())

    pairs: List[SentencePair] = []

    # Positive pairs: same signature
    for sig, group in sig_groups.items():
        if len(group) < 2:
            continue
        shuffled = group[:]
        random.shuffle(shuffled)
        cap = min(len(shuffled) - 1, 50)
        for i in range(cap):
            pairs.append(SentencePair(
                sentence_a=shuffled[i].recap_text,
                sentence_b=shuffled[i + 1].recap_text,
                score=0.9,
                pair_type="real_same_signature",
            ))

    # Hard negatives: disjoint notable signatures
    attempts = 0
    hard_neg_target = max_pairs // 3
    while len([p for p in pairs if p.pair_type == "real_same_signature"]) + \
          len([p for p in pairs if p.pair_type == "real_disjoint_signature"]) < \
          hard_neg_target + len([p for p in pairs if p.pair_type == "real_same_signature"]) \
          and attempts < hard_neg_target * 5 and len(sig_list) >= 2:
        s1, s2 = random.sample(sig_list, 2)
        if s1 & s2:   # not disjoint — skip
            attempts += 1
            continue
        g1 = random.choice(sig_groups[s1])
        g2 = random.choice(sig_groups[s2])
        pairs.append(SentencePair(
            sentence_a=g1.recap_text,
            sentence_b=g2.recap_text,
            score=0.1,
            pair_type="real_disjoint_signature",
        ))
        attempts += 1

    # True negatives: notable vs routine
    true_neg_target = max_pairs // 4
    for _ in range(true_neg_target):
        if not notable_all or not routine:
            break
        pairs.append(SentencePair(
            sentence_a=random.choice(notable_all).recap_text,
            sentence_b=random.choice(routine).recap_text,
            score=0.0,
            pair_type="real_notable_vs_routine",
        ))

    random.shuffle(pairs)
    result = pairs[:max_pairs]
    print(f"[PairGen] Generated {len(result)} real-data pairs "
          f"({len(sig_groups)} rule signatures, {len(notable_all)} notable / {len(routine)} routine games)")
    return result


# ── Main Dataset Builder ───────────────────────────────────────────────────────

def build_finetuning_dataset(
    n_paraphrase: int = 600,
    n_cross_type: int = 300,
    n_hard_neg: int = 500,
    n_true_neg: int = 300,
    seed: int = 42,
    real_features=None,          # Optional[List[GameFeatures]]
    n_real_pairs: int = 2000,
) -> List[SentencePair]:
    """
    Build the full fine-tuning dataset by combining all pair types.

    Args:
        n_paraphrase: Number of positive paraphrase pairs (synthetic)
        n_cross_type: Number of loosely related pairs (synthetic)
        n_hard_neg: Number of hard negative pairs (synthetic)
        n_true_neg: Number of true negative pairs (synthetic)
        seed: Random seed for reproducibility
        real_features: Optional list of GameFeatures with recap_text for real pairs.
                       Pass load_features_as_objects() result here.
        n_real_pairs: Max number of real-data pairs to generate.

    Returns:
        Shuffled list of SentencePairs ready for fine-tuning
    """
    random.seed(seed)
    np.random.seed(seed)

    print("[PairGen] Generating paraphrase pairs...")
    paraphrase = generate_paraphrase_pairs(n=n_paraphrase)
    print(f"  Generated {len(paraphrase)} paraphrase pairs")

    print("[PairGen] Generating cross-type pairs...")
    cross = generate_cross_type_pairs(n=n_cross_type)
    print(f"  Generated {len(cross)} cross-type pairs")

    print("[PairGen] Generating hard negative pairs...")
    hard_neg = generate_hard_negative_pairs(n=n_hard_neg)
    print(f"  Generated {len(hard_neg)} hard negative pairs")

    print("[PairGen] Generating true negative pairs...")
    true_neg = generate_true_negative_pairs(n=n_true_neg)
    print(f"  Generated {len(true_neg)} true negative pairs")

    all_pairs = paraphrase + cross + hard_neg + true_neg

    if real_features is not None:
        print("[PairGen] Generating real-data pairs...")
        real_pairs = build_real_data_pairs(real_features, max_pairs=n_real_pairs, seed=seed)
        all_pairs = all_pairs + real_pairs

    random.shuffle(all_pairs)

    print(f"\n[PairGen] Total pairs: {len(all_pairs)}")
    _print_pair_stats(all_pairs)

    return all_pairs


def _print_pair_stats(pairs: List[SentencePair]) -> None:
    """Print dataset composition summary."""
    from collections import Counter
    type_counts = Counter(p.pair_type for p in pairs)
    scores = [p.score for p in pairs]

    print("\nPair type breakdown:")
    for pair_type, count in sorted(type_counts.items()):
        print(f"  {pair_type:35s}: {count:4d}")

    print(f"\nScore distribution:")
    print(f"  Mean:  {np.mean(scores):.3f}")
    print(f"  Std:   {np.std(scores):.3f}")
    print(f"  Min:   {np.min(scores):.3f}")
    print(f"  Max:   {np.max(scores):.3f}")


# ── Quick Test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pairs = build_finetuning_dataset()

    print("\n── Sample Pairs ──")
    shown = {}
    for p in pairs:
        if p.pair_type not in shown:
            print(f"\n[{p.pair_type}] score={p.score}")
            print(f"  A: {p.sentence_a}")
            print(f"  B: {p.sentence_b}")
            shown[p.pair_type] = True
        if len(shown) >= 5:
            break
