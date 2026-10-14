import re
from collections import Counter
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def normalize_text(texts: pd.Series) -> pd.Series:
    return texts.str.strip().str.lower()


def conflicting_duplicates(frame: pd.DataFrame) -> pd.Series:
    """Texts that occur several times with different labels."""
    labels = frame.groupby("text")["label"].agg(lambda x: sorted(set(x)))
    return labels[labels.map(len) > 1]


def conflicting_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Every occurrence of a conflicting text, grouped by text."""
    texts = conflicting_duplicates(frame).index
    rows = frame[frame["text"].isin(texts)]
    return rows.sort_values("text", kind="stable")


def duplicates_summary(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = {
        name: {
            "rows": len(frame),
            "exact_duplicates": frame["text"].duplicated().sum(),
            "normalized_duplicates": (
                normalize_text(frame["text"]).duplicated().sum()
            ),
            "conflicting_texts": len(conflicting_duplicates(frame)),
        }
        for name, frame in splits.items()
    }
    return pd.DataFrame(rows).T


def split_overlap(source: pd.DataFrame, target: pd.DataFrame) -> int:
    """How many target texts also appear in source, after normalization."""
    known = set(normalize_text(source["text"]))
    return int(normalize_text(target["text"]).isin(known).sum())


def class_distribution(
    splits: dict[str, pd.DataFrame],
    labels: dict[int, str],
    normalize: bool = False,
) -> pd.DataFrame:
    """Rows are splits, columns are label names."""
    counts = {
        name: frame["label"].value_counts(normalize=normalize)
        for name, frame in splits.items()
    }
    table = pd.DataFrame(counts).T.reindex(columns=list(labels)).fillna(0)
    return table.rename(columns=labels)


def text_lengths(texts: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "chars": texts.str.len(),
            "words": texts.str.count(r"\S+"),
        }
    )


def truncation_shares(
    token_counts: pd.Series, max_lengths: Sequence[int]
) -> pd.Series:
    """Share of texts that would be cut at every max_length."""
    return pd.Series(
        {limit: float((token_counts > limit).mean()) for limit in max_lengths},
        name="truncated_share",
    )


TWITTER_PATTERNS = {
    "user_mention": r"@user\b",
    "url": r"\bhttp",
    "hashtag": r"#\w+",
    "emoticon": r"(?:^|\s)[:;=]-?[()DPp](?:\s|$)",
    "emoji": r"[\U0001F300-\U0001FAFF☀-➿]",
    "html_entity": r"&(?:amp|lt|gt|quot|#\d+);",
    "unicode_escape": r"\\u[0-9a-fA-F]{4}",
    "non_ascii": r"[^\x00-\x7f]",
    "caps_word": r"\b[A-Z]{3,}\b",
    "elongated_word": r"([a-zA-Z])\1{2,}",
    "repeated_punct": r"[!?]{2,}",
    "quote_wrapped": r'^".*"$',
}


def matches(texts: pd.Series, pattern: str) -> pd.Series:
    """Boolean mask of texts containing the pattern.

    Uses Python's `re`, because pandas string methods run on RE2, which
    has no backreferences and no `\\U` escapes.
    """
    regex = re.compile(pattern)

    def found(text: str) -> bool:
        return regex.search(text) is not None

    return texts.map(found).astype(bool)


def pattern_shares(
    splits: dict[str, pd.DataFrame], patterns: dict[str, str]
) -> pd.DataFrame:
    """Rows are patterns, columns are splits."""
    return pd.DataFrame(
        {
            name: {
                key: float(matches(frame["text"], pattern).mean())
                for key, pattern in patterns.items()
            }
            for name, frame in splits.items()
        }
    )


WORD = re.compile(r"#?[a-z0-9]+(?:['’][a-z]+)?")
STOP_WORDS = frozenset(ENGLISH_STOP_WORDS | {"user"})


def word_counts(
    texts: pd.Series, stop_words: frozenset[str] = STOP_WORDS
) -> pd.Series:
    """Lowercased word and hashtag counts, most frequent first."""
    counts = Counter(
        word
        for text in texts
        for word in WORD.findall(text.lower())
        if len(word) > 1 and word not in stop_words
    )
    return pd.Series(counts, dtype=float).sort_values(ascending=False)


def log_odds_z(target: pd.Series, rest: pd.Series) -> pd.Series:
    """How strongly each word is tied to `target` rather than `rest`.

    Log-odds ratio with an informative Dirichlet prior (Monroe et al.,
    2008), divided by its standard deviation. Unlike raw frequency it
    ignores words that are common everywhere, and unlike plain log-odds
    it does not overrate words seen only a couple of times.
    """
    words = target.index.union(rest.index)
    target = target.reindex(words, fill_value=0)
    rest = rest.reindex(words, fill_value=0)
    prior = target + rest
    prior_total = prior.sum()

    def log_odds(counts: pd.Series) -> pd.Series:
        total = counts.sum()
        return np.log(
            (counts + prior) / (total + prior_total - counts - prior)
        )

    delta = log_odds(target) - log_odds(rest)
    variance = 1 / (target + prior) + 1 / (rest + prior)
    return (delta / np.sqrt(variance)).sort_values(ascending=False)


def sample_per_class(
    frame: pd.DataFrame, labels: dict[int, str], n: int, seed: int = 42
) -> pd.DataFrame:
    """`n` random rows of every class, indexed by label name and row."""
    parts = {
        name: frame.loc[frame["label"] == label, ["text"]].sample(
            n, random_state=seed
        )
        for label, name in labels.items()
    }
    return pd.concat(parts, names=["label", "row"])


def pattern_error_rates(
    texts: pd.Series, correct: pd.Series, patterns: dict[str, str]
) -> pd.DataFrame:
    """Error rate of tweets with and without every pattern."""
    rows = {}
    for name, pattern in patterns.items():
        has = matches(texts, pattern)
        rows[name] = {
            "tweets": int(has.sum()),
            "error_rate_with": float(1 - correct[has].mean()),
            "error_rate_without": float(1 - correct[~has].mean()),
        }
    return pd.DataFrame(rows).T
