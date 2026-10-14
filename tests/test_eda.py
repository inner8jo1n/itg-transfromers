import pandas as pd
import pytest

from itg_transformers.eda import (
    class_distribution,
    conflicting_duplicates,
    conflicting_rows,
    duplicates_summary,
    log_odds_z,
    matches,
    pattern_error_rates,
    pattern_shares,
    sample_per_class,
    split_overlap,
    text_lengths,
    truncation_shares,
    word_counts,
)


def make_frame(texts, labels):
    return pd.DataFrame({"text": texts, "label": labels})


def test_conflicting_duplicates_finds_texts_with_different_labels():
    frame = make_frame(["a", "a", "b", "b", "c"], [0, 1, 2, 2, 1])

    conflicts = conflicting_duplicates(frame)

    assert conflicts.to_dict() == {"a": [0, 1]}


def test_conflicting_rows_returns_every_occurrence_grouped_by_text():
    frame = make_frame(["b", "a", "c", "b", "a"], [0, 1, 2, 1, 1])

    rows = conflicting_rows(frame)

    assert rows.index.tolist() == [0, 3]
    assert rows["label"].tolist() == [0, 1]


def test_duplicates_summary_counts_normalized_duplicates():
    frame = make_frame(["Hi", "hi ", "Hi", "bye"], [1, 1, 2, 0])

    summary = duplicates_summary({"train": frame}).loc["train"]

    assert summary["rows"] == 4
    assert summary["exact_duplicates"] == 1
    assert summary["normalized_duplicates"] == 2
    assert summary["conflicting_texts"] == 1


def test_split_overlap_ignores_case_and_spaces():
    source = make_frame(["Hello", "world"], [1, 1])
    target = make_frame([" hello", "other"], [1, 0])

    assert split_overlap(source, target) == 1


def test_class_distribution_keeps_every_label_in_order():
    splits = {
        "train": make_frame(["a", "b", "c"], [2, 2, 0]),
        "test": make_frame(["d"], [1]),
    }
    labels = {0: "neg", 1: "neu", 2: "pos"}

    shares = class_distribution(splits, labels, normalize=True)

    assert shares.columns.tolist() == ["neg", "neu", "pos"]
    assert shares.index.tolist() == ["train", "test"]
    assert shares.round(2).to_numpy().tolist() == [
        [0.33, 0.0, 0.67],
        [0.0, 1.0, 0.0],
    ]


def test_text_lengths_counts_chars_and_words():
    lengths = text_lengths(pd.Series(["hi there", "  one  "]))

    assert lengths["chars"].tolist() == [8, 7]
    assert lengths["words"].tolist() == [2, 1]


def test_truncation_shares_counts_texts_longer_than_limit():
    shares = truncation_shares(pd.Series([10, 20, 30, 40]), [20, 40])

    assert shares.to_dict() == {20: 0.5, 40: 0.0}


def test_matches_supports_backreferences():
    texts = pd.Series(["soooo good", "so good"])

    assert matches(texts, r"([a-z])\1{2,}").tolist() == [True, False]


def test_pattern_shares_per_split():
    splits = {
        "train": make_frame(["@user hi", "hi"], [1, 1]),
        "test": make_frame(["hi #tag"], [1]),
    }
    patterns = {"mention": r"@user", "hashtag": r"#\w+"}

    shares = pattern_shares(splits, patterns)

    assert shares.to_dict() == {
        "train": {"mention": 0.5, "hashtag": 0.0},
        "test": {"mention": 0.0, "hashtag": 1.0},
    }


def test_word_counts_lowercases_and_drops_stop_words():
    counts = word_counts(pd.Series(["The GOOD day", "good #Day @user don't"]))

    assert counts.to_dict() == {
        "good": 2.0,
        "day": 1.0,
        "#day": 1.0,
        "don't": 1.0,
    }


def test_log_odds_z_ranks_target_specific_words_first():
    target = pd.Series({"love": 30.0, "game": 10.0})
    rest = pd.Series({"hate": 30.0, "game": 10.0})

    z = log_odds_z(target, rest)

    assert z.index[0] == "love"
    assert z.index[-1] == "hate"
    assert z["game"] == pytest.approx(0, abs=0.5)


def test_sample_per_class_takes_n_rows_of_every_class():
    frame = make_frame(list("abcdef"), [0, 0, 0, 1, 1, 1])

    sample = sample_per_class(frame, {0: "neg", 1: "pos"}, n=2)

    assert sample.index.names == ["label", "row"]
    assert sample.groupby(level="label").size().to_dict() == {
        "neg": 2,
        "pos": 2,
    }
    assert set(sample.loc["neg"].index) <= {0, 1, 2}


def test_pattern_error_rates_compares_tweets_with_and_without():
    texts = pd.Series(["not good", "good", "not bad", "fine"])
    correct = pd.Series([False, True, True, True])

    rates = pattern_error_rates(texts, correct, {"negation": r"\bnot\b"})

    assert rates.loc["negation", "tweets"] == 2
    assert rates.loc["negation", "error_rate_with"] == 0.5
    assert rates.loc["negation", "error_rate_without"] == 0.0
