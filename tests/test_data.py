import pandas as pd
import pytest

from itg_transformers.data import clean_text, load_split, split_path


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("I can\\u2019t see him", "I can’t see him"),
        ("Yes glass of red\\u002c Rammstein", "Yes glass of red, Rammstein"),
        ("Seth Rollins talks &amp; talks", "Seth Rollins talks & talks"),
        ("Britain&#039;s got tech talent", "Britain's got tech talent"),
        ("Twitter: &amp;quot;Vote now&amp;quot;", 'Twitter: "Vote now"'),
        ('"Random Friday fun: where?"', "Random Friday fun: where?"),
        (
            '"\\""""Im going home.\\"""" -@Alwhit14"',
            '"Im going home." -@Alwhit14',
        ),
        ('"He said ""Bad Blood,"" today"', 'He said "Bad Blood," today'),
        ('"Oh Yeah!" #android', '"Oh Yeah!" #android'),
        ("  plain tweet ", "plain tweet"),
    ],
)
def test_clean_text(raw, expected):
    assert clean_text(raw) == expected


def test_load_split_cleans_texts_unless_asked_not_to(tmp_path):
    pd.DataFrame({"text": ["it\\u2019s"], "label": [1]}).to_csv(
        split_path("train", tmp_path), index=False
    )

    assert load_split("train", tmp_path)["text"].tolist() == ["it’s"]
    assert load_split("train", tmp_path, clean=False)["text"].tolist() == [
        "it\\u2019s"
    ]
