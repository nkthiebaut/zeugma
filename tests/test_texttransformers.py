from zeugma.texttransformers import ItemSelector, Namer, RareWordsTagger, TextStats


def test_rare_words_tagger(sample_corpus):
    """Test rare words tagging on a basic corpus."""
    rare_tagger = RareWordsTagger(min_count=2)
    tag = rare_tagger.oov_tag
    out = rare_tagger.fit_transform(sample_corpus)
    assert len(out) == len(sample_corpus)
    assert out[0] == tag + " a " + tag + " " + tag + " text"


def test_item_selector():
    """Test selecting items in a mappable from previous pipeline step."""
    test_case = {"a": 1, "b": 2}
    item_selector = ItemSelector("a")
    out = item_selector.fit_transform(test_case)
    assert out == test_case["a"]


def test_text_stats(sample_corpus):
    """Test basic text statistics extraction transformer."""
    text_stats = TextStats()
    out = text_stats.fit_transform(sample_corpus)
    assert out[0]["length"] == len(sample_corpus[0])
    assert out[0]["num_sentences"] == 0


def test_namer():
    """Test turning features into mappables."""
    namer = Namer("feature")
    out = namer.fit_transform([1, 2, 3])
    assert out == {"feature": [1, 2, 3]}
