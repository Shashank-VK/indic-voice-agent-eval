import pytest
from src.utterances import UTTERANCES, Utterance, get_utterance_by_id, filter_utterances_by_tag

VALID_TAGS = {"clean", "code_mixing", "regional_accent", "background_noise"}
VALID_LANGS = {"hi", "hi-en"}

def test_utterances_non_empty():
    assert len(UTTERANCES) >= 40, f"Expected at least 40 utterances, found {len(UTTERANCES)}"

def test_utterances_unique_ids():
    ids = [u.id for u in UTTERANCES]
    assert len(ids) == len(set(ids)), f"Duplicate utterance IDs found: {[x for x in ids if ids.count(x) > 1]}"

def test_utterance_fields_valid():
    for u in UTTERANCES:
        assert isinstance(u, Utterance)
        assert u.id.strip() != ""
        assert u.text.strip() != ""
        assert u.lang in VALID_LANGS, f"Invalid lang {u.lang} for {u.id}"
        assert u.domain.strip() != ""
        assert len(u.failure_modes) > 0, f"No failure modes for {u.id}"
        for tag in u.failure_modes:
            assert tag in VALID_TAGS, f"Invalid tag '{tag}' in {u.id}"

def test_failure_mode_tag_coverage():
    for tag in VALID_TAGS:
        matching = [u for u in UTTERANCES if tag in u.failure_modes]
        assert len(matching) >= 5, f"Tag '{tag}' has only {len(matching)} utterances (min 5 expected)"

def test_domain_diversity():
    domains = {u.domain for u in UTTERANCES}
    assert len(domains) >= 4, f"Expected at least 4 diverse domains, found {domains}"
    assert "banking" in domains or "payments" in domains
    assert "ecommerce" in domains

def test_get_utterance_by_id():
    first = UTTERANCES[0]
    fetched = get_utterance_by_id(first.id)
    assert fetched == first
    assert get_utterance_by_id("non_existent_id") is None

def test_filter_utterances_by_tag():
    clean_list = filter_utterances_by_tag("clean")
    assert len(clean_list) >= 5
    for u in clean_list:
        assert "clean" in u.failure_modes
