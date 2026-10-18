from src.agri_qa import (
    QAPair, assign_groups, clean_pairs, group_split, leakage_report, question_key,
)

ROWS = [
    {"question": "What is crop rotation?", "answer": "Growing different crops in sequence on the same land."},
    {"question": "what is crop rotation", "answer": "Growing a series of different crops in the same area over seasons."},
    {"question": "Crop rotation - what is it?", "answer": "Growing different crops in sequence on the same land."},
    {"question": "Which crop is the major food source?", "answer": "Cassava "},                    # too short
    {"question": "How do I control aphids?", "answer": "  Spray neem oil 5 ml per litre   of water. "},
    {"question": "How do I control aphids?", "answer": "Spray neem oil 5 ml per litre of water."},  # exact dup
    {"question": "When to sow wheat in Punjab?", "answer": "Sow timely wheat between 25 October and 15 November."},
]


def test_question_key_ignores_case_order_and_stopwords():
    assert question_key("What is crop rotation?") == question_key("crop rotation - what is it")


def test_cleaning_drops_short_and_duplicate_answers_and_normalises_whitespace():
    pairs, stats = clean_pairs(ROWS)
    assert stats["dropped_short_answer"] == 1
    assert stats["dropped_duplicate"] == 2   # duplicate aphid row + reordered crop-rotation question with same answer
    assert all("  " not in p.answer for p in pairs)


def test_paraphrased_questions_and_shared_answers_land_in_one_group():
    pairs = [QAPair(r["question"], r["answer"]) for r in ROWS[:3]]
    assert assign_groups(pairs) == 1


def test_split_has_no_leakage_and_is_deterministic():
    pairs = [QAPair(f"How to grow crop number {i} well?", f"Answer text number {i} with enough words.")
             for i in range(400)]
    pairs += [QAPair(f"how to grow crop number {i} well", f"A paraphrased answer {i} with enough words.")
              for i in range(0, 400, 7)]
    splits = group_split(pairs, val_frac=0.1, test_frac=0.1)
    assert set(leakage_report(splits).values()) == {0}
    assert all(splits[name] for name in ("train", "val", "test"))

    again = [QAPair(p.question, p.answer) for p in pairs]
    splits_again = group_split(again, val_frac=0.1, test_frac=0.1)
    assert [p.split for p in pairs] == [p.split for p in again]
    assert {k: len(v) for k, v in splits.items()} == {k: len(v) for k, v in splits_again.items()}
