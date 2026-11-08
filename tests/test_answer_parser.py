"""Answer parser + abstention rule shared by llm_reader, rescore and hallucination.
Cases are real reader outputs from the P9 review (Nov 2026).

Run:  cd seminar2_kgrag && python -m pytest -q tests/test_answer_parser.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from qa_text_eval import extract_answer_entities, split_answer, eval_hit1  # noqa: E402
from kgqa_bench.data import BenchDataset  # noqa: E402
from kgqa_bench import hallucination as H  # noqa: E402


def test_lead_in_colon_is_not_a_claim():
    # Llama-3.1 chat: "the answer is:" used to give the claim ':' (absent -> hallucinated) and hit1 = 0
    ents, ab = split_answer("Based on the given reasoning path, the answer is:\n\n* Mobile")
    assert ents == ["Mobile"] and not ab
    assert eval_hit1(ents, ["Mobile"]) == 1
    assert split_answer("The answer is: Paris")[0] == ["Paris"]
    assert split_answer("Based on the paths, the answers are Paris, Lyon and Nice.")[0] == ["Paris", "Lyon", "Nice"]


def test_copied_path_gives_tail_entity():
    pred = "Here is the answer:\n* The Social Network -> film.director.film -> David Fincher\n* A -> r1 -> B -> r2 -> C"
    assert extract_answer_entities(pred) == ["David Fincher", "C"]


def test_json_list_items():
    assert split_answer('["Unitary state", "Republic", "m.01dw03"]')[0] == ["Unitary state", "Republic", "m.01dw03"]
    assert split_answer('["2012 Benghazi attack", "First Battle')[0] == ["2012 Benghazi attack"]   # truncated


def test_abstention_sentences():
    for pred in ("unknown",
                 "Unfortunately, the provided reasoning paths do not contain any information about the governor of Ohio "
                 "in 2011. The paths only mention Salmon P. Chase and Thomas Ewing, who are historical figures.",
                 "Based on the reasoning paths, there is no direct information about who Demi dated."):
        ents, ab = split_answer(pred)
        assert ents == [] and ab, pred
    assert split_answer("Paris") == (["Paris"], False)
    assert split_answer("") == ([], False)


def test_parametric_answer_after_abstention_is_an_answer():
    pred = ("Unfortunately, the reasoning paths provided do not contain any information about who plays Ken Barlow.\n"
            "However, I can tell you that the character Ken Barlow is played by actor William Roache.")
    ents, ab = split_answer(pred)
    assert not ab and len(ents) == 1 and "William Roache" in ents[0]
    # header sentence ending with ':' is dropped, the listed item is kept
    pred2 = ("Unfortunately, the paths do not contain information about his role. The paths only provide his positions.\n"
             "However, based on general knowledge, he was the author. I would say:\n\n* Author")
    assert split_answer(pred2) == (["Author"], False)


def test_verifier_counts_parametric_answer_as_hallucination():
    ds = BenchDataset.synthetic(4, seed=31).annotate_hops()
    q = ds[0]
    rec = {"qid": q.qid, "method": "m", "selected_paths": [], "llm_hit": 0,
           "llm_pred": "The paths do not provide any information.\nHowever, I believe it is Berlin_fabricated_entity."}
    v = H.verify_question(q, rec)
    assert v["abstained"] == 0 and v["answered"] == 1 and v["hallucinated"] == 1
    v2 = H.verify_question(q, dict(rec, llm_pred="The paths do not provide any information."))
    assert v2["abstained"] == 1 and v2["answered"] == 0 and v2["n_claims"] == 0


if __name__ == "__main__":
    import traceback
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("  ok", name)
            except Exception:
                fails += 1; print("  FAIL", name); traceback.print_exc()
    print("ALL PASSED" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
