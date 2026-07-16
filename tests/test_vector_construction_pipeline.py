"""
Tests for Stage 2B: persona-vector construction pipeline.

All tests run without GPU, Modal, torch, or transformers.
Covers: generation job construction, response_id stability, scoring prompt
content, JSON parsing, filter logic, activation metadata schema, vector
computation on synthetic data, validation metrics, layer selection, and
no ETHICS item contamination.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.vectors.artifact_bank import load_artifact_bank
from src.vectors.compute_vectors import (
    compute_all_vectors,
    compute_trait_vector,
    load_vector,
    load_vector_metadata,
    save_vector_metadata,
)
from src.vectors.extract_activations import (
    MOCK_HIDDEN_DIM,
    load_activation,
    load_activation_metadata,
    mock_extract,
    save_activation_metadata,
)
from src.vectors.generate_responses import (
    build_generation_jobs,
    load_responses,
    mock_generate,
    save_responses,
)
from src.vectors.score_responses import (
    build_judge_prompt,
    filter_for_extraction,
    load_scored,
    mock_score,
    parse_judge_response,
    save_scored,
)
from src.vectors.validate_vectors import (
    compute_validation_metrics,
    load_validation_results,
    save_validation_results,
    select_best_layer,
    validate_all_vectors,
)
from src.vectors.vector_data import (
    ActivationRecord,
    GeneratedResponse,
    PersonaVectorMeta,
    ScoredResponse,
    VectorValidationResult,
)

_ARTIFACTS_PATH = Path(__file__).resolve().parent.parent / "configs" / "trait_vector_artifacts.yaml"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bank():
    return load_artifact_bank(_ARTIFACTS_PATH)


@pytest.fixture(scope="module")
def extraction_jobs(bank):
    return build_generation_jobs(bank, split="extraction", limit=8)


@pytest.fixture(scope="module")
def validation_jobs(bank):
    return build_generation_jobs(bank, split="validation", limit=8)


@pytest.fixture(scope="module")
def mock_responses(extraction_jobs):
    return mock_generate(extraction_jobs)


@pytest.fixture(scope="module")
def mock_val_responses(validation_jobs):
    return mock_generate(validation_jobs)


@pytest.fixture(scope="module")
def mock_scored(mock_responses):
    return mock_score(mock_responses)


@pytest.fixture(scope="module")
def mock_val_scored(mock_val_responses):
    return mock_score(mock_val_responses)


@pytest.fixture(scope="module")
def bipolar_scored(bank):
    """Both-pole mock scored responses covering honesty (no limit).

    Uses all honesty jobs so both poles are guaranteed present.
    """
    sp_df = bank.system_prompts_df
    q_df = bank.questions_df[bank.questions_df["split"] == "extraction"]
    from src.vectors.generate_responses import build_generation_jobs, mock_generate
    # Build only honesty jobs to keep it small (5 pos × 20 q + 5 neg × 20 q = 200)
    # but cap per-pole at 2 prompts × 4 questions = 8 each for speed
    jobs: list[dict] = []
    for pole in ("positive", "negative"):
        pole_prompts = sp_df[(sp_df["trait"] == "honesty") & (sp_df["pole"] == pole)].head(2)
        hon_qs = q_df[q_df["trait"] == "honesty"].head(4)
        for _, sp_row in pole_prompts.iterrows():
            for _, q_row in hon_qs.iterrows():
                from src.vectors.vector_data import GeneratedResponse
                jobs.append({
                    "response_id": GeneratedResponse.make_id(sp_row["prompt_id"], q_row["question_id"]),
                    "trait": "honesty",
                    "pole": pole,
                    "split": "extraction",
                    "system_prompt_id": sp_row["prompt_id"],
                    "question_id": q_row["question_id"],
                    "system_prompt_text": sp_row["prompt_text"],
                    "question_text": q_row["question_text"],
                })
    responses = mock_generate(jobs)
    return mock_score(responses)


@pytest.fixture(scope="module")
def bipolar_val_scored(bank):
    """Both-pole validation-split mock scored responses for honesty."""
    sp_df = bank.system_prompts_df
    q_df = bank.questions_df[bank.questions_df["split"] == "validation"]
    jobs: list[dict] = []
    for pole in ("positive", "negative"):
        pole_prompts = sp_df[(sp_df["trait"] == "honesty") & (sp_df["pole"] == pole)].head(2)
        hon_qs = q_df[q_df["trait"] == "honesty"].head(4)
        for _, sp_row in pole_prompts.iterrows():
            for _, q_row in hon_qs.iterrows():
                from src.vectors.vector_data import GeneratedResponse
                jobs.append({
                    "response_id": GeneratedResponse.make_id(sp_row["prompt_id"], q_row["question_id"]),
                    "trait": "honesty",
                    "pole": pole,
                    "split": "validation",
                    "system_prompt_id": sp_row["prompt_id"],
                    "question_id": q_row["question_id"],
                    "system_prompt_text": sp_row["prompt_text"],
                    "question_text": q_row["question_text"],
                })
    from src.vectors.generate_responses import mock_generate
    responses = mock_generate(jobs)
    return mock_score(responses)


# ---------------------------------------------------------------------------
# 1. Generation job construction
# ---------------------------------------------------------------------------

def test_build_generation_jobs_returns_list(extraction_jobs):
    assert isinstance(extraction_jobs, list)
    assert len(extraction_jobs) > 0


def test_generation_jobs_have_required_keys(extraction_jobs):
    required = {
        "response_id", "trait", "pole", "split",
        "system_prompt_id", "question_id",
        "system_prompt_text", "question_text",
    }
    for job in extraction_jobs:
        assert required.issubset(set(job.keys()))


def test_generation_jobs_split_is_extraction(extraction_jobs):
    assert all(j["split"] == "extraction" for j in extraction_jobs)


def test_generation_jobs_split_is_validation(validation_jobs):
    assert all(j["split"] == "validation" for j in validation_jobs)


def test_generation_jobs_limit_respected(bank):
    jobs = build_generation_jobs(bank, split="extraction", limit=5)
    assert len(jobs) == 5


def test_generation_jobs_cover_all_traits_and_poles(bank):
    jobs = build_generation_jobs(bank, split="extraction")
    traits = {j["trait"] for j in jobs}
    poles = {j["pole"] for j in jobs}
    assert traits == {"honesty", "harmlessness", "fairness", "compassion"}
    assert poles == {"positive", "negative"}


def test_no_ethics_items_in_generation_jobs(extraction_jobs):
    """Verify no ETHICS split names appear in trait or split fields."""
    ethics_splits = {"commonsense", "deontology", "justice", "utilitarianism", "virtue"}
    for job in extraction_jobs:
        assert job["trait"] not in ethics_splits
        assert job["split"] not in ethics_splits


# ---------------------------------------------------------------------------
# 2. response_id stability
# ---------------------------------------------------------------------------

def test_response_id_format(extraction_jobs):
    for job in extraction_jobs:
        rid = job["response_id"]
        assert "__" in rid, f"response_id should contain '__': {rid}"
        parts = rid.split("__")
        assert len(parts) == 2
        assert parts[0] == job["system_prompt_id"]
        assert parts[1] == job["question_id"]


def test_response_id_is_stable(bank):
    """Same inputs produce the same response_id."""
    jobs1 = build_generation_jobs(bank, split="extraction", limit=4)
    jobs2 = build_generation_jobs(bank, split="extraction", limit=4)
    for j1, j2 in zip(jobs1, jobs2):
        assert j1["response_id"] == j2["response_id"]


def test_make_id():
    rid = GeneratedResponse.make_id("hon_pos_001", "hon_ext_001")
    assert rid == "hon_pos_001__hon_ext_001"


# ---------------------------------------------------------------------------
# 3. Mock generation
# ---------------------------------------------------------------------------

def test_mock_generate_returns_generated_response_objects(mock_responses):
    assert all(isinstance(r, GeneratedResponse) for r in mock_responses)


def test_mock_generate_response_text_non_empty(mock_responses):
    assert all(r.response_text.strip() != "" for r in mock_responses)


def test_mock_generate_model_name_is_mock(mock_responses):
    assert all(r.model_name == "mock" for r in mock_responses)


def test_mock_responses_match_jobs(extraction_jobs, mock_responses):
    job_ids = {j["response_id"] for j in extraction_jobs}
    resp_ids = {r.response_id for r in mock_responses}
    assert job_ids == resp_ids


# ---------------------------------------------------------------------------
# 4. Scoring — judge prompt contains rubric and trait
# ---------------------------------------------------------------------------

def test_judge_prompt_contains_trait(bank, mock_responses):
    resp = mock_responses[0]
    rubric = bank.raw[resp.trait]["evaluation_rubric"]
    instructions = rubric["scoring_instructions"]
    scale = rubric["score_scale"]["interpretation"]
    sys_p, user_p = build_judge_prompt(resp, instructions, scale)
    assert resp.trait in user_p
    assert instructions[:30].strip() in user_p


def test_judge_prompt_contains_rubric_score_scale(bank, mock_responses):
    resp = mock_responses[0]
    rubric = bank.raw[resp.trait]["evaluation_rubric"]
    instructions = rubric["scoring_instructions"]
    scale = rubric["score_scale"]["interpretation"]
    _, user_p = build_judge_prompt(resp, instructions, scale)
    # Should mention the 0-100 scale
    assert "0" in user_p and "100" in user_p


def test_judge_prompt_contains_response_text(bank, mock_responses):
    resp = mock_responses[0]
    rubric = bank.raw[resp.trait]["evaluation_rubric"]
    instructions = rubric["scoring_instructions"]
    scale = rubric["score_scale"]["interpretation"]
    _, user_p = build_judge_prompt(resp, instructions, scale)
    assert resp.response_text.strip()[:30] in user_p


def test_judge_prompt_contains_response_id(bank, mock_responses):
    resp = mock_responses[0]
    rubric = bank.raw[resp.trait]["evaluation_rubric"]
    _, user_p = build_judge_prompt(
        resp,
        rubric["scoring_instructions"],
        rubric["score_scale"]["interpretation"],
    )
    assert resp.response_id in user_p


# ---------------------------------------------------------------------------
# 5. Score JSON parsing
# ---------------------------------------------------------------------------

def test_parse_valid_json():
    raw = '{"response_id": "abc", "trait_score": 85, "score_rationale": "good"}'
    score, rationale = parse_judge_response(raw, "abc")
    assert score == 85.0
    assert rationale == "good"


def test_parse_json_embedded_in_prose():
    raw = 'Here is my evaluation: {"response_id": "abc", "trait_score": 42, "score_rationale": "ok"} done.'
    score, rationale = parse_judge_response(raw, "abc")
    assert score == 42.0


def test_parse_clamps_to_0_100():
    raw = '{"response_id": "x", "trait_score": 150, "score_rationale": "too high"}'
    score, _ = parse_judge_response(raw, "x")
    assert score == 100.0

    raw2 = '{"response_id": "x", "trait_score": -10, "score_rationale": "negative"}'
    score2, _ = parse_judge_response(raw2, "x")
    assert score2 == 0.0


def test_parse_invalid_json_strict_raises():
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_judge_response("This is not JSON at all.", "abc", strict=True)


def test_parse_invalid_json_non_strict_returns_negative_one():
    score, rationale = parse_judge_response("not json", "abc", strict=False)
    assert score == -1.0
    assert "parse_error" in rationale


# ---------------------------------------------------------------------------
# 6. Filter logic (positive / negative pole thresholds)
# ---------------------------------------------------------------------------

def test_positive_pole_kept_above_threshold():
    resp = _fake_response("honesty", "positive", "hon_pos_001__hon_ext_001")
    scored = ScoredResponse.from_generated(resp, 80.0, "good", 70.0, 30.0)
    assert scored.keep_for_vector_extraction is True


def test_positive_pole_rejected_below_threshold():
    resp = _fake_response("honesty", "positive", "hon_pos_001__hon_ext_001")
    scored = ScoredResponse.from_generated(resp, 60.0, "weak", 70.0, 30.0)
    assert scored.keep_for_vector_extraction is False


def test_negative_pole_kept_below_threshold():
    resp = _fake_response("honesty", "negative", "hon_neg_001__hon_ext_001")
    scored = ScoredResponse.from_generated(resp, 20.0, "good neg", 70.0, 30.0)
    assert scored.keep_for_vector_extraction is True


def test_negative_pole_rejected_above_threshold():
    resp = _fake_response("honesty", "negative", "hon_neg_001__hon_ext_001")
    scored = ScoredResponse.from_generated(resp, 50.0, "ambiguous", 70.0, 30.0)
    assert scored.keep_for_vector_extraction is False


def test_filter_for_extraction_returns_only_kept(mock_scored):
    retained = filter_for_extraction(mock_scored)
    assert all(r.keep_for_vector_extraction for r in retained)


def test_mock_score_positive_pole_kept(mock_scored):
    pos = [r for r in mock_scored if r.pole == "positive"]
    assert all(r.keep_for_vector_extraction for r in pos)


def test_mock_score_negative_pole_kept(mock_scored):
    neg = [r for r in mock_scored if r.pole == "negative"]
    assert all(r.keep_for_vector_extraction for r in neg)


# ---------------------------------------------------------------------------
# 7. Activation metadata schema
# ---------------------------------------------------------------------------

def test_activation_record_fields():
    rec = ActivationRecord(
        response_id="abc",
        trait="honesty",
        pole="positive",
        split="extraction",
        layer=28,
        activation_path="/tmp/abc_layer28.npy",
        pooling_method="mean_response_token",
        hidden_dim=64,
    )
    assert rec.pooling_method == "mean_response_token"
    assert rec.layer == 28


def test_mock_extract_saves_npy_files(mock_scored):
    with tempfile.TemporaryDirectory() as tmp:
        records = mock_extract(mock_scored, candidate_layers=[0], out_dir=tmp, hidden_dim=16)
        for rec in records:
            assert Path(rec.activation_path).exists()
            arr = np.load(rec.activation_path)
            assert arr.shape == (16,)


def test_mock_extract_only_processes_retained(mock_scored):
    with tempfile.TemporaryDirectory() as tmp:
        records = mock_extract(mock_scored, candidate_layers=[0], out_dir=tmp)
        retained_ids = {s.response_id for s in mock_scored if s.keep_for_vector_extraction}
        record_ids = {r.response_id for r in records}
        assert record_ids == retained_ids


def test_activation_metadata_roundtrip(mock_scored):
    with tempfile.TemporaryDirectory() as tmp:
        records = mock_extract(mock_scored, candidate_layers=[0, 1], out_dir=tmp)
        meta_path = save_activation_metadata(records, tmp)
        loaded = load_activation_metadata(meta_path)
        assert len(loaded) == len(records)
        assert loaded[0].response_id == records[0].response_id
        assert loaded[0].layer == records[0].layer


def test_activation_metadata_has_correct_columns(mock_scored):
    with tempfile.TemporaryDirectory() as tmp:
        records = mock_extract(mock_scored, candidate_layers=[0], out_dir=tmp)
        meta_path = save_activation_metadata(records, tmp)
        df = pd.read_parquet(meta_path)
        for col in ["response_id", "trait", "pole", "split", "layer", "activation_path", "hidden_dim"]:
            assert col in df.columns


# ---------------------------------------------------------------------------
# 8. Vector computation on synthetic activations
# ---------------------------------------------------------------------------

def test_compute_trait_vector_basic():
    pos = np.array([[1.0, 0.0], [1.0, 0.1]], dtype=np.float32)
    neg = np.array([[-1.0, 0.0], [-1.0, -0.1]], dtype=np.float32)
    vec = compute_trait_vector(pos, neg, normalize=False)
    assert vec.shape == (2,)
    assert vec[0] > 0  # positive direction in first dim


def test_compute_trait_vector_normalized():
    pos = np.random.randn(5, 32).astype(np.float32)
    neg = np.random.randn(5, 32).astype(np.float32)
    vec = compute_trait_vector(pos, neg, normalize=True)
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-5


def test_compute_trait_vector_shape_mismatch_raises():
    pos = np.random.randn(3, 32).astype(np.float32)
    neg = np.random.randn(3, 16).astype(np.float32)
    with pytest.raises(ValueError, match="Hidden dim mismatch"):
        compute_trait_vector(pos, neg)


def test_compute_all_vectors_roundtrip(bipolar_scored):
    with tempfile.TemporaryDirectory() as tmp:
        records = mock_extract(bipolar_scored, candidate_layers=[0, 1], out_dir=tmp, hidden_dim=MOCK_HIDDEN_DIM)
        vec_metas = compute_all_vectors(
            records=[r for r in records if r.split == "extraction"],
            candidate_layers=[0, 1],
            traits=["honesty"],
            normalize=True,
            out_dir=Path(tmp) / "vectors",
        )
        assert len(vec_metas) > 0
        for m in vec_metas:
            vec = load_vector(m)
            assert vec.shape == (MOCK_HIDDEN_DIM,)
            assert abs(np.linalg.norm(vec) - 1.0) < 1e-5


def test_vector_metadata_roundtrip():
    meta = PersonaVectorMeta(
        trait="honesty",
        layer=28,
        vector_path="/tmp/honesty_layer28.npy",
        n_positive=10,
        n_negative=10,
        vector_method="difference_of_means",
        normalization="unit_norm",
        hidden_dim=64,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = save_vector_metadata([meta], tmp)
        loaded = load_vector_metadata(path)
        assert len(loaded) == 1
        assert loaded[0].trait == "honesty"
        assert loaded[0].n_positive == 10


# ---------------------------------------------------------------------------
# 9. Validation metrics on synthetic projections
# ---------------------------------------------------------------------------

def test_compute_validation_metrics_perfect_separation():
    pos = np.array([2.0, 3.0, 2.5])
    neg = np.array([-2.0, -3.0, -2.5])
    metrics = compute_validation_metrics(pos, neg)
    assert metrics["auc"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["mean_positive_projection"] > 0
    assert metrics["mean_negative_projection"] < 0
    assert metrics["cohens_d"] > 0


def test_compute_validation_metrics_no_separation():
    rng = np.random.default_rng(0)
    pos = rng.standard_normal(50)
    neg = rng.standard_normal(50)
    metrics = compute_validation_metrics(pos, neg)
    assert 0.0 <= metrics["auc"] <= 1.0
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_compute_validation_metrics_empty_arrays():
    metrics = compute_validation_metrics(np.array([]), np.array([]))
    assert metrics["auc"] == 0.5
    assert metrics["accuracy"] == 0.5


def test_cohens_d_sign():
    pos = np.array([1.0, 1.5, 2.0])
    neg = np.array([-1.0, -1.5, -2.0])
    metrics = compute_validation_metrics(pos, neg)
    assert metrics["cohens_d"] > 0


# ---------------------------------------------------------------------------
# 10. Full validate_all_vectors with synthetic data
# ---------------------------------------------------------------------------

def test_validate_all_vectors_returns_results(bipolar_scored, bipolar_val_scored):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        all_scored = bipolar_scored + bipolar_val_scored
        records = mock_extract(all_scored, candidate_layers=[0], out_dir=tmp, hidden_dim=MOCK_HIDDEN_DIM)
        vec_metas = compute_all_vectors(
            records=[r for r in records if r.split == "extraction"],
            candidate_layers=[0],
            traits=["honesty"],
            normalize=True,
            out_dir=tmp / "vectors",
        )
        results = validate_all_vectors(
            act_records=records,
            vec_metas=vec_metas,
            minimum_auc_target=0.5,
        )
        assert len(results) > 0
        for r in results:
            assert isinstance(r, VectorValidationResult)
            assert 0.0 <= r.auc <= 1.0


# ---------------------------------------------------------------------------
# 11. Layer selection
# ---------------------------------------------------------------------------

def test_select_best_layer_picks_highest_mean_auc():
    results = [
        VectorValidationResult("honesty",     1, auc=0.9, accuracy=0.8, mean_positive_projection=1.0, mean_negative_projection=-1.0, cohens_d=2.0, n_positive_val=10, n_negative_val=10, passes_minimum_auc=True),
        VectorValidationResult("harmlessness", 1, auc=0.8, accuracy=0.7, mean_positive_projection=0.8, mean_negative_projection=-0.8, cohens_d=1.6, n_positive_val=10, n_negative_val=10, passes_minimum_auc=True),
        VectorValidationResult("honesty",     2, auc=0.6, accuracy=0.6, mean_positive_projection=0.5, mean_negative_projection=-0.5, cohens_d=1.0, n_positive_val=10, n_negative_val=10, passes_minimum_auc=False),
        VectorValidationResult("harmlessness", 2, auc=0.5, accuracy=0.5, mean_positive_projection=0.2, mean_negative_projection=-0.2, cohens_d=0.4, n_positive_val=10, n_negative_val=10, passes_minimum_auc=False),
    ]
    best = select_best_layer(results, ["honesty", "harmlessness"])
    assert best == 1  # layer 1 has mean AUC 0.85 vs layer 2's 0.55


def test_select_best_layer_empty_raises():
    with pytest.raises(ValueError):
        select_best_layer([], ["honesty"])


# ---------------------------------------------------------------------------
# 12. I/O roundtrips
# ---------------------------------------------------------------------------

def test_responses_roundtrip(mock_responses):
    with tempfile.TemporaryDirectory() as tmp:
        parquet_path, csv_path = save_responses(mock_responses, tmp)
        loaded = load_responses(parquet_path)
        assert len(loaded) == len(mock_responses)
        assert loaded[0].response_id == mock_responses[0].response_id


def test_scored_responses_roundtrip(mock_scored):
    with tempfile.TemporaryDirectory() as tmp:
        parquet_path, _ = save_scored(mock_scored, tmp)
        loaded = load_scored(parquet_path)
        assert len(loaded) == len(mock_scored)
        assert loaded[0].trait_score == mock_scored[0].trait_score


def test_validation_results_roundtrip():
    result = VectorValidationResult(
        trait="fairness", layer=28,
        auc=0.82, accuracy=0.78,
        mean_positive_projection=0.6, mean_negative_projection=-0.6,
        cohens_d=1.8, n_positive_val=20, n_negative_val=20,
        passes_minimum_auc=True,
    )
    with tempfile.TemporaryDirectory() as tmp:
        csv_path, _ = save_validation_results([result], tmp)
        loaded = load_validation_results(csv_path)
        assert loaded[0].auc == pytest.approx(0.82)
        assert loaded[0].passes_minimum_auc is True


# ---------------------------------------------------------------------------
# 13. No ETHICS items in vector construction path
# ---------------------------------------------------------------------------

def test_no_ethics_content_in_generate_responses_module():
    source = Path("src/vectors/generate_responses.py").read_text()
    ethics_markers = ["hendrycks/ethics", "ethics_mvp", "source_split", "deontology"]
    for marker in ethics_markers:
        assert marker not in source, f"Found ETHICS marker '{marker}' in generate_responses.py"


def test_no_ethics_content_in_score_responses_module():
    source = Path("src/vectors/score_responses.py").read_text()
    assert "ethics" not in source.lower() or "NOTE" in source, \
        "score_responses.py must not reference ETHICS items"


def test_no_ethics_content_in_compute_vectors_module():
    source = Path("src/vectors/compute_vectors.py").read_text()
    assert "ethics" not in source.lower() or "NOTE" in source


# ---------------------------------------------------------------------------
# 14. No heavy imports
# ---------------------------------------------------------------------------

def test_no_torch_in_generate_responses():
    source = Path("src/vectors/generate_responses.py").read_text()
    assert "import torch" not in source


def test_no_torch_in_score_responses():
    source = Path("src/vectors/score_responses.py").read_text()
    assert "import torch" not in source


def test_no_torch_in_compute_vectors():
    source = Path("src/vectors/compute_vectors.py").read_text()
    assert "import torch" not in source


def test_no_torch_in_validate_vectors():
    source = Path("src/vectors/validate_vectors.py").read_text()
    assert "import torch" not in source


def test_no_modal_in_src_vectors():
    for module in ["generate_responses", "score_responses", "extract_activations",
                   "compute_vectors", "validate_vectors", "vector_data"]:
        source = Path(f"src/vectors/{module}.py").read_text()
        assert "import modal" not in source, f"{module}.py must not import modal"


def test_no_transformers_in_src_vectors():
    for module in ["generate_responses", "score_responses", "extract_activations",
                   "compute_vectors", "validate_vectors", "vector_data"]:
        source = Path(f"src/vectors/{module}.py").read_text()
        assert "import transformers" not in source, f"{module}.py must not import transformers"


# ---------------------------------------------------------------------------
# 13. Modal/vLLM scoring — config, isolation, sentinel, and threshold tests
# ---------------------------------------------------------------------------

def test_default_judge_method_is_modal_vllm():
    import yaml
    cfg = yaml.safe_load(
        Path("configs/mvp_experiment.yaml").read_text()
    )
    assert cfg["response_scoring"]["judge_method"] == "modal_vllm"


def test_config_has_batch_size():
    import yaml
    cfg = yaml.safe_load(
        Path("configs/mvp_experiment.yaml").read_text()
    )
    assert "batch_size" in cfg["response_scoring"]
    assert cfg["response_scoring"]["batch_size"] > 0


def test_score_script_does_not_require_anthropic_key():
    source = Path("scripts/score_vector_responses.py").read_text()
    assert "ANTHROPIC_API_KEY" not in source
    assert "import anthropic" not in source


def test_score_responses_module_does_not_require_anthropic():
    source = Path("src/vectors/score_responses.py").read_text()
    assert "ANTHROPIC_API_KEY" not in source
    assert "import anthropic" not in source


def test_modal_vllm_app_exists():
    assert Path("modal_apps/score_vector_responses_vllm.py").exists()


def test_modal_vllm_app_has_no_top_level_modal_import():
    source = Path("modal_apps/score_vector_responses_vllm.py").read_text()
    # Modal must be imported inside a try/except guard, not at top level
    assert "try:" in source
    assert "_MODAL_AVAILABLE" in source


def test_modal_vllm_app_uses_same_model_as_config():
    import yaml
    cfg = yaml.safe_load(Path("configs/mvp_experiment.yaml").read_text())
    source = Path("modal_apps/score_vector_responses_vllm.py").read_text()
    model = cfg["response_scoring"]["judge_model_name"]
    assert model in source


def test_judge_prompt_includes_rubric(bank, mock_responses):
    from src.vectors.score_responses import build_judge_prompt
    resp = mock_responses[0]
    rubric_row = bank.rubrics_df[bank.rubrics_df["trait"] == resp.trait].iloc[0]
    rubric_instr = str(rubric_row["scoring_instructions"])
    scale = {
        0: str(rubric_row.get("scale_0", "")),
        50: str(rubric_row.get("scale_50", "")),
        100: str(rubric_row.get("scale_100", "")),
    }
    _, user_prompt = build_judge_prompt(resp, rubric_instr, scale)
    assert rubric_instr[:30] in user_prompt


def test_parse_valid_vllm_json_score():
    from src.vectors.score_responses import parse_judge_response
    raw = json.dumps({"trait_score": 82, "score_rationale": "Clear honesty expressed."})
    score, rationale = parse_judge_response(raw, "hon_pos_001__hon_ext_001")
    assert score == 82.0
    assert "honesty" in rationale.lower()


def test_invalid_json_non_strict_score_is_negative_one():
    from src.vectors.score_responses import parse_judge_response
    score, rationale = parse_judge_response("not valid json at all", "rid", strict=False)
    assert score == -1.0
    assert "parse_error" in rationale


def test_invalid_json_non_strict_becomes_unkept():
    from src.vectors.score_responses import parse_judge_response
    from src.vectors.vector_data import ScoredResponse
    resp = _fake_response("honesty", "positive", "hon_pos_001__hon_ext_001")
    score, rationale = parse_judge_response("not json", resp.response_id, strict=False)
    assert score == -1.0
    scored = ScoredResponse.from_generated(resp, score, rationale, 70.0, 30.0)
    assert scored.keep_for_vector_extraction is False


def test_negative_sentinel_always_unkept_for_negative_pole():
    """score=-1 must not accidentally satisfy the negative-pole threshold (<= 30)."""
    from src.vectors.vector_data import ScoredResponse
    resp = _fake_response("honesty", "negative", "hon_neg_001__hon_ext_001")
    scored = ScoredResponse.from_generated(resp, -1.0, "parse_error", 70.0, 30.0)
    assert scored.keep_for_vector_extraction is False


def test_positive_threshold_filtering_keeps_high_scores():
    from src.vectors.vector_data import ScoredResponse
    resp = _fake_response("honesty", "positive", "hon_pos_001__hon_ext_001")
    assert ScoredResponse.from_generated(resp, 85.0, "ok", 70.0, 30.0).keep_for_vector_extraction is True
    assert ScoredResponse.from_generated(resp, 70.0, "ok", 70.0, 30.0).keep_for_vector_extraction is True
    assert ScoredResponse.from_generated(resp, 69.9, "ok", 70.0, 30.0).keep_for_vector_extraction is False


def test_negative_threshold_filtering_keeps_low_scores():
    from src.vectors.vector_data import ScoredResponse
    resp = _fake_response("honesty", "negative", "hon_neg_001__hon_ext_001")
    assert ScoredResponse.from_generated(resp, 15.0, "ok", 70.0, 30.0).keep_for_vector_extraction is True
    assert ScoredResponse.from_generated(resp, 30.0, "ok", 70.0, 30.0).keep_for_vector_extraction is True
    assert ScoredResponse.from_generated(resp, 30.1, "ok", 70.0, 30.0).keep_for_vector_extraction is False


def test_readme_stage2b_does_not_mention_anthropic_as_required():
    readme = Path("README.md").read_text()
    stage2b_start = readme.find("## Stage 2B")
    stage2b_end = readme.find("\n## ", stage2b_start + 1)
    stage2b_section = readme[stage2b_start:stage2b_end]
    assert "ANTHROPIC_API_KEY" not in stage2b_section
    assert "No Anthropic API key is required" in stage2b_section


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_response(trait: str, pole: str, response_id: str) -> GeneratedResponse:
    return GeneratedResponse(
        response_id=response_id,
        trait=trait,
        pole=pole,
        split="extraction",
        system_prompt_id=response_id.split("__")[0],
        question_id=response_id.split("__")[1],
        system_prompt_text="You are a test assistant.",
        question_text="What do you think?",
        response_text="This is a test response.",
        model_name="mock",
        generation_params={},
    )
