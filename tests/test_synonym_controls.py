"""
Tests for synonym vector convergent-validity controls.
All CPU-only — no torch, Modal, or transformers.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TRAITS = ["honesty", "harmlessness", "fairness", "compassion"]
CONFIG_PATH = "configs/synonym_vector_artifacts.yaml"

REQUIRED_TRAIT_FIELDS = ["parent_trait", "construct_name", "abbreviation"]
REQUIRED_TOP_FIELDS = [
    "positive_system_prompts",
    "negative_system_prompts",
    "elicitation_questions",
    "evaluation_rubric",
]
MVP_SYNONYMS = ["truthfulness", "harm_avoidance", "impartiality", "empathy"]
EXPECTED_PARENTS = {
    "truthfulness": "honesty",
    "harm_avoidance": "harmlessness",
    "impartiality": "fairness",
    "empathy": "compassion",
}


# ---------------------------------------------------------------------------
# Config schema validation
# ---------------------------------------------------------------------------

class TestSynonymConfigSchema:
    @pytest.fixture(autouse=True)
    def load_config(self):
        import yaml
        with open(CONFIG_PATH) as f:
            self.raw_config = yaml.safe_load(f)

    def test_all_mvp_synonyms_present(self):
        for synonym_id in MVP_SYNONYMS:
            assert synonym_id in self.raw_config, f"Missing synonym: {synonym_id}"

    def test_required_trait_fields_present(self):
        for synonym_id in MVP_SYNONYMS:
            entry = self.raw_config[synonym_id]
            for field in REQUIRED_TRAIT_FIELDS:
                assert field in entry, f"{synonym_id} missing field: {field}"
                assert entry[field], f"{synonym_id}.{field} is empty"

    def test_required_content_fields_present(self):
        for synonym_id in MVP_SYNONYMS:
            entry = self.raw_config[synonym_id]
            for field in REQUIRED_TOP_FIELDS:
                assert field in entry, f"{synonym_id} missing: {field}"

    def test_parent_trait_values(self):
        for synonym_id, expected_parent in EXPECTED_PARENTS.items():
            actual = self.raw_config[synonym_id]["parent_trait"]
            assert actual == expected_parent, (
                f"{synonym_id}: expected parent={expected_parent}, got {actual}"
            )

    def test_prompt_counts(self):
        for synonym_id in MVP_SYNONYMS:
            entry = self.raw_config[synonym_id]
            pos = entry.get("positive_system_prompts", [])
            neg = entry.get("negative_system_prompts", [])
            assert len(pos) == 5, f"{synonym_id}: expected 5 positive prompts, got {len(pos)}"
            assert len(neg) == 5, f"{synonym_id}: expected 5 negative prompts, got {len(neg)}"

    def test_elicitation_question_count(self):
        for synonym_id in MVP_SYNONYMS:
            entry = self.raw_config[synonym_id]
            questions = entry.get("elicitation_questions", [])
            assert len(questions) == 40, (
                f"{synonym_id}: expected 40 elicitation questions, got {len(questions)}"
            )

    def test_elicitation_splits(self):
        for synonym_id in MVP_SYNONYMS:
            entry = self.raw_config[synonym_id]
            questions = entry.get("elicitation_questions", [])
            extraction = [q for q in questions if q.get("split") == "extraction"]
            validation = [q for q in questions if q.get("split") == "validation"]
            assert len(extraction) == 20, f"{synonym_id}: expected 20 extraction, got {len(extraction)}"
            assert len(validation) == 20, f"{synonym_id}: expected 20 validation, got {len(validation)}"

    def test_evaluation_rubric_scale(self):
        for synonym_id in MVP_SYNONYMS:
            entry = self.raw_config[synonym_id]
            rubric = entry.get("evaluation_rubric", {})
            assert rubric.get("scale") == "0-100", f"{synonym_id}: rubric scale should be 0-100"


# ---------------------------------------------------------------------------
# load_synonym_config
# ---------------------------------------------------------------------------

class TestLoadSynonymConfig:
    def test_returns_correct_parent_trait_mapping(self):
        from src.controls.synonym_vectors import load_synonym_config
        config = load_synonym_config(CONFIG_PATH)
        for synonym_id, expected_parent in EXPECTED_PARENTS.items():
            assert synonym_id in config, f"Missing: {synonym_id}"
            assert config[synonym_id]["parent_trait"] == expected_parent

    def test_returns_all_mvp_synonyms(self):
        from src.controls.synonym_vectors import load_synonym_config
        config = load_synonym_config(CONFIG_PATH)
        for synonym_id in MVP_SYNONYMS:
            assert synonym_id in config

    def test_construct_name_present(self):
        from src.controls.synonym_vectors import load_synonym_config
        config = load_synonym_config(CONFIG_PATH)
        for synonym_id in MVP_SYNONYMS:
            assert "construct_name" in config[synonym_id]
            assert config[synonym_id]["construct_name"]


# ---------------------------------------------------------------------------
# run_synonym_similarity_analysis with synthetic vectors
# ---------------------------------------------------------------------------

class TestSynonymSimilarityAnalysis:
    def _make_orthonormal_originals(self):
        """4 orthonormal vectors (one-hot style in 4D)."""
        return {t: np.eye(4)[i] for i, t in enumerate(TRAITS)}

    def test_closest_matches_parent_true_when_aligned(self):
        from src.controls.synonym_vectors import run_synonym_similarity_analysis
        originals = self._make_orthonormal_originals()
        # Synonym vectors = parent + small noise → should match parent
        rng = np.random.default_rng(0)
        synonym_vecs = {}
        for synonym_id, parent in EXPECTED_PARENTS.items():
            parent_vec = originals[parent].copy()
            noisy = parent_vec + rng.standard_normal(4) * 0.05
            synonym_vecs[synonym_id] = {"vector": noisy, "parent_trait": parent}

        df = run_synonym_similarity_analysis(synonym_vecs, originals)
        assert df["closest_matches_parent"].all(), (
            f"All synonyms with small noise should match parent:\n{df}"
        )

    def test_closest_matches_parent_false_when_misaligned(self):
        from src.controls.synonym_vectors import run_synonym_similarity_analysis
        originals = self._make_orthonormal_originals()
        # Synonym vector aligned with wrong trait
        synonym_vecs = {
            "truthfulness": {
                "vector": originals["harmlessness"],  # wrong parent
                "parent_trait": "honesty",
            }
        }
        df = run_synonym_similarity_analysis(synonym_vecs, originals)
        assert not df["closest_matches_parent"].any()

    def test_output_columns(self):
        from src.controls.synonym_vectors import run_synonym_similarity_analysis
        originals = self._make_orthonormal_originals()
        synonym_vecs = {"truthfulness": {"vector": originals["honesty"], "parent_trait": "honesty"}}
        df = run_synonym_similarity_analysis(synonym_vecs, originals)
        expected_cols = {"synonym_id", "parent_trait", "closest_parent", "closest_matches_parent"}
        for trait in TRAITS:
            expected_cols.add(f"cosine_{trait}")
        assert expected_cols.issubset(set(df.columns))


# ---------------------------------------------------------------------------
# compute_projection_agreement
# ---------------------------------------------------------------------------

class TestProjectionAgreement:
    def test_perfectly_correlated_pearson_near_one(self):
        from src.controls.synonym_vectors import compute_projection_agreement
        x = pd.Series(np.linspace(0, 10, 100))
        y = x * 3 - 1
        result = compute_projection_agreement(x, y)
        assert abs(result["pearson_r"] - 1.0) < 1e-9
        assert abs(result["spearman_r"] - 1.0) < 1e-9
        assert abs(result["mean_abs_dev"]) > 0  # different scales → nonzero MAD

    def test_anticorrelated(self):
        from src.controls.synonym_vectors import compute_projection_agreement
        x = pd.Series(np.linspace(0, 10, 100))
        y = -x
        result = compute_projection_agreement(x, y)
        assert result["pearson_r"] < -0.99

    def test_handles_nan(self):
        from src.controls.synonym_vectors import compute_projection_agreement
        x = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
        y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_projection_agreement(x, y)
        assert not np.isnan(result["pearson_r"])


# ---------------------------------------------------------------------------
# save_synonym_controls
# ---------------------------------------------------------------------------

class TestSaveSynonymControls:
    def test_saves_expected_files(self, tmp_path):
        from src.controls.synonym_vectors import save_synonym_controls

        sim_df = pd.DataFrame({
            "synonym_id": ["truthfulness"],
            "parent_trait": ["honesty"],
            "cosine_honesty": [0.9],
            "cosine_harmlessness": [0.1],
            "cosine_fairness": [0.05],
            "cosine_compassion": [0.05],
            "closest_parent": ["honesty"],
            "closest_matches_parent": [True],
        })
        agree_df = pd.DataFrame({
            "synonym_id": ["truthfulness"],
            "parent_trait": ["honesty"],
            "layer": [32],
            "pearson_r": [0.95],
            "spearman_r": [0.94],
            "mean_abs_dev": [0.12],
        })
        save_synonym_controls(
            {"similarity_df": sim_df, "agreement_df": agree_df},
            tmp_path,
        )
        assert (tmp_path / "synonym_cosine_similarity.csv").exists()
        assert (tmp_path / "synonym_projection_agreement.csv").exists()
        assert (tmp_path / "synonym_controls_report.md").exists()

    def test_report_contains_closest_parent_info(self, tmp_path):
        from src.controls.synonym_vectors import save_synonym_controls
        sim_df = pd.DataFrame({
            "synonym_id": ["empathy"],
            "parent_trait": ["compassion"],
            "cosine_honesty": [0.1],
            "cosine_harmlessness": [0.1],
            "cosine_fairness": [0.1],
            "cosine_compassion": [0.9],
            "closest_parent": ["compassion"],
            "closest_matches_parent": [True],
        })
        save_synonym_controls({"similarity_df": sim_df, "agreement_df": None}, tmp_path)
        report = (tmp_path / "synonym_controls_report.md").read_text()
        assert "1/1" in report  # 1 of 1 match
