import os
import sys
from pathlib import Path
import pytest
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
)
from krag_retrival import krag_retrieval_pipeline


load_dotenv()
EVAL_MODEL = os.getenv("OPENAI_GENERATIVE_MODEL", "gpt-4o-mini")


@pytest.fixture(scope="module")
def relevancy_metric():
    return AnswerRelevancyMetric(threshold=0.7, model=EVAL_MODEL)


@pytest.fixture(scope="module")
def faithfulness_metric():
    return FaithfulnessMetric(threshold=0.7, model=EVAL_MODEL)


@pytest.fixture(scope="module")
def contextual_metric():
    return ContextualRelevancyMetric(threshold=0.7, model=EVAL_MODEL)


def test_irrigation_scheme_relevancy(relevancy_metric, faithfulness_metric):
    """
    Test that questions about crop watering retrieve relevant and faithful answers.
    """
    query = "Explain schema which has helps watering the crops?"
    result = krag_retrieval_pipeline(query, verbose=False)

    actual_output = result["step_4_execution"]["final_answer"]
    raw_results = result["step_4_execution"].get("raw_results", [])
    
    # Extract retrieval context lines from raw Neo4j results
    retrieval_context = [
        f"Scheme: {r.get('scheme_name')}, Rel: {r.get('relationship')}, Target: {r.get('target_entity')}, Doc: {r.get('document_text')}"
        for r in raw_results
    ] or ["Primary Cooperative Agriculture and Rural Development Bank minor irrigation loans."]

    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
    )

    assert_test(test_case, [relevancy_metric, faithfulness_metric])


def test_sericulture_loans(relevancy_metric, faithfulness_metric):
    """
    Test that queries about sericulture loans are accurately and faithfully answered.
    """
    query = "What schemes provide loans for sericulture in irrigated area?"
    result = krag_retrieval_pipeline(query, verbose=False)

    actual_output = result["step_4_execution"]["final_answer"]
    raw_results = result["step_4_execution"].get("raw_results", [])
    
    retrieval_context = [
        f"Scheme: {r.get('scheme_name')}, Rel: {r.get('relationship')}, Target: {r.get('target_entity')}, Doc: {r.get('document_text')}"
        for r in raw_results
    ] or ["Sericulture in irrigated area loan scheme."]

    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
    )

    assert_test(test_case, [relevancy_metric, faithfulness_metric])
