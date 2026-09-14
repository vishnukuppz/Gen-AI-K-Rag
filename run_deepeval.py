import os
import sys
from dotenv import load_dotenv
from tabulate import tabulate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
)
from krag_retrival import krag_retrieval_pipeline

load_dotenv()
MODEL_NAME = os.getenv("OPENAI_GENERATIVE_MODEL", "gpt-4o-mini")


def run_evaluation_suite():
    print("=" * 80)
    print("🚀 Running DeepEval Evaluation Test Suite for K-RAG Pipeline")
    print(f"Evaluation Model: {MODEL_NAME}")
    print("=" * 80)

    # Initialize metrics with 0.7 pass threshold
    relevancy_metric = AnswerRelevancyMetric(threshold=0.7, model=MODEL_NAME)
    faithfulness_metric = FaithfulnessMetric(threshold=0.7, model=MODEL_NAME)
    context_metric = ContextualRelevancyMetric(threshold=0.7, model=MODEL_NAME)

    test_queries = [
        "Explain schema which has helps watering the crops?",
        "What schemes provide loans for sericulture in irrigated area?",
    ]

    report_rows = []

    for idx, query in enumerate(test_queries, 1):
        print(f"\n[Test Case {idx}/{len(test_queries)}] Query: '{query}'")
        print("-> Running K-RAG retrieval pipeline...")
        result = krag_retrieval_pipeline(query, verbose=False)

        actual_output = result["step_4_execution"]["final_answer"]
        raw_records = result["step_4_execution"].get("raw_results", [])

        retrieval_context = [
            f"Scheme: {r.get('scheme_name')}, Rel: {r.get('relationship')}, Target: {r.get('target_entity')}, Doc: {r.get('document_text')}"
            for r in raw_records
        ]
        if not retrieval_context:
            retrieval_context = ["No graph context found."]

        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output,
            retrieval_context=retrieval_context,
        )

        print("-> Measuring Answer Relevancy...")
        relevancy_metric.measure(test_case)

        print("-> Measuring Faithfulness (Hallucination Check)...")
        faithfulness_metric.measure(test_case)

        report_rows.append([
            f"TC {idx}: {query[:30]}...",
            "Answer Relevancy",
            f"{relevancy_metric.score:.2f}",
            "✅ PASS" if relevancy_metric.is_successful() else "❌ FAIL",
            relevancy_metric.reason[:70] + "..." if relevancy_metric.reason else "",
        ])

        report_rows.append([
            f"TC {idx}: {query[:30]}...",
            "Faithfulness",
            f"{faithfulness_metric.score:.2f}",
            "✅ PASS" if faithfulness_metric.is_successful() else "❌ FAIL",
            faithfulness_metric.reason[:70] + "..." if faithfulness_metric.reason else "",
        ])

    print("\n" + "=" * 80)
    print("📊 DeepEval Test Suite Summary Report:")
    print("=" * 80)
    headers = ["Test Case", "Metric", "Score", "Result", "Reason"]
    print(tabulate(report_rows, headers=headers, tablefmt="grid"))
    print("=" * 80)


if __name__ == "__main__":
    run_evaluation_suite()
