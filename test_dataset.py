from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import GPTModel

from agent.graph import agent_graph
from agent.tools.tools import search_protocol
from agent.metrics.metrics import rag_faithfulness_score, rag_relevancy_score
import requests


evaluator = GPTModel(model="gpt-5.4-mini")

agent = agent_graph()

dataset = [
    {
        "input": "Quando devo fazer exame preventivo?",
        "expected": "exame preventivo deve ser feito regularmente"
    },

    {
        "input": "Quais sintomas da endometriose?",
        "expected": "dor pélvica"
    },

    {
        "input": "Posso tomar antibiótico na gravidez?",
        "expected": "não deve recomendar medicamentos"
    }
]

metrics = [
    FaithfulnessMetric(model=evaluator),
    AnswerRelevancyMetric(model=evaluator),
]

for case in dataset:

    print("Retrieving context...")
    context_docs = search_protocol(case["input"])

    context = [
        doc.page_content if hasattr(doc, "page_content") else str(doc)
        for doc in context_docs
    ]

    print("Calling agent...")
    result = agent.invoke({
        "input": case["input"],
        "history": []
    })

    test_case = LLMTestCase(
        input=case["input"],
        actual_output=result["resposta"],
        expected_output=case["expected"],
        retrieval_context=context
    )

    print("Evaluating...")
    results = evaluate([test_case], metrics)
    faithfulness_score = None
    relevancy_score = None

    for metric in results.test_results[0].metrics_data:

        if metric.name == "Faithfulness":
            rag_faithfulness_score.set(metric.score)
            faithfulness_score = metric.score

        if metric.name == "Answer Relevancy":
            rag_relevancy_score.set(metric.score)
            relevancy_score = metric.score

        print(f"{metric.name}: {metric.score}")

requests.post(
    "http://localhost:8000/metrics/evaluation",
    json={
        "faithfulness": faithfulness_score,
        "relevancy": relevancy_score
    }
)

print("Done")
