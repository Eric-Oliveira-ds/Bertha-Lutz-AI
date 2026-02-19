from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input="Quando devo fazer exame preventivo?",
    actual_output="...",
    expected_output="Resposta baseada em diretrizes oficiais"
)

metric = FaithfulnessMetric()

assert_test(test_case, [metric])
