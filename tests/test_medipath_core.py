import unittest

from medipath.agents import (
    food_recognition_agent,
    health_literacy_agent,
    prescription_digitization_agent,
)
from medipath.state import SharedState


class MediPathCoreTests(unittest.TestCase):
    def test_health_literacy_falls_back_when_source_not_found(self):
        state = SharedState(
            user_id="u1",
            language="English",
            input_type="text",
            intent="health",
            raw_input="Tell me about rare condition",
        )
        result = health_literacy_agent(state)
        self.assertIn("do not cover", result["answer"].lower())
        self.assertEqual(result["source"], "No source match")

    def test_prescription_extraction_requires_confirmation(self):
        out = prescription_digitization_agent("Paracetamol 500 mg BD 5 days")
        self.assertTrue(out["requires_user_confirmation"])
        self.assertTrue(out["extracted"])
        self.assertTrue(out["extracted"][0]["medication_name"])

    def test_food_recognition_low_confidence_requires_confirmation(self):
        out = food_recognition_agent("idli", 0.4)
        self.assertTrue(out["requires_user_confirmation"])


if __name__ == "__main__":
    unittest.main()
