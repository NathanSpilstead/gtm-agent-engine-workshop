import json
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ["LANGSMITH_TRACING"] = "false"

from gtm_agent import data_service, gtm_agent


class ProspectPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.original_profiles = data_service._PROFILES.copy()
        data_service._PROFILES.clear()

    def tearDown(self):
        data_service._PROFILES.clear()
        data_service._PROFILES.update(self.original_profiles)

    def assert_no_billing_data(self, value):
        serialized = json.dumps(value)
        self.assertNotIn("billing_qualification", serialized)
        for secret in ("TEST-TAX-ID", "TEST-DOB", "TEST-CARD-NUMBER", "TEST-CREDIT-REF"):
            self.assertNotIn(secret, serialized)

    def test_get_prospect_returns_only_contact_allowlist(self):
        billing = {
            "tax_id": "TEST-TAX-ID",
            "date_of_birth": "TEST-DOB",
            "card_on_file": "TEST-CARD-NUMBER",
            "credit_check_ref": "TEST-CREDIT-REF",
        }
        record = {
            "name": "Avery Customer",
            "email": "avery@example.com",
            "disqualified": False,
            "billing_qualification": billing,
            "engagement_history": ["private"],
            "account_details": ["private"],
            "tech_stack": ["private"],
        }
        with patch.object(data_service, "get_prospect_record", return_value=record):
            result = gtm_agent.get_prospect.func("LEAD-TEST")

        self.assertEqual(
            result["prospect"],
            {
                "prospect_id": "LEAD-TEST",
                "name": "Avery Customer",
                "email": "avery@example.com",
                "disqualified": False,
            },
        )
        self.assert_no_billing_data(result)

    def test_fresh_profile_is_sanitized_before_caching(self):
        billing = {
            "tax_id": "TEST-TAX-ID",
            "date_of_birth": "TEST-DOB",
            "card_on_file": "TEST-CARD-NUMBER",
            "credit_check_ref": "TEST-CREDIT-REF",
        }
        record = {
            "name": "Avery Customer",
            "email": "avery@example.com",
            "billing_qualification": billing,
            "annual_revenue": 90000000,
        }
        with (
            patch.object(data_service, "get_prospect_record", return_value=record),
            patch.object(data_service, "fetch_engagement_history", return_value=[]),
            patch.object(data_service, "fetch_account_details", return_value=[]),
            patch.object(data_service, "fetch_tech_stack", return_value=["Kafka"]),
        ):
            result = gtm_agent.build_prospect_profile.func("LEAD-TEST")

        self.assert_no_billing_data(result)
        self.assert_no_billing_data(data_service._PROFILES["LEAD-TEST"])

    def test_cached_profile_is_sanitized_at_retrieval_boundary(self):
        data_service._PROFILES["LEAD-TEST"] = {
            "prospect_id": "LEAD-TEST",
            "billing_qualification": {
                "tax_id": "TEST-TAX-ID",
                "date_of_birth": "TEST-DOB",
                "card_on_file": "TEST-CARD-NUMBER",
                "credit_check_ref": "TEST-CREDIT-REF",
            },
            "annual_revenue": 90000000,
        }

        result = data_service.get_profile_from_db("LEAD-TEST")
        tool_result = gtm_agent.build_prospect_profile.func("LEAD-TEST")

        self.assert_no_billing_data(result)
        self.assert_no_billing_data(tool_result)
        self.assert_no_billing_data(data_service._PROFILES["LEAD-TEST"])

    def test_scoring_llm_receives_only_scoring_fields(self):
        profile = {
            "prospect_id": "LEAD-TEST",
            "annual_revenue": 90000000,
            "tech_stack": ["Kafka"],
            "account_details": [{"industry": "Software"}],
            "engagement_history": [{"channel": "Demo"}],
            "billing_qualification": {
                "tax_id": "TEST-TAX-ID",
                "date_of_birth": "TEST-DOB",
                "card_on_file": "TEST-CARD-NUMBER",
                "credit_check_ref": "TEST-CREDIT-REF",
            },
            "name": "Avery Customer",
        }
        offering = {
            "required_tech_stack": ["Kafka"],
            "min_annual_revenue": 1000000,
            "description": "Platform",
        }
        model_result = type("Result", (), {"model_dump": lambda self: {"score": 90}})()
        with (
            patch.object(data_service, "fetch_tech_stack", return_value=["Kafka"]),
            patch.object(gtm_agent, "_scoring_llm") as scoring_llm,
        ):
            scoring_llm.invoke.return_value = model_result
            gtm_agent.score_prospect.func(profile, offering)

        messages = scoring_llm.invoke.call_args.args[0]
        user_content = messages[1]["content"]
        scoring_input = json.loads(user_content.split("\n\nProspect profile:\n", 1)[1])
        self.assertEqual(
            set(scoring_input),
            {"annual_revenue", "tech_stack", "account_details", "engagement_history"},
        )
        self.assert_no_billing_data(user_content)


if __name__ == "__main__":
    unittest.main()
