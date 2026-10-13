import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from gtm_agent.gtm_agent import SYSTEM_PROMPT, send_prospect_email


def _send(prospect):
    return send_prospect_email.func(
        prospect=prospect,
        subject="Demo invitation",
        body="Let's book a demo.",
        runtime=SimpleNamespace(config={"metadata": {}}),
        from_rep={"email": "rep@example.com", "name": "Sales Rep"},
    )


class SendProspectEmailTests(unittest.TestCase):
    def test_disqualified_prospect_is_blocked_without_message_id(self):
        with patch("gtm_agent.gtm_agent.data_service.get_prospect_record", return_value={"disqualified": True}):
            result = _send({"prospect_id": "LEAD-50001", "email": "lead@example.com"})

        self.assertEqual(result, {"status": "blocked", "error": "Prospect is disqualified; email not sent."})

    def test_missing_prospect_record_is_blocked_without_claiming_disqualification(self):
        with patch("gtm_agent.gtm_agent.data_service.get_prospect_record", return_value=None):
            result = _send({"prospect_id": "LEAD-00000", "email": "lead@example.com"})

        self.assertEqual(result["status"], "blocked")
        self.assertIn("not found", result["error"])
        self.assertNotIn("disqualified", result["error"])
        self.assertNotIn("message_id", result)

    def test_eligible_prospect_with_email_is_sent(self):
        with patch("gtm_agent.gtm_agent.data_service.get_prospect_record", return_value={"disqualified": False}):
            result = _send({"prospect_id": "LEAD-18993", "email": "lead@example.com"})

        self.assertEqual(result["status"], "sent")
        self.assertTrue(result["message_id"].startswith("msg-"))
        self.assertEqual(result["to"], "lead@example.com")

    def test_prompt_explains_blocked_sends_without_claiming_success(self):
        self.assertIn("Explain accurately whether a send was blocked", SYSTEM_PROMPT)
        self.assertIn("the record was not found", SYSTEM_PROMPT)
        self.assertIn("never claim a blocked email was sent", SYSTEM_PROMPT)
