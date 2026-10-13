import os
import unittest

os.environ.setdefault("OPENAI_API_KEY", "test")

from gtm_agent.gtm_agent import SYSTEM_PROMPT, send_prospect_email


class EmailCollateralTests(unittest.TestCase):
    def test_document_request_without_link_requires_followup(self):
        self.assertIn("If they request a document but have not provided a link", SYSTEM_PROMPT)
        self.assertIn("ask the rep for a link, and do not send the email yet", SYSTEM_PROMPT)
        self.assertIn("plain-text email only", send_prospect_email.description)
        self.assertIn("files cannot be attached", send_prospect_email.description)

    def test_provided_document_link_is_sent_without_attachment_claim(self):
        self.assertIn("Once they provide a link, include it in the email", SYSTEM_PROMPT)
        self.assertIn("do not claim a file is attached", SYSTEM_PROMPT)
        message = send_prospect_email.func(
            prospect={"name": "Taylor Prospect", "email": "taylor@example.com"},
            subject="Pricing deck",
            body="Here is our pricing deck: https://example.com/pricing.pdf",
            runtime=None,
            from_rep={"name": "Rep", "email": "rep@example.com"},
        )
        self.assertEqual(message["status"], "sent")
        self.assertIn("https://example.com/pricing.pdf", message["body"])
        self.assertNotIn("attached", message["body"].lower())

    def test_ordinary_email_still_sends(self):
        self.assertIn("For other email requests, send the email as requested", SYSTEM_PROMPT)
        message = send_prospect_email.func(
            prospect={"name": "Taylor Prospect", "email": "taylor@example.com"},
            subject="Scheduling a call",
            body="Would you be available for a call next week?",
            runtime=None,
            from_rep={"name": "Rep", "email": "rep@example.com"},
        )
        self.assertEqual(message["status"], "sent")
        self.assertEqual(message["body"], "Would you be available for a call next week?")


if __name__ == "__main__":
    unittest.main()
