import json
import os
import unittest
from io import BytesIO
from unittest.mock import patch

from core import ai_client


class AIClientTests(unittest.TestCase):
    def setUp(self):
        ai_client._status_cache.update({"checked_at": 0.0, "base_url": "", "status": None})
        ai_client._last_call = None

    def test_missing_cloud_configuration_uses_local_rule_intake(self):
        with patch.dict(os.environ, {"AI_BASE_URL": "", "AI_API_KEY": ""}, clear=False):
            result = ai_client.ask_ai("Cerco ripetizioni di matematica nel weekend")
        self.assertEqual(result["backend"], "local")
        parsed = json.loads(result["text"])
        self.assertEqual(parsed["category"], "Education")
        self.assertEqual(parsed["availability"], "weekend")
        self.assertEqual(parsed["language"], "it")

    def test_cloud_request_sends_api_key_and_returns_amd_result(self):
        body = BytesIO(b'{"text":"response","latency_s":0.3,"tokens":4,"model":"test-model"}')
        with patch.dict(os.environ, {"AI_BASE_URL": "http://127.0.0.1:8000", "AI_API_KEY": "test-secret"}), \
                patch("core.ai_client.urlopen", return_value=body) as open_request:
            result = ai_client.ask_ai("hello", max_new_tokens=40, timeout=3)
        self.assertEqual(result["backend"], "amd")
        self.assertEqual(result["metrics"]["latency_s"], 0.3)
        self.assertEqual(result["metrics"]["tokens_per_second"], 13.33)
        sent_request = open_request.call_args.args[0]
        self.assertEqual(sent_request.get_header("X-api-key"), "test-secret")
        self.assertEqual(json.loads(sent_request.data)["max_new_tokens"], 40)
        self.assertEqual(open_request.call_args.kwargs["timeout"], 3)

    def test_cloud_failure_retries_once_then_falls_back(self):
        with patch.dict(os.environ, {"AI_BASE_URL": "http://127.0.0.1:8000", "AI_API_KEY": "test-secret"}), \
                patch("core.ai_client.urlopen", side_effect=OSError("offline")) as open_request:
            result = ai_client.ask_ai("Math tutoring Saturday")
        self.assertEqual(open_request.call_count, 2)
        self.assertEqual(result["backend"], "local")
        self.assertEqual(json.loads(result["text"])["category"], "Education")

    def test_image_fallback_is_explicit_about_not_analyzing_image(self):
        with patch.dict(os.environ, {"AI_BASE_URL": "", "AI_API_KEY": ""}, clear=False):
            result = ai_client.ask_ai("", image_base64="not-uploaded")
        self.assertIn("Image was not analyzed", json.loads(result["text"])["summary"])

    def test_status_reports_amd_only_when_health_is_ready(self):
        response = BytesIO(b'{"ready":true,"authorized":true,"model":"test-model"}')
        with patch.dict(os.environ, {"AI_BASE_URL": "http://127.0.0.1:8000", "AI_API_KEY": "test-secret"}), \
                patch("core.ai_client.urlopen", return_value=response) as open_request:
            status = ai_client.get_backend_status()
        self.assertEqual(status["backend"], "amd")
        self.assertEqual(status["model"], "test-model")
        self.assertIsNone(status["last_call"])
        self.assertEqual(open_request.call_args.args[0].get_header("X-api-key"), "test-secret")

    def test_nonlocal_http_endpoint_is_not_used(self):
        with patch.dict(os.environ, {"AI_BASE_URL": "http://example.test", "AI_API_KEY": "test-secret"}), \
                patch("core.ai_client.urlopen") as open_request:
            result = ai_client.ask_ai("Math tutoring Saturday")
        self.assertEqual(result["backend"], "local")
        open_request.assert_not_called()

    def test_unaccepted_health_key_keeps_local_backend_status(self):
        response = BytesIO(b'{"ready":true,"authorized":false,"model":"test-model"}')
        with patch.dict(os.environ, {"AI_BASE_URL": "http://127.0.0.1:8000", "AI_API_KEY": "wrong-key"}), \
                patch("core.ai_client.urlopen", return_value=response):
            status = ai_client.get_backend_status()
        self.assertEqual(status["backend"], "local")


if __name__ == "__main__":
    unittest.main()
