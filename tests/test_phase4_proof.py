import json
import unittest
from click.testing import CliRunner
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import communityconnect
from core import ai_client

ROOT = Path(__file__).resolve().parents[1]


class PhaseFourProofTests(unittest.TestCase):
    def setUp(self):
        self.original_secret = communityconnect.app.config.get("SECRET_KEY")
        self.original_testing = communityconnect.app.config.get("TESTING")
        communityconnect.app.config["SECRET_KEY"] = "test-session-secret-value-at-least-32-chars"
        communityconnect.app.config["TESTING"] = True
        self.client = communityconnect.app.test_client()

    def tearDown(self):
        communityconnect.app.config["SECRET_KEY"] = self.original_secret
        communityconnect.app.config["TESTING"] = self.original_testing

    def test_labeled_evaluation_set_has_30_to_50_bilingual_rows_and_emergencies(self):
        cases = json.loads((ROOT / "data" / "evaluation_requests.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(cases), 30)
        self.assertLessEqual(len(cases), 50)
        self.assertEqual({case["language"] for case in cases}, {"it", "en"})
        self.assertTrue(any(case["emergency"] for case in cases))

    def test_demo_seed_creates_40_flagged_providers_and_30_flagged_requests(self):
        profile_select = Mock()
        profile_select.eq.return_value = profile_select
        profile_select.execute.return_value = SimpleNamespace(data=[])
        profile_upsert = Mock()
        profile_upsert.execute.return_value = SimpleNamespace(data=[])
        profiles_table = Mock()
        profiles_table.select.return_value = profile_select
        profiles_table.upsert.return_value = profile_upsert

        request_delete = Mock()
        request_delete.eq.return_value = request_delete
        request_delete.execute.return_value = SimpleNamespace(data=[])
        request_insert = Mock()
        request_insert.execute.return_value = SimpleNamespace(data=[])
        requests_table = Mock()
        requests_table.delete.return_value = request_delete
        requests_table.insert.return_value = request_insert

        admin = Mock()
        admin.list_users.return_value = []
        admin.create_user.side_effect = lambda payload: SimpleNamespace(
            user=SimpleNamespace(id=f"demo-user-{admin.create_user.call_count}")
        )
        service_client = SimpleNamespace(
            auth=SimpleNamespace(admin=admin),
            table=lambda name: {"profiles": profiles_table, "help_requests": requests_table}[name],
        )
        with patch("communityconnect.get_service_client", return_value=service_client), \
                patch.object(communityconnect, "DEMO_PASSWORD", "long-enough-demo-password"):
            result = CliRunner().invoke(communityconnect.app.cli, ["seed-demo"])
        self.assertEqual(result.exit_code, 0, result.output)

        profiles = [call.args[0] for call in profiles_table.upsert.call_args_list]
        requests = [call.args[0] for call in requests_table.insert.call_args_list]
        providers = [profile for profile in profiles if profile["role"] in communityconnect.HELPER_ROLES]
        self.assertEqual(len(providers), 40)
        self.assertEqual(len(requests), 30)
        self.assertTrue(all(profile["is_demo"] for profile in profiles))
        self.assertTrue(all(item["is_demo"] for item in requests))

    def test_service_directory_labels_demo_profiles(self):
        offer = {
            "category": "Education", "provider_type": "Volunteer",
            "profile": {
                "id": "demo-helper", "name": "Tutor Demo", "role": "volunteer",
                "skills": "Ripetizioni di matematica", "location": "Genova Centro",
                "availability": "Weekend", "bio": "Tutor sintetico.", "is_demo": True,
            },
        }
        with self.client.session_transaction() as session:
            session["language"] = "en"
        with patch("communityconnect.current_user", return_value=None), \
                patch("communityconnect.get_service_offers", return_value=[offer]), \
                patch("communityconnect.translate_content", side_effect=lambda text, language: text):
            response = self.client.get("/services")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Demo data", response.data)

    def test_dashboard_status_exposes_latest_ai_metrics(self):
        ai_client._last_call = {
            "backend": "amd", "model": "test-model", "latency_s": 0.25,
            "tokens": 10, "tokens_per_second": 40.0, "operation": "generate",
        }
        status = ai_client.get_backend_status()
        self.assertEqual(status["last_call"]["latency_s"], 0.25)
        self.assertEqual(status["last_call"]["tokens_per_second"], 40.0)

    def test_reset_demo_deletes_only_rows_marked_demo(self):
        profile_select = Mock()
        profile_select.eq.return_value = profile_select
        profile_select.execute.return_value = SimpleNamespace(data=[{"id": "demo-user-1"}])
        profile_delete = Mock()
        profile_delete.eq.return_value = profile_delete
        profile_delete.execute.return_value = SimpleNamespace(data=[])
        profiles_table = Mock()
        profiles_table.select.return_value = profile_select
        profiles_table.delete.return_value = profile_delete

        request_delete = Mock()
        request_delete.eq.return_value = request_delete
        request_delete.execute.return_value = SimpleNamespace(data=[])
        requests_table = Mock()
        requests_table.delete.return_value = request_delete

        admin = Mock()
        service_client = SimpleNamespace(
            auth=SimpleNamespace(admin=admin),
            table=lambda name: {"profiles": profiles_table, "help_requests": requests_table}[name],
        )
        with patch("communityconnect.get_service_client", return_value=service_client):
            result = CliRunner().invoke(communityconnect.app.cli, ["reset-demo", "--yes"])
        self.assertEqual(result.exit_code, 0, result.output)
        request_delete.eq.assert_called_once_with("is_demo", True)
        profile_select.eq.assert_called_once_with("is_demo", True)
        admin.delete_user.assert_called_once_with("demo-user-1")


if __name__ == "__main__":
    unittest.main()
