import os
import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.parse import parse_qs

import communityconnect
from core.scoring import WEIGHTS


class PhaseOneSecurityTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {
            "SUPABASE_URL": "https://example.supabase.co",
            "SUPABASE_PUBLISHABLE_KEY": "test-publishable-key",
            "SUPABASE_SECRET_KEY": "test-secret-key",
        })
        self.environment.start()
        self.original_secret = communityconnect.app.config.get("SECRET_KEY")
        communityconnect.app.config["SECRET_KEY"] = "test-session-secret-value-at-least-32-chars"
        communityconnect.app.config["TESTING"] = True
        self.client = communityconnect.app.test_client()

    def tearDown(self):
        communityconnect.app.config["SECRET_KEY"] = self.original_secret
        communityconnect.app.config["TESTING"] = False
        self.environment.stop()

    def set_csrf(self, user_id=None):
        with self.client.session_transaction() as session:
            session["_csrf_token"] = "test-csrf-token"
            if user_id:
                session["user_id"] = user_id

    def test_authenticated_login_page_redirects_to_dashboard(self):
        with patch("communityconnect.current_user", return_value={"id": "demo-user", "role": "helper"}):
            response = self.client.get("/login")
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/dashboard")

    def test_stale_duplicate_login_post_redirects_after_authentication(self):
        with self.client.session_transaction() as session:
            session["user_id"] = "demo-user"
        with patch("communityconnect.get_auth_client") as get_auth_client:
            response = self.client.post("/login", data={"csrf_token": "stale-token"})
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/dashboard")
        get_auth_client.assert_not_called()

    def test_unauthenticated_login_still_rejects_stale_csrf_token(self):
        response = self.client.post("/login", data={"csrf_token": "stale-token"})
        self.assertEqual(response.status_code, 400)

    def test_match_score_renormalizes_available_components(self):
        result = communityconnect.calculate_match(
            {
                "category": "Education", "user_lang": "it", "user_lang_confirmed": True,
                "availability_slots": [], "explicit_constraints": {},
            },
            {"id": "helper", "active": True, "categories": ["Education"], "spoken_languages": ["it"]},
            cosine=.85, request_place=None, helper_place=None, backend="amd", model="test-model",
        )
        expected = (
            WEIGHTS["semantic"] * result["components"]["semantic"]
            + WEIGHTS["language"] * 1.0
        ) / (WEIGHTS["semantic"] + WEIGHTS["language"])
        self.assertAlmostEqual(result["raw_score"], expected)
        self.assertEqual(result["confidence"], min(99, round(expected * 100)))
        self.assertEqual(result["backend"], "amd")

    def test_get_one_returns_none_when_supabase_has_no_row(self):
        query = Mock()
        query.eq.return_value = query
        query.maybe_single.return_value.execute.return_value = None
        service_client = Mock()
        service_client.table.return_value.select.return_value = query
        with patch("communityconnect.get_service_client", return_value=service_client):
            self.assertIsNone(communityconnect.get_one("profiles", email="missing@example.test"))

    def test_italian_request_keywords_are_categorized(self):
        self.assertEqual(communityconnect.understand_request("Cerco ripetizioni di matematica")[0], "Education")
        self.assertEqual(communityconnect.understand_request("Aiuto per affitto di una stanza")[0], "Housing Support")
        self.assertEqual(communityconnect.understand_request("Orientamento affitto studenti, ricerca stanza, moduli")[0], "Housing Support")

    def test_service_directory_filters_without_exposing_contact_data(self):
        offers = [
            {
                "category": "Housing Support",
                "provider_type": "Association / NGO",
                "profile": {
                    "id": "housing-id", "name": "Housing Desk", "role": "ngo",
                    "skills": "Student rental guidance", "location": "Genova Centro",
                    "availability": "Weekdays", "bio": "Help finding a student room.",
                    "email": "private@example.test", "phone": "123456",
                },
            },
            {
                "category": "Digital Support",
                "provider_type": "Volunteer",
                "profile": {
                    "id": "digital-id", "name": "Digital Help", "role": "volunteer",
                    "skills": "SPID and forms", "location": "Genova Albaro",
                    "availability": "Weekends", "bio": "Help with online forms.",
                    "email": "other@example.test", "phone": "654321",
                },
            },
        ]
        with patch("communityconnect.current_user", return_value=None), \
            patch("communityconnect.get_service_offers", return_value=offers), \
            patch("communityconnect.translate_content", side_effect=lambda text, language: text):
            response = self.client.get("/services?category=Housing%20Support")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Student rental guidance", response.data)
        self.assertNotIn(b"SPID and forms", response.data)
        self.assertNotIn(b"private@example.test", response.data)
        self.assertNotIn(b"123456", response.data)

    def test_service_directory_translates_user_supplied_content(self):
        offer = {
            "category": "Education", "provider_type": "Volunteer",
            "profile": {
                "id": "tutor-id", "name": "Tutor Example", "role": "volunteer",
                "skills": "Ripetizioni di matematica", "location": "Genova Centro",
                "availability": "Pomeriggi", "bio": "Offro aiuto con i compiti.",
            },
        }
        with self.client.session_transaction() as session:
            session["language"] = "en"
        with patch("communityconnect.current_user", return_value=None), \
                patch("communityconnect.get_service_offers", return_value=[offer]), \
                patch("communityconnect.translate_content", side_effect=lambda text, language: f"{language}: {text}"):
            response = self.client.get("/services")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"en: Ripetizioni di matematica", response.data)
        self.assertIn(b"en: Genova Centro", response.data)
        self.assertIn(b"en: Pomeriggi", response.data)
        self.assertIn(b"en: Offro aiuto con i compiti.", response.data)
        self.assertIn(b"Tutor Example", response.data)

    def test_rental_search_translates_demo_cards_when_translation_service_is_throttled(self):
        offer = {
            "category": "Housing Support", "provider_type": "Association / NGO",
            "profile": {
                "id": "housing-id", "name": "Housing Desk", "role": "ngo",
                "skills": "Orientamento pratico per affitti studenti", "location": "Genova",
                "availability": "Lunedi-venerdi",
                "bio": "Aiuto a organizzare documenti e cercare informazioni sugli affitti.",
                "is_demo": True,
            },
        }
        with self.client.session_transaction() as session:
            session["language"] = "en"
        with patch("communityconnect.current_user", return_value=None), \
                patch("communityconnect.get_service_offers", return_value=[offer]), \
                patch("communityconnect._translate_content_cached", side_effect=OSError("translation quota")):
            response = self.client.get("/services?q=rentals")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Practical guidance for student rentals", response.data)
        self.assertIn(b"Monday-Friday", response.data)
        self.assertIn(b"I help organize documents and find information about rentals.", response.data)

    def test_external_translation_uses_selected_target_and_detects_source(self):
        communityconnect._translate_content_cached.cache_clear()
        translated_response = BytesIO(b'{"responseStatus":200,"responseData":{"translatedText":"Good morning"}}')
        with patch("communityconnect.urlopen", return_value=translated_response) as translate_request:
            result = communityconnect.translate_content("Buongiorno", "en")

        self.assertEqual(result, "Good morning")
        api_request = translate_request.call_args.args[0]
        self.assertEqual(api_request.full_url, "https://api.mymemory.translated.net/get")
        self.assertEqual(parse_qs(api_request.data.decode()), {
            "q": ["Buongiorno"], "langpair": ["autodetect|en"], "mt": ["1"],
        })
        communityconnect._translate_content_cached.cache_clear()

    def test_external_translation_retries_after_api_failure(self):
        communityconnect._translate_content_cached.cache_clear()
        translated_response = BytesIO(b'{"responseStatus":200,"responseData":{"translatedText":"Good morning"}}')
        with patch("communityconnect.urlopen", side_effect=[OSError("offline"), translated_response]) as translate_request:
            self.assertEqual(communityconnect.translate_content("Buongiorno", "en"), "Buongiorno")
            self.assertEqual(communityconnect.translate_content("Buongiorno", "en"), "Good morning")

        self.assertEqual(translate_request.call_count, 2)
        communityconnect._translate_content_cached.cache_clear()

    def test_page_translation_stops_when_shared_request_budget_is_exhausted(self):
        with communityconnect.app.test_request_context("/services"):
            communityconnect.g._translation_deadline = communityconnect.monotonic() - 1
            with patch("communityconnect.urlopen") as translate_request:
                result = communityconnect.translate_content("Buongiorno", "en")

        self.assertEqual(result, "Buongiorno")
        translate_request.assert_not_called()

    def test_translation_chunks_respect_service_byte_limit(self):
        chunks = communityconnect.split_translation_text("ripetizioni " * 100)
        self.assertTrue(chunks)
        self.assertTrue(all(len(chunk.encode("utf-8")) <= 450 for chunk in chunks))
        self.assertEqual("".join(chunks), "ripetizioni " * 100)

    def test_language_switch_persists_and_translates_privacy_page(self):
        response = self.client.get("/language/en?next=/privacy")
        self.assertEqual(response.status_code, 302)
        self.client.delete_cookie(communityconnect.app.config["SESSION_COOKIE_NAME"])
        english_page = self.client.get("/privacy")
        self.assertIn(b'<html lang="en">', english_page.data)
        self.assertIn(b"Your data and choices", english_page.data)

        response = self.client.get("/language/it?next=/privacy")
        self.assertEqual(response.status_code, 302)
        italian_page = self.client.get("/privacy")
        self.assertIn(b'<html lang="it">', italian_page.data)
        self.assertIn("I tuoi dati e le tue scelte".encode(), italian_page.data)

    def test_registration_cannot_assign_admin_role(self):
        self.set_csrf()
        response = self.client.post("/register", data={
            "csrf_token": "test-csrf-token", "name": "Intruder", "email": "user@example.com",
            "password": "long-enough-password", "role": "admin", "skills": "Admin",
            "consent": "yes",
        })
        self.assertEqual(response.status_code, 403)

    def test_registration_requires_consent(self):
        self.set_csrf()
        form_data = {
            "csrf_token": "test-csrf-token", "name": "Rita", "email": "rita@example.com",
            "password": "long-enough-password", "role": "seeker", "skills": "Tutoring",
        }
        with patch("communityconnect.current_user", return_value=None), patch("communityconnect.get_auth_client") as auth_client:
            response = self.client.post("/register", data=form_data)
        self.assertEqual(response.status_code, 400)
        auth_client.assert_not_called()

    def test_registration_form_uses_structured_profile_controls(self):
        response = self.client.get("/register")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'name="place_id"', response.data)
        self.assertIn(b'name="spoken_languages"', response.data)
        self.assertIn(b'name="availability_slots"', response.data)
        self.assertIn(b'name="max_travel_km"', response.data)
        self.assertNotIn(b'name="location"', response.data)

    def test_seeker_can_register_without_service_skills(self):
        self.set_csrf()
        auth_response = SimpleNamespace(user=SimpleNamespace(id="new-seeker"), session=None)
        sign_up = Mock(return_value=auth_response)
        insert = Mock()
        insert.execute.return_value = SimpleNamespace(data=[])
        service_client = Mock()
        service_client.table.return_value.insert.return_value = insert
        with patch("communityconnect.current_user", return_value=None), \
                patch("communityconnect.get_auth_client", return_value=SimpleNamespace(auth=SimpleNamespace(sign_up=sign_up))), \
                patch("communityconnect.get_service_client", return_value=service_client):
            response = self.client.post("/register", data={
                "csrf_token": "test-csrf-token", "name": "Student", "email": "student@example.com",
                "password": "long-enough-password", "role": "seeker", "consent": "yes",
                "place_id": "010025", "spoken_languages": ["it"],
            })
        self.assertEqual(response.status_code, 302)
        profile = service_client.table.return_value.insert.call_args.args[0]
        self.assertEqual(profile["skills"], "")
        self.assertEqual(profile["place_id"], "010025")
        self.assertEqual(profile["spoken_languages"], ["it"])

    def test_helper_registration_persists_numeric_offers(self):
        self.set_csrf()
        auth_response = SimpleNamespace(user=SimpleNamespace(id="new-helper"), session=None)
        service_client = Mock()
        insert = service_client.table.return_value.insert.return_value
        insert.execute.return_value = SimpleNamespace(data=[])
        with patch("communityconnect.current_user", return_value=None), \
                patch("communityconnect.get_auth_client", return_value=SimpleNamespace(auth=SimpleNamespace(sign_up=Mock(return_value=auth_response)))), \
                patch("communityconnect.get_service_client", return_value=service_client):
            response = self.client.post("/register", data={
                "csrf_token": "test-csrf-token", "name": "Tutor", "email": "tutor@example.com",
                "password": "long-enough-password", "role": "helper", "consent": "yes",
                "place_id": "010025", "spoken_languages": ["it"], "skill_tags": "Math tutoring",
                "categories": ["Education"], "availability_slots": ["saturday|afternoon"],
                "max_travel_km": "5", "numeric_offers": "hours=2 hours; travel=1-3 km",
            })
        self.assertEqual(response.status_code, 302)
        profile = service_client.table.return_value.insert.call_args.args[0]
        self.assertEqual(profile["numeric_offers"], [
            {"field": "hours", "value": 2.0, "range": None, "unit": "hours"},
            {"field": "travel", "value": None, "range": [1.0, 3.0], "unit": "km"},
        ])

    def test_numeric_items_require_units_unique_fields_and_ordered_ranges(self):
        parsed = communityconnect._parse_numeric_items("hours=2 hours; distance=1-3 km")
        self.assertEqual(parsed[0]["value"], 2.0)
        self.assertEqual(parsed[1]["range"], [1.0, 3.0])
        for invalid in ("hours=2", "hours=2 hours; HOURS=3 HOURS", "distance=3-1 km"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                communityconnect._parse_numeric_items(invalid)

    def test_provider_registration_requires_an_offering(self):
        self.set_csrf()
        form_data = {
            "csrf_token": "test-csrf-token", "name": "Tutor", "email": "tutor@example.com",
            "password": "long-enough-password", "role": "helper", "consent": "yes",
        }
        with patch("communityconnect.current_user", return_value=None), patch("communityconnect.get_auth_client") as auth_client:
            response = self.client.post("/register", data=form_data)
        self.assertEqual(response.status_code, 400)
        auth_client.assert_not_called()

    def test_helper_cannot_create_a_help_request_connection(self):
        self.set_csrf("helper-id")
        helper = {"id": "helper-id", "role": "helper"}
        with patch("communityconnect.current_user", return_value=helper):
            response = self.client.post("/connect/1/other-helper", data={"csrf_token": "test-csrf-token"})
        self.assertEqual(response.status_code, 403)

    def test_helper_can_explicitly_toggle_profile_visibility(self):
        self.set_csrf("helper-id")
        helper = {"id": "helper-id", "role": "helper"}
        service_client = Mock()
        update_query = service_client.table.return_value.update.return_value
        update_query.eq.return_value = update_query
        update_query.execute.return_value = SimpleNamespace(data=[])
        with patch("communityconnect.current_user", return_value=helper), \
                patch("communityconnect.get_service_client", return_value=service_client):
            response = self.client.post("/profile/active", data={
                "csrf_token": "test-csrf-token", "active": "yes",
            })
        self.assertEqual(response.status_code, 302)
        self.assertEqual(service_client.table.return_value.update.call_args.args[0], {"active": True})

    def test_connection_uses_server_computed_score_not_submitted_score(self):
        self.set_csrf("seeker-id")
        seeker = {"id": "seeker-id", "role": "seeker"}
        help_request = {"id": 3, "seeker_id": "seeker-id", "description": "Maths tutoring", "status": "open"}
        helper = {"id": "helper-id", "role": "helper"}
        select_query = Mock()
        select_query.eq.return_value = select_query
        select_query.maybe_single.return_value.execute.return_value = SimpleNamespace(data=None)
        insert_query = Mock()
        insert_query.execute.return_value = SimpleNamespace(data=[])
        service_client = Mock()
        service_client.table.return_value.select.return_value = select_query
        service_client.table.return_value.insert.return_value = insert_query
        candidate = {"helper": helper, "confidence": 87, "components": {}, "weights": {}}
        with patch("communityconnect.current_user", return_value=seeker), \
                patch("communityconnect.get_one", side_effect=[help_request, helper]), \
            patch("communityconnect.find_matches", return_value={"status": "ok", "matches": [candidate], "rejections": []}), \
                patch("communityconnect.get_service_client", return_value=service_client):
            response = self.client.post("/connect/3/helper-id", data={"csrf_token": "test-csrf-token", "score": "0"})
        self.assertEqual(response.status_code, 302)
        inserted = service_client.table.return_value.insert.call_args.args[0]
        self.assertEqual(inserted["score"], 87)
        self.assertFalse(inserted["seeker_accepted"])
        self.assertFalse(inserted["helper_accepted"])

    def test_helper_acceptance_waits_for_seeker_confirmation(self):
        self.set_csrf("helper-id")
        helper = {"id": "helper-id", "role": "helper"}
        connection = {"id": 7, "request_id": 3, "helper_id": "helper-id", "status": "pending", "helper_accepted": False, "seeker_accepted": False}
        update = Mock()
        update.eq.return_value = update
        update.execute.return_value = SimpleNamespace(data=[])
        service_client = Mock()
        service_client.table.return_value.update.return_value = update
        with patch("communityconnect.current_user", return_value=helper), \
                patch("communityconnect.get_one", return_value=connection), \
                patch("communityconnect.get_service_client", return_value=service_client):
            response = self.client.post("/connection/7/accepted", data={"csrf_token": "test-csrf-token"})
        self.assertEqual(response.status_code, 302)
        changes = service_client.table.return_value.update.call_args.args[0]
        self.assertEqual(changes, {"status": "pending", "helper_accepted": True})

    def test_seeker_confirmation_completes_connection(self):
        self.set_csrf("seeker-id")
        seeker = {"id": "seeker-id", "role": "seeker"}
        connection = {"id": 7, "request_id": 3, "helper_id": "helper-id", "status": "pending", "helper_accepted": True, "seeker_accepted": False}
        update = Mock()
        update.eq.return_value = update
        update.execute.return_value = SimpleNamespace(data=[])
        service_client = Mock()
        service_client.table.return_value.update.return_value = update
        with patch("communityconnect.current_user", return_value=seeker), \
                patch("communityconnect.get_one", side_effect=[connection, {"id": 3, "seeker_id": "seeker-id"}]), \
                patch("communityconnect.get_service_client", return_value=service_client):
            response = self.client.post("/connection/7/accepted", data={"csrf_token": "test-csrf-token"})
        self.assertEqual(response.status_code, 302)
        changes = service_client.table.return_value.update.call_args.args[0]
        self.assertEqual(changes, {"status": "accepted", "seeker_accepted": True})

    def test_contact_details_require_both_acceptances(self):
        self.set_csrf("seeker-id")
        seeker = {"id": "seeker-id", "role": "seeker"}
        with patch("communityconnect.current_user", return_value=seeker), \
                patch("communityconnect.get_one", return_value=None) as get_one:
            response = self.client.get("/connection/7")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(get_one.call_args.kwargs, {
            "id": 7, "status": "accepted", "seeker_accepted": True, "helper_accepted": True,
        })

    def test_contact_page_displays_both_participants_contact_details(self):
        self.set_csrf("seeker-id")
        seeker = {"id": "seeker-id", "role": "seeker"}
        connection = {"id": 7, "request_id": 3, "helper_id": "helper-id"}
        help_request = {"id": 3, "seeker_id": "seeker-id"}
        seeker_profile = {"id": "seeker-id", "name": "Rita", "email": "rita@example.test", "phone": "111"}
        helper_profile = {"id": "helper-id", "name": "Tutor", "email": "tutor@example.test", "phone": "222"}
        with patch("communityconnect.current_user", return_value=seeker), \
                patch("communityconnect.get_one", side_effect=[connection, help_request, seeker_profile, helper_profile]):
            response = self.client.get("/connection/7")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"rita@example.test", response.data)
        self.assertIn(b"tutor@example.test", response.data)

    def test_delete_flow_reauthenticates_and_deletes_auth_user(self):
        self.set_csrf("seeker-id")
        profile = {"id": "seeker-id", "email": "rita@example.com", "role": "seeker"}
        admin = SimpleNamespace(delete_user=lambda user_id: self.assertEqual(user_id, "seeker-id"))
        auth = SimpleNamespace(sign_in_with_password=lambda credentials: self.assertEqual(credentials["password"], "correct-password"))
        with patch("communityconnect.current_user", return_value=profile), \
                patch("communityconnect.get_auth_client", return_value=SimpleNamespace(auth=auth)), \
                patch("communityconnect.get_service_client", return_value=SimpleNamespace(auth=SimpleNamespace(admin=admin))):
            response = self.client.post("/delete-account", data={
                "csrf_token": "test-csrf-token", "password": "correct-password", "confirm_delete": "yes",
            })
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/")


if __name__ == "__main__":
    unittest.main()
