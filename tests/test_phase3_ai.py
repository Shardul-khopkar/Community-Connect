import base64
import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.parse import parse_qs

import communityconnect
from core.ai_client import ask_ai, embed_texts
from core.ai_schemas import (
    validate_document_summary,
    validate_intake,
    validate_translation,
    validate_triage,
)
from core.embedding_cache import (
    get_cached_embeddings,
    load_or_create_embeddings,
    load_or_create_embeddings_with_backend,
    store_embeddings,
)
from core.privacy import deidentify_text, restore_redactions


class PhaseThreeCoreTests(unittest.TestCase):
    def test_local_intake_extracts_only_explicit_place_and_daypart(self):
        intake = communityconnect.local_intake(
            "Cerco ripetizioni di matematica il sabato mattina a Genova",
        )
        self.assertEqual(intake["availability_slots"], [{"day": "saturday", "part_of_day": "morning"}])
        self.assertEqual(intake["place"], "Genova")
        self.assertEqual(intake["category"], "Education")
        self.assertEqual(intake["skills"], ["Maths tutoring"])

    def test_local_intake_does_not_invent_rental_place_or_time(self):
        intake = communityconnect.local_intake("Need help with rentals")
        self.assertEqual(intake["category"], "Housing Support")
        self.assertEqual(intake["skills"], ["Rental guidance"])
        self.assertIsNone(intake["place"])
        self.assertEqual(intake["availability_slots"], [])

    def test_local_intake_maps_explicit_time_words_to_daypart(self):
        self.assertEqual(
            communityconnect.local_intake("Cerco aiuto il sabato dalle 16")["availability_slots"],
            [{"day": "saturday", "part_of_day": "afternoon"}],
        )
        self.assertEqual(
            communityconnect.local_intake("I am free Monday and Thursday evening")["availability_slots"],
            [
                {"day": "monday", "part_of_day": "evening"},
                {"day": "thursday", "part_of_day": "evening"},
            ],
        )

    def test_local_intake_extracts_explicit_spoken_language_and_numeric_ranges(self):
        spoken = communityconnect.local_intake(
            "Cerco ripetizioni di matematica, preferisco parlare italiano",
        )
        self.assertEqual(spoken["stated_lang"], "it")
        numeric = communityconnect.local_intake(
            "Need 2-3 hours, 10-20 euros, within 5-10 km, for ages 65-80 years",
        )
        self.assertEqual([item["field"] for item in numeric["number_constraints"]], ["hours", "distance", "age", "price"])
        self.assertEqual(numeric["number_constraints"][0]["range"], [2.0, 3.0])
        self.assertEqual(numeric["number_constraints"][3]["range"], [10.0, 20.0])

    def test_intake_does_not_accept_guessed_urgency_from_ai(self):
        guessed = {
            "category": "Housing Support", "urgency": "urgent", "skills": ["Rental guidance"],
            "availability_slots": [], "number_constraints": [], "language": "en",
            "place": None, "stated_lang": None, "summary": "Need help with rentals",
        }
        with patch("communityconnect.ask_ai", return_value={"text": json.dumps(guessed), "backend": "amd"}):
            intake = communityconnect.intake_request("Need help with rentals")
        self.assertEqual(intake["urgency"], "normal")

    def test_semantic_request_text_adds_bilingual_terms_without_making_category_required(self):
        request = {
            "description": "Need help with rentals",
            "category": "Community Support",
            "skill_tags": ["Rental guidance"],
            "explicit_constraints": {"category": False},
        }
        matching_text = communityconnect._matching_request_text(request)
        self.assertIn("Need help with rentals", matching_text)
        self.assertIn("Rental guidance", matching_text)
        self.assertIn("affitto stanza", matching_text)
        self.assertFalse(request["explicit_constraints"]["category"])

    def test_matching_uses_manual_components_when_embeddings_are_unavailable(self):
        helpers = [
            {
                "id": "rental-helper", "active": True, "categories": ["Housing Support"],
                "skill_tags": ["Student rental guidance", "room search"],
                "bio": "Student rental guidance, room search, and forms",
            },
            {
                "id": "grocery-helper", "active": True, "categories": ["Food Support"],
                "skill_tags": ["Groceries"], "bio": "Grocery shopping and small errands",
            },
        ]
        stored_matches = []

        class Query:
            def __init__(self, table_name):
                self.table_name = table_name

            def select(self, *_args):
                return self

            def in_(self, *_args):
                return self

            def delete(self):
                stored_matches.clear()
                return self

            def upsert(self, rows, **_kwargs):
                stored_matches.extend(rows)
                return self

            def eq(self, *_args):
                return self

            def execute(self):
                rows = helpers if self.table_name == "profiles" else stored_matches
                return SimpleNamespace(data=rows)

        client = SimpleNamespace(table=lambda name: Query(name))
        request = {
            "id": 1, "description": "I need help finding rentals in Sanremo",
            "skill_tags": ["Rental guidance"], "category": "Housing Support",
            "emergency": False, "medical": False,
        }
        with patch("communityconnect.get_service_client", return_value=client), patch(
            "communityconnect.load_or_create_embeddings_with_backend", return_value=None,
        ):
            result = communityconnect.find_matches(request)

        self.assertEqual(result["status"], "ok")
        self.assertEqual([item["helper"]["id"] for item in result["matches"]], ["rental-helper"])
        self.assertEqual(result["matches"][0]["backend"], {"request": "manual", "helper": "manual"})
        self.assertEqual(result["matches"][0]["confidence"], 99)
        self.assertIsNone(result["matches"][0]["components"]["semantic"])
        self.assertEqual(result["matches"][0]["components"]["relevance"], 1.0)
        self.assertFalse(hasattr(communityconnect, "score_match"))

    def test_deidentification_masks_names_contact_and_address_then_restores_locally(self):
        source = "Mi chiamo Mario Rossi, email mario@example.org, telefono +39 333 1234567, via Garibaldi 12"
        safe, redactions = deidentify_text(source)
        for private_value in ("Mario Rossi", "mario@example.org", "+39 333 1234567", "via Garibaldi 12"):
            self.assertNotIn(private_value, safe)
        self.assertEqual(restore_redactions(safe, redactions), source)

    def test_ai_request_sends_redacted_text_and_restores_output_tokens(self):
        reply = BytesIO(b'{"text":"Request from [PERSON_1]","tokens":3,"latency_s":0.2}')
        text = "Mi chiamo Mario Rossi, cerco aiuto"
        with patch.dict(os.environ, {"AI_BASE_URL": "https://ai.example.test", "AI_API_KEY": "private-key"}), \
                patch("core.ai_client.urlopen", return_value=reply) as open_request:
            result = ask_ai(text)
        payload = json.loads(open_request.call_args.args[0].data)
        self.assertNotIn("Mario Rossi", payload["text"])
        self.assertIn("Mario Rossi", result["text"])

    def test_ui_translation_masks_pii_before_external_translation(self):
        communityconnect._translate_content_cached.cache_clear()
        body = BytesIO(b'{"responseStatus":200,"responseData":{"translatedText":"[PERSON_1] is here"}}')
        text = "My name is Mario Rossi"
        with patch("communityconnect.urlopen", return_value=body) as open_request:
            translated = communityconnect.translate_content(text, "en")
        self.assertNotIn("Mario Rossi", parse_qs(open_request.call_args.args[0].data.decode())["q"][0])
        self.assertIn("Mario Rossi", translated)
        communityconnect._translate_content_cached.cache_clear()

    def test_embedding_request_is_redacted(self):
        reply = BytesIO(b'{"vectors":[[0.2,0.8]],"model":"test-model"}')
        with patch.dict(os.environ, {
            "AI_BASE_URL": "https://ai.example.test", "AI_API_KEY": "private-key",
            "AI_EMBEDDING_MODEL_ID": "test-model",
        }), \
                patch("core.ai_client.urlopen", return_value=reply) as open_request:
            vectors = embed_texts(["My name is Alex Morgan; I need tutoring"])
        payload = json.loads(open_request.call_args.args[0].data)
        self.assertNotIn("Alex Morgan", payload["texts"][0])
        self.assertEqual(vectors, [[0.2, 0.8]])

    def test_task_json_schemas_accept_valid_intake_triage_translation_and_document(self):
        self.assertEqual(validate_intake(json.dumps({
            "category": "Education", "urgency": "normal", "skills": ["math tutoring"],
            "availability_slots": [{"day": "saturday", "part_of_day": "afternoon"}],
            "number_constraints": [{"field": "hours", "value": 2, "range": None, "unit": "hours"}],
            "language": "en", "place": "Genova", "stated_lang": None, "summary": "Tutoring request",
        }))['language'], "en")
        self.assertIsNone(validate_intake(json.dumps({
            "category": "Housing Support", "urgency": "normal", "skills": ["room rental"],
            "availability_slots": [], "number_constraints": [], "language": None,
            "place": None, "stated_lang": None, "summary": "Find a room",
        }))['language'])
        self.assertTrue(validate_triage('{"emergency":true,"medical_advice":false,"legal_advice":false,"reason":"urgent symptoms"}')['emergency'])
        self.assertEqual(validate_translation('{"translation":"Bonjour"}')['translation'], "Bonjour")
        self.assertEqual(validate_document_summary(json.dumps({
            "summary": "Bring the listed form.", "actions": [{"action": "Bring the form", "deadline": "2026-11-01"}],
            "help_request": "I need help understanding this form", "deadline": "2026-11-01",
            "emergency": False, "needs_professional_advice": False,
        }))['actions'][0]['deadline'], "2026-11-01")

    def test_task_json_schemas_reject_extra_keys_and_invalid_fields(self):
        with self.assertRaises(ValueError):
            validate_intake('{"category":"Education","urgency":"normal","skills":["tutoring"],"availability_slots":[{"day":"funday","part_of_day":"night"}],"language":"it","summary":"Help","extra":1}')
        with self.assertRaises(ValueError):
            validate_triage('{"emergency":"no","medical_advice":false,"legal_advice":false,"reason":""}')
        with self.assertRaises(ValueError):
            validate_translation('{"translation":4}')

    def test_invalid_intake_json_falls_back_to_rule_based_fields(self):
        with patch("communityconnect.ask_ai", return_value={"text": "not-json", "backend": "amd"}):
            intake = communityconnect.intake_request("Cerco ripetizioni di matematica il sabato")
        self.assertEqual(intake["category"], "Education")
        self.assertEqual(intake["language"], "it")
        self.assertEqual(intake["backend"], "amd")

    def test_sqlite_embedding_cache_round_trips_without_storing_source_text(self):
        with tempfile.TemporaryDirectory() as directory:
            database = os.path.join(directory, "embeddings.db")
            with patch.dict(os.environ, {"AI_EMBEDDING_CACHE_DB": database}):
                source = "Mi chiamo Mario Rossi, ripetizioni di matematica"
                store_embeddings([source], [[0.25, 0.75]], "test-model")
                self.assertEqual(get_cached_embeddings([source], "test-model"), [[0.25, 0.75]])
            with closing(sqlite3.connect(database)) as connection:
                columns = {row[1] for row in connection.execute("pragma table_info(ai_embedding_cache)")}
            self.assertNotIn("source_text", columns)
            self.assertNotIn("text", columns)

    def test_vector_cache_fetches_missing_embeddings_once(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"AI_EMBEDDING_CACHE_DB": os.path.join(directory, "cache.db")}), \
                patch("core.ai_client.embed_texts", return_value=[[1.0, 0.0]]) as embed:
            text = "ripetizioni di matematica"
            self.assertEqual(load_or_create_embeddings([text], "test-model"), [[1.0, 0.0]])
            self.assertEqual(load_or_create_embeddings([text], "test-model"), [[1.0, 0.0]])
        embed.assert_called_once_with([text])

    def test_embedding_cache_does_not_fall_back_to_local_cpu(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"AI_EMBEDDING_CACHE_DB": os.path.join(directory, "cache.db")}):
            store_embeddings(["ripetizioni matematica"], [[0.6, 0.8]], "test-model", "v-test", "local-cpu")
            with patch("core.ai_client.embed_texts", return_value=None) as amd_embed, \
                    patch("core.local_embeddings.embed_texts_local") as local_embed:
                result = load_or_create_embeddings_with_backend(["ripetizioni matematica"], "test-model", "v-test")
        self.assertIsNone(result)
        amd_embed.assert_called_once_with(["ripetizioni matematica"])
        local_embed.assert_not_called()

    def test_emergency_triage_blocks_before_any_ai_call(self):
        with patch("communityconnect.ask_ai") as ask:
            result = communityconnect.triage_request("I cannot breathe")
        self.assertTrue(result["emergency"])
        ask.assert_not_called()

    def test_medical_and_legal_advice_requests_are_flagged_locally(self):
        medical = communityconnect.triage_request("What medicine should I take?")
        legal = communityconnect.triage_request("Please give legal advice about this contract")
        self.assertTrue(medical["medical_advice"])
        self.assertTrue(legal["legal_advice"])

    def test_match_results_store_components_used_for_displayed_score(self):
        helper = {
            "id": "helper-id", "name": "Tutor", "role": "helper", "active": True,
            "categories": ["Education"], "skill_tags": ["Maths tutoring"], "bio": "Tutoring",
            "place_id": "010025", "max_travel_km": 10,
            "availability_slots": [{"day": "saturday", "part_of_day": "afternoon"}],
            "spoken_languages": ["it"], "ratings": [], "response_rate": None,
            "average_response_seconds": None,
        }
        stored_rows = []

        class Query:
            def __init__(self, table_name):
                self.table_name = table_name

            def select(self, *_args):
                return self

            def in_(self, *_args):
                return self

            def delete(self):
                if self.table_name == "match_results":
                    stored_rows.clear()
                return self

            def eq(self, *_args):
                return self

            def upsert(self, rows, **_kwargs):
                stored_rows[:] = rows
                return self

            def execute(self):
                return SimpleNamespace(data=[helper] if self.table_name == "profiles" else stored_rows)

        service_client = SimpleNamespace(table=lambda name: Query(name))
        request = {
            "id": 11, "description": "Cerco ripetizioni di matematica nel weekend",
            "skill": "matematica", "skill_tags": ["Maths tutoring"],
            "category": "Education", "place_id": "010025",
            "availability_slots": [{"day": "saturday", "part_of_day": "afternoon"}],
            "user_lang": "it", "stated_lang": None, "translation_ok": False,
            "explicit_constraints": {}, "number_constraints": [],
        }
        with patch("communityconnect.get_service_client", return_value=service_client), \
                patch("communityconnect.load_or_create_embeddings_with_backend", return_value={
                    "vectors": [[1.0, 0.0], [1.0, 0.0]], "backends": ["amd", "amd"],
                    "model": "test-model", "model_version": "v2",
                }):
            match_result = communityconnect.find_matches(request)
        matches = match_result["matches"]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["confidence"], 99)
        self.assertEqual(stored_rows[0]["score"], matches[0]["confidence"])
        self.assertEqual(matches[0]["components"]["semantic"], stored_rows[0]["score_components"]["components"]["semantic"])
        self.assertEqual(stored_rows[0]["rank"], 1)
        self.assertEqual(stored_rows[0]["score_components"]["backend"], {"request": "amd", "helper": "amd"})


class PhaseThreeRouteTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {
            "SUPABASE_URL": "https://example.supabase.co",
            "SUPABASE_PUBLISHABLE_KEY": "test-publishable-key",
            "SUPABASE_SECRET_KEY": "test-secret-key",
            "AI_BASE_URL": "",
            "AI_API_KEY": "",
        })
        self.environment.start()
        self.original_secret = communityconnect.app.config.get("SECRET_KEY")
        communityconnect.app.config["SECRET_KEY"] = "test-session-secret-value-at-least-32-chars"
        communityconnect.app.config["TESTING"] = True
        self.client = communityconnect.app.test_client()
        self.seeker = {
            "id": "seeker-id", "name": "Rita Example", "email": "rita@example.test", "phone": "",
            "role": "seeker", "location": "", "skills": "", "languages": "it,en", "availability": "",
            "experience": "", "bio": "",
        }

    def tearDown(self):
        communityconnect.app.config["SECRET_KEY"] = self.original_secret
        communityconnect.app.config["TESTING"] = False
        self.environment.stop()

    def set_csrf(self, user_id="seeker-id", role="seeker"):
        with self.client.session_transaction() as session:
            session["_csrf_token"] = "phase3-csrf-token"
            session["user_id"] = user_id
            session["language"] = "en"

    def test_dashboard_renders_intake_and_photo_consent_controls(self):
        self.set_csrf()
        from unittest.mock import Mock

        query = Mock()
        query.eq.return_value = query
        query.order.return_value = query
        query.execute.return_value = SimpleNamespace(data=[])
        service_client = Mock()
        service_client.table.return_value.select.return_value = query
        with patch("communityconnect.current_user", return_value=self.seeker), \
                patch("communityconnect.get_service_client", return_value=service_client), \
                patch("communityconnect.get_connections_for_user", return_value=[]):
            response = self.client.get("/dashboard")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"/request-intake-preview", response.data)
        self.assertIn(b"request-description", response.data)
        self.assertIn(b"Add more details", response.data)
        self.assertIn(b"Not sure / anything", response.data)
        self.assertIn(b"document_consent", response.data)
        self.assertIn(b"contact details are not shared", response.data.lower())

    def test_emergency_and_advice_requests_skip_database_matching(self):
        self.set_csrf()
        safety_results = [
            {"emergency": True, "medical_advice": False, "legal_advice": False, "reason": "emergency"},
            {"emergency": False, "medical_advice": True, "legal_advice": False, "reason": "medical advice"},
        ]
        with patch("communityconnect.current_user", return_value=self.seeker), \
                patch("communityconnect.triage_request", side_effect=safety_results), \
                patch("communityconnect.render_seeker_dashboard", return_value="triage notice"), \
                patch("communityconnect.get_service_client") as database:
            emergency = self.client.post("/request-help", data={
                "csrf_token": "phase3-csrf-token", "description": "I cannot breathe", "location": "Genoa",
            })
            advice = self.client.post("/request-help", data={
                "csrf_token": "phase3-csrf-token", "description": "What medicine should I take?", "location": "Genoa",
            })
        self.assertEqual(emergency.status_code, 200)
        self.assertEqual(advice.status_code, 200)
        database.assert_not_called()

    def test_intake_preview_returns_explicit_place_and_time_as_editable_data(self):
        self.set_csrf()
        parsed = {
            "category": "Education", "skills": ["Maths tutoring"],
            "availability_slots": [{"day": "saturday", "part_of_day": "morning"}],
            "number_constraints": [], "language": "it", "language_needs_confirmation": False,
            "place_id": "010025", "stated_lang": None, "urgency": "normal",
        }
        with patch("communityconnect.current_user", return_value=self.seeker), \
                patch("communityconnect.intake_request", return_value=parsed):
            response = self.client.post("/request-intake-preview", data={
                "csrf_token": "phase3-csrf-token",
                "description": "Cerco ripetizioni di matematica il sabato mattina a Genova",
            })

        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(result["availability_slots"], [{"day": "saturday", "part_of_day": "morning"}])
        self.assertEqual(result["place_name"], "Genova")
        self.assertEqual(result["category"], "Education")
        self.assertIsNone(result["stated_lang"])

    def test_description_only_request_saves_and_returns_ranked_matches(self):
        self.set_csrf()
        parsed = {
            "category": "Housing Support", "urgency": "normal", "skills": ["Rental guidance"],
            "availability_slots": [], "number_constraints": [], "language": "en",
            "language_needs_confirmation": False, "ai_language": "en", "place": None,
            "place_id": None, "stated_lang": None, "number_constraints_text": "",
        }
        saved_request = {
            "id": 83, "seeker_id": "seeker-id", "description": "Need help with rentals",
            "category": "Housing Support", "skill": "Rental guidance", "location": "",
            "place_id": None, "availability": "Flexible", "language": "en", "user_lang": "en",
            "user_lang_confirmed": True, "availability_slots": [], "stated_lang": None,
            "explicit_constraints": {"category": False, "place_id": False},
        }
        duplicate_query = Mock()
        duplicate_query.eq.return_value = duplicate_query
        duplicate_query.limit.return_value = duplicate_query
        duplicate_query.execute.return_value = SimpleNamespace(data=[])
        insert_query = Mock()
        insert_query.execute.return_value = SimpleNamespace(data=[saved_request])
        service_client = Mock()
        service_client.table.return_value.select.return_value = duplicate_query
        service_client.table.return_value.insert.return_value = insert_query
        match = {
            "helper": {"id": "housing-helper", "name": "Housing Desk", "skills": "Rental guidance",
                       "location": "Genova", "availability": "Flexible", "role": "ngo"},
            "confidence": 71, "label": "Good", "exact_all": False, "exact_hits": 0,
            "components": {"semantic": 0.7, "location": None, "availability": None,
                           "language": 0.6, "reliability": None},
            "weights": {"semantic": 0.524, "location": None, "availability": None,
                        "language": 0.0, "reliability": None},
            "distance_km": None, "backend": {"request": "local-cpu", "helper": "local-cpu"},
            "model": "test-model", "ranked_above_next_by": None,
        }
        with patch("communityconnect.current_user", return_value=self.seeker), \
                patch("communityconnect.triage_request", return_value={"emergency": False, "medical_advice": False, "legal_advice": False}), \
                patch("communityconnect.intake_request", return_value=parsed), \
                patch("communityconnect.get_service_client", return_value=service_client), \
                patch("communityconnect._precompute_matching_embedding", return_value=None), \
                patch("communityconnect.find_matches", return_value={"status": "ok", "matches": [match], "rejections": []}):
            response = self.client.post("/request-help", data={
                "csrf_token": "phase3-csrf-token", "description": "Need help with rentals",
            })

        self.assertEqual(response.status_code, 200)
        saved = service_client.table.return_value.insert.call_args.args[0]
        self.assertIsNone(saved["place_id"])
        self.assertEqual(saved["availability_slots"], [])
        self.assertEqual(saved["category"], "Housing Support")
        self.assertFalse(saved["explicit_constraints"]["category"])
        self.assertIn(b"Housing Desk", response.data)

    def test_ambiguous_written_language_reasks_without_losing_preview_state(self):
        self.set_csrf()
        description = "affitto stanza"
        preview_hash = communityconnect.hashlib.sha256(description.encode("utf-8")).hexdigest()
        with patch("communityconnect.current_user", return_value=self.seeker), \
                patch("communityconnect.triage_request", return_value={"emergency": False, "medical_advice": False, "legal_advice": False}), \
                patch("communityconnect.render_seeker_dashboard", return_value="confirm language") as render_dashboard:
            response = self.client.post("/request-help", data={
                "csrf_token": "phase3-csrf-token", "description": description,
                "intake_preview_hash": preview_hash, "place_id": "010025",
                "place_query": "Genova · Genova", "place_source": "manual",
                "category": "Housing Support", "category_source": "ai",
            })

        self.assertEqual(response.status_code, 200)
        form_values = render_dashboard.call_args.kwargs["request_form_values"]
        self.assertEqual(form_values["intake_preview_hash"], preview_hash)
        self.assertEqual(form_values["place_id"], "010025")
        self.assertEqual(form_values["place_source"], "manual")
        self.assertEqual(form_values["user_lang"], "")
        self.assertTrue(form_values["language_needs_confirmation"])

    def test_identical_open_request_redirects_to_existing_request(self):
        self.set_csrf()
        parsed = {
            "category": "Housing Support", "urgency": "normal", "skills": ["Rental guidance"],
            "availability_slots": [], "number_constraints": [], "language": "en",
            "language_needs_confirmation": False, "ai_language": "en", "place": None,
            "place_id": None, "stated_lang": None, "number_constraints_text": "",
        }
        duplicate_query = Mock()
        duplicate_query.eq.return_value = duplicate_query
        duplicate_query.limit.return_value = duplicate_query
        duplicate_query.execute.return_value = SimpleNamespace(data=[{"id": 44}])
        service_client = Mock()
        service_client.table.return_value.select.return_value = duplicate_query
        with patch("communityconnect.current_user", return_value=self.seeker), \
                patch("communityconnect.triage_request", return_value={"emergency": False, "medical_advice": False, "legal_advice": False}), \
                patch("communityconnect.intake_request", return_value=parsed), \
                patch("communityconnect.get_service_client", return_value=service_client):
            response = self.client.post("/request-help", data={
                "csrf_token": "phase3-csrf-token", "description": "Need help with rentals",
            })

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/dashboard#request-44")
        service_client.table.return_value.insert.assert_not_called()

    def test_smart_intake_prefills_editable_request_fields(self):
        self.set_csrf()
        intake = {
            "category": "Education", "urgency": "soon", "skills": ["math tutoring"],
            "availability": "Saturday", "language": "en", "summary": "Help with math tutoring",
        }
        with patch("communityconnect.current_user", return_value=self.seeker), \
                patch("communityconnect.triage_request", return_value={"emergency": False, "medical_advice": False, "legal_advice": False, "reason": ""}), \
                patch("communityconnect.intake_request", return_value=intake), \
                patch("communityconnect.render_seeker_dashboard", return_value="prefilled") as render_dashboard:
            response = self.client.post("/intake", data={
                "csrf_token": "phase3-csrf-token", "description": "Maths help Saturday", "place_id": "010025",
            })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_data(as_text=True), "prefilled")
        self.assertEqual(render_dashboard.call_args.kwargs["intake"]["category"], "Education")
        self.assertEqual(render_dashboard.call_args.kwargs["draft_description"], "Help with math tutoring")

    def test_request_help_persists_edited_intake_language_and_urgency(self):
        self.set_csrf()
        with self.client.session_transaction() as session:
            session["language"] = "en"
        inserted_request = {
            "id": 12, "seeker_id": "seeker-id", "description": "Math tutoring on Saturday",
            "category": "Education", "skill": "Math tutoring", "location": "Genova",
            "availability": "Saturday", "language": "en", "urgency": "soon", "status": "open",
        }
        from unittest.mock import Mock

        insert_query = Mock()
        insert_query.execute.return_value = SimpleNamespace(data=[inserted_request])
        duplicate_query = Mock()
        duplicate_query.eq.return_value = duplicate_query
        duplicate_query.limit.return_value = duplicate_query
        duplicate_query.execute.return_value = SimpleNamespace(data=[])
        service_client = Mock()
        service_client.table.return_value.select.return_value = duplicate_query
        service_client.table.return_value.insert.return_value = insert_query
        match_helper = {
            "id": "housing-helper", "name": "Sportello Casa Studenti", "role": "ngo",
            "skills": "Orientamento affitto studenti, ricerca stanza, moduli", "location": "Genova",
            "availability": "Lunedi-venerdi", "is_demo": True,
        }
        match = {
            "helper": match_helper, "confidence": 75, "label": "Good", "exact_all": False,
            "exact_hits": 0, "components": {
                "semantic": 0.7, "location": None, "availability": None,
                "language": None, "reliability": None,
            },
            "weights": {
                "semantic": 0.524, "location": None, "availability": None,
                "language": None, "reliability": None,
            },
            "distance_km": None, "backend": {"request": "local-cpu", "helper": "local-cpu"},
            "model": "test-model", "ranked_above_next_by": None,
        }
        with patch("communityconnect.current_user", return_value=self.seeker), \
                patch("communityconnect.triage_request", return_value={"emergency": False, "medical_advice": False, "legal_advice": False, "reason": ""}), \
                patch("communityconnect.intake_request", return_value={
                    "category": "Housing Support", "urgency": "normal", "skills": ["AI guessed rental skill"],
                    "availability_slots": [], "number_constraints": [], "language": "en",
                    "language_needs_confirmation": False, "ai_language": "en", "place": "Arenzano",
                    "place_id": "010004", "stated_lang": None, "number_constraints_text": "",
                }), \
                patch("communityconnect.get_service_client", return_value=service_client), \
                patch("communityconnect._precompute_matching_embedding", return_value=None), \
                patch("communityconnect.find_matches", return_value={"status": "ok", "matches": [match], "rejections": []}), \
                patch("communityconnect._translate_content_cached", side_effect=OSError("translation quota")):
            response = self.client.post("/request-help", data={
                "csrf_token": "phase3-csrf-token", "description": "Math tutoring on Saturday",
                "place_id": "010025", "category": "Education", "skill_tags": "Math tutoring",
                "availability_slots": ["saturday|afternoon"], "user_lang": "en", "urgency": "soon",
                "number_constraints": "hours=2 hours", "numbers_exact": "yes",
            })
        self.assertEqual(response.status_code, 200)
        saved = service_client.table.return_value.insert.call_args.args[0]
        self.assertEqual(saved["language"], "en")
        self.assertEqual(saved["user_lang"], "en")
        self.assertEqual(saved["place_id"], "010025")
        self.assertEqual(saved["skill_tags"], ["Math tutoring"])
        self.assertEqual(saved["availability_slots"], [{"day": "saturday", "part_of_day": "afternoon"}])
        self.assertEqual(saved["number_constraints"], [{"field": "hours", "value": 2.0, "range": None, "unit": "hours"}])
        self.assertTrue(saved["explicit_constraints"]["numbers"])
        self.assertEqual(saved["urgency"], "soon")
        self.assertEqual(saved["skill"], "Math tutoring")
        self.assertIn(b"Student rental guidance, room search, and forms", response.data)
        self.assertIn(b"Monday-Friday", response.data)

    def test_document_upload_requires_explicit_consent(self):
        self.set_csrf()
        with patch("communityconnect.current_user", return_value=self.seeker), \
                patch("communityconnect.render_seeker_dashboard", return_value="dashboard"), \
                patch("communityconnect.summarize_document") as summarize:
            response = self.client.post("/document-intake", data={"csrf_token": "phase3-csrf-token"})
        self.assertEqual(response.status_code, 200)
        summarize.assert_not_called()

    def test_consented_document_is_processed_in_memory_and_offers_prefill(self):
        self.set_csrf()
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"fake-demo-image"
        document = {
            "summary": "Bring the appointment form.",
            "actions": [{"action": "Bring the form", "deadline": "2026-11-01"}],
            "help_request": "I need help understanding an appointment form",
            "deadline": "2026-11-01", "emergency": False,
            "needs_professional_advice": False, "backend": "amd",
        }
        from unittest.mock import Mock

        query = Mock()
        query.eq.return_value = query
        query.order.return_value = query
        query.execute.return_value = SimpleNamespace(data=[])
        service_client = Mock()
        service_client.table.return_value.select.return_value = query
        with patch("communityconnect.current_user", return_value=self.seeker), \
                patch("communityconnect.summarize_document", return_value=document) as summarize, \
                patch("communityconnect.triage_request", return_value={"emergency": False, "medical_advice": False, "legal_advice": False, "reason": ""}), \
                patch("communityconnect.get_service_client", return_value=service_client), \
                patch("communityconnect.get_connections_for_user", return_value=[]):
            response = self.client.post("/document-intake", data={
                "csrf_token": "phase3-csrf-token", "document_consent": "yes",
                "document_image": (BytesIO(png_bytes), "fake.png", "image/png"),
            }, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(base64.b64decode(summarize.call_args.args[0]), png_bytes)
        self.assertIn(b"AI summary, please check the original document", response.data)
        self.assertIn(b"Bring the appointment form.", response.data)
        self.assertIn(b"I need help understanding an appointment form", response.data)
        service_client.table.return_value.insert.assert_not_called()

    def test_connection_participant_translation_shows_original_and_translation(self):
        self.set_csrf()
        connection = {"id": 7, "request_id": 3, "helper_id": "helper-id"}
        help_request = {"id": 3, "seeker_id": "seeker-id"}
        seeker_profile = {**self.seeker}
        helper_profile = {"id": "helper-id", "name": "Tutor", "email": "tutor@example.test", "phone": ""}
        with patch("communityconnect.current_user", return_value=self.seeker), \
                patch("communityconnect.get_one", side_effect=[connection, help_request, seeker_profile, helper_profile]), \
                patch("communityconnect.translate_message", return_value="Necesito ayuda con el formulario") as translate:
            response = self.client.post("/connection/7/translate", data={
                "csrf_token": "phase3-csrf-token", "message": "I need help with the form", "target_language": "es",
            })
        self.assertEqual(response.status_code, 200)
        translate.assert_called_once()
        self.assertIn(b"I need help with the form", response.data)
        self.assertIn("Necesito ayuda con el formulario".encode(), response.data)

    def test_nonparticipant_cannot_translate_connection_message(self):
        self.set_csrf(user_id="stranger-id", role="helper")
        stranger = {"id": "stranger-id", "role": "helper"}
        with patch("communityconnect.current_user", return_value=stranger), \
                patch("communityconnect.get_one", side_effect=[
                    {"id": 7, "request_id": 3, "helper_id": "helper-id"},
                    {"id": 3, "seeker_id": "seeker-id"},
                ]), \
                patch("communityconnect.translate_message") as translate:
            response = self.client.post("/connection/7/translate", data={
                "csrf_token": "phase3-csrf-token", "message": "hello", "target_language": "it",
            })
        self.assertEqual(response.status_code, 404)
        translate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
