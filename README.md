# CommunityConnect: Community Support in Genova and Liguria

**A local platform for turning a practical need into a clear, consent-based connection with someone nearby who can help.**

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20app-000000?logo=flask)](https://flask.palletsprojects.com/)
[![Supabase](https://img.shields.io/badge/Supabase-Auth%20%2B%20Postgres-3ECF8E?logo=supabase&logoColor=white)](https://supabase.com/)

## The Idea

CommunityConnect is a community-support prototype for Genova and Liguria. It helps people find practical services offered by students, volunteers, and local organizations, and helps seekers describe what they need without having to know the right category or terminology first.

The project focuses on three things: making local help easier to discover, explaining why a provider may fit a request, and keeping personal contact details private until both participants agree to connect. Profiles and demo data are self-reported or synthetic; the platform does not verify identities, qualifications, or service quality.

## The Problem and the Approach

People may need tutoring, digital-form support, help understanding an administrative letter, groceries, transport, or guidance finding student housing. The people and organizations willing to help can be difficult to discover, especially when information is scattered across neighborhood groups and service directories.

CommunityConnect brings service discovery, structured requests, provider profiles, and a transparent matching workflow into one place. A seeker can write naturally, review suggested details, correct them, and see eligible providers ranked with a breakdown of the available matching signals. The app is a directory and connection aid, not an emergency service or a substitute for professional medical, legal, or housing advice.

## What Visitors See

### Home page

The home page introduces the service, offers direct paths for people seeking or offering help, and shows featured local services. Search and category links lead to the service directory. Listings come from active helper, volunteer, and NGO profiles in Supabase; demo profiles are visibly labeled as demo data.

### Service directory

The directory can be searched by words such as tutoring, housing, or digital support and filtered by service category. Each listing shows the provider type, offered skills, location, availability, and a short self-reported description. Public listings do not expose provider email addresses or phone numbers. A visitor can start a seeker registration from a listing.

### Language and accessibility

Italian is the default interface language, with an English switch in the header. The interface remembers the selected language for the session. The app can translate displayed user content and can translate a drafted connection message; translation output is shown for review and is not sent to the other participant by the app.

## A Seeker's Journey

1. **Create an account.** Registration uses Supabase Auth and creates a matching profile. The seeker chooses a municipality from the Liguria place catalog, supplies a password of at least 12 characters, and gives explicit consent to profile-data storage.
2. **Describe the practical need.** The seeker enters a short description, selects or confirms a service category, and can add a preferred municipality, skills, availability, urgency, and any explicit numeric or spoken-language requirements. Suggestions can be edited or removed before submitting.
3. **Review request understanding.** The intake flow may suggest a category, skills, language, and other request details. It distinguishes the language the request is written in from a language the seeker explicitly wants a helper to speak. Ambiguous language detection can ask the user to confirm rather than silently deciding.
4. **Pass safety checks.** Local rules check for emergency, medical-advice, and legal-advice requests. A request that appears to need emergency or professional help is not sent through ordinary matching. Possible emergencies display guidance to call 112; the platform does not provide emergency response or professional advice.
5. **Compare matches.** Eligible local providers are ranked using multilingual semantic similarity and any available structured signals. The results show a match score and the components that contributed to it, as well as exact stated constraints and the embedding backend/model where available.
6. **Request a connection.** The seeker can send a connection request to an eligible provider. It appears in the provider's workspace as a pending request.
7. **Confirm contact sharing.** The provider must accept first. The seeker must then separately confirm. Only after both steps does the connection become accepted and the two participants' contact details become visible to each other.

## A Helper or Organization's Journey

A helper, volunteer, or NGO registers an offer profile with skill tags, service categories, spoken languages, availability slots, a Liguria municipality, and a maximum travel distance. The profile is self-reported. Providers choose whether their profile is active and discoverable; only active profiles are considered for matching and public service listings.

Incoming connection requests appear in the provider workspace. The provider can accept or reject. Accepting does not immediately expose contact information: the seeker must confirm the connection as well. Once confirmed, both participants can see the contact details and use the page's translation helper to prepare a message in another supported language. CommunityConnect does not provide an in-app messaging inbox or transmit that drafted message.

## AI-Assisted Intake and Document Support

AI is optional and is not the matching authority. If `AI_BASE_URL` and `AI_API_KEY` configure the separate inference service, the Flask app can send a de-identified text prompt to it. The client retries once when the service fails. Ordinary request intake can fall back to local rules that suggest basic categories, skills, and timing. The app validates structured AI responses before using them.

The seeker can optionally upload an administrative document image to request a plain-language summary, action items, or dates. The upload requires a separate consent checkbox and accepts PNG, JPEG, or WebP images up to 10 MB. The image is sent to the configured AI service for that request and is not stored by the app. Use synthetic documents for demos; do not upload real personal documents during a hackathon presentation. If the AI service is unavailable, the local fallback explicitly says that it did not analyze the image.

Safety checks still apply to the proposed help request. If the document appears to require emergency, medical, or legal assistance, the app avoids turning the summary into a matching request and directs the user to appropriate real-world help.

## How Matching Works

Matching is designed as an eligibility filter followed by a transparent score:

1. **Eligibility first.** A provider must have an active profile. A category the seeker explicitly marks as required must be offered. A stated spoken-language requirement must be met unless the seeker explicitly allows translation. Emergency and professional medical-advice requests are not routed as ordinary matches.
2. **Semantic similarity.** The app embeds the request and provider skill/profile text with the multilingual `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` model. Cosine similarity is calibrated to a bounded semantic component.
3. **Location.** Municipality coordinates from the Liguria place catalog estimate distance. A provider's declared travel radius determines the location component; the app does not use precise live tracking.
4. **Availability.** Explicit day-and-part-of-day slots are compared. Missing availability information is not treated as a mismatch; it is omitted from the score.
5. **Language.** Confirmed written-language preference and any explicitly stated spoken-language need contribute where applicable.
6. **Reliability.** Existing response-rate and rating data can contribute when available. The interface does not claim that a profile or skill has been independently verified.
7. **Normalize and rank.** The calibrated component weights are renormalized over the components that have data. This avoids treating missing information as zero. Exact constraints are shown separately. A score of 100 is reserved for a result meeting every explicitly marked hard constraint; other scores are capped at 99.

A match score is a ranking aid, **not a probability, guarantee, or quality certification**. If neither the AMD embedding service nor the optional local CPU embedding package is available, the app reports that it cannot rank the request rather than inventing keyword-based match scores.

## Site Areas and Their Purpose

| Area | What it does |
| --- | --- |
| `/` | Introduces CommunityConnect, shows featured services, and routes visitors into the main flows. |
| `/services` | Searches and filters active local service listings without exposing contact details. |
| `/register` and `/login` | Creates Supabase Auth accounts and signs users in. Public registration supports seeker, helper, volunteer, and NGO roles; administrator accounts are created privately by an administrator. |
| `/dashboard` | Shows a seeker's requests and outgoing connections, or a provider's profile visibility and incoming requests. |
| Request intake and matching | Checks safety, presents editable structured details, saves the request, applies eligibility rules, computes scores, and displays match explanations. |
| `/connection/<id>` | Shows contact details only for an accepted connection confirmed by both participants. It also offers a message-translation form. |
| `/privacy` | Explains how account/profile data and optional AI processing are handled. |
| `/admin` | Shows platform totals, request-category counts, and profiles marked for review. It is a protected administrative view, not a public registration path. |

## Privacy, Safety, and Trust

- Supabase Auth handles email/password authentication. The service key is used only by the server; the browser receives no database service key.
- Supabase Row Level Security is enabled. The app performs privileged data access on the server.
- Registration requires consent, and profiles are labeled as self-reported.
- Provider contact details are withheld until helper acceptance and separate seeker confirmation.
- AI text prompts pass through de-identification for common names and contact details before being sent. Uploaded images are a separate case: they are transmitted only after specific document consent, so use synthetic images in demos.
- Uploaded document images are processed for the request and are not stored by this app.
- Account deletion requires the user's current password and explicit confirmation.
- Emergency and professional medical/legal requests are screened out of ordinary matching. CommunityConnect is not an emergency, medical, legal, or identity-verification service.
- `.env` is ignored by Git. Never commit secrets, API keys, real personal information, or real uploaded documents.

## Technology and Project Structure

- **Flask and Jinja** serve the site and its request workflows.
- **Supabase Auth and PostgreSQL** store accounts, profiles, requests, connections, and match results.
- **Core modules** separate AI clients and response validation, privacy/de-identification, language detection, the Liguria place catalog, embedding cache, and match scoring.
- **Optional AMD inference service** handles configured text/image generation and remote embeddings. Its dependencies are separate from the Flask app.
- **Optional local CPU embeddings** provide multilingual matching when installed. The application does not silently switch to a weaker keyword scorer.

```text
app.py                    Flask entry point
communityconnect.py       Routes and application workflows
core/ai_client.py         AMD client, retry, local intake fallback, metrics
core/ai_schemas.py        Validation for structured AI responses
core/privacy.py           Text de-identification and restoration
core/scoring.py           Eligibility, score components, and ranking
core/place_catalog.py     Liguria municipality catalog
core/embedding_cache.py   Persistent embedding cache and backend selection
server.py                 Separate AI inference service
supabase/schema.sql       Database schema, indexes, and row-level security
templates/                Home, service directory, forms, and workspace UI
static/                   Stylesheets and browser-side request interactions
data/                     Liguria places and synthetic evaluation/calibration data
tests/                    Unit, matching, privacy, and security tests
```

## Setup and Run

### Requirements

- Python 3.12 or newer
- A Supabase project
- Git

### Install

```powershell
git clone https://github.com/Shardul-khopkar/Community-Connect.git
cd Community-Connect
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set:

```dotenv
COMMUNITYCONNECT_SECRET_KEY=<long-random-session-secret>
SUPABASE_URL=https://<your-project>.supabase.co
SUPABASE_PUBLISHABLE_KEY=<your-publishable-key>
SUPABASE_SECRET_KEY=<your-server-side-secret-key>
COMMUNITYCONNECT_HTTPS_ONLY=0
COMMUNITYCONNECT_DEMO_PASSWORD=<private-password-of-at-least-12-characters>
AI_BASE_URL=
AI_API_KEY=
AI_TIMEOUT_SECONDS=20
AI_EMBEDDING_MODEL_ID=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Generate the Flask session secret locally; do not reuse the placeholder. Keep all real keys in `.env`, never in this README or source control.

Run `supabase/schema.sql` in the Supabase SQL Editor. Then verify the database connection:

```powershell
python -m flask --app app init-db
```

Optional: create clearly labeled synthetic demo records. `COMMUNITYCONNECT_DEMO_PASSWORD` must be at least 12 characters. Reset removes only rows marked as demo data.

```powershell
python -m flask --app app seed-demo
python -m flask --app app reset-demo --yes
```

Start the local site:

```powershell
python app.py
```

Open <http://127.0.0.1:5000>. Italian is the default interface language; use the IT/EN control in the header to switch.

On macOS/Linux, create the environment with `python3 -m venv .venv`, activate it with `source .venv/bin/activate`, then use the same `pip` and `python` commands.

### Optional Local Matching Model

When AMD embeddings are unavailable, install the CPU embedding dependency:

```powershell
pip install -r local_ai_requirements.txt
```

Without either embedding backend, request creation remains available but results cannot be ranked. See [docs/amd_deploy.md](docs/amd_deploy.md) for the separate AMD service setup; verify current hardware and model requirements before deployment.

## Hackathon Walkthrough

A short, end-to-end demo can show both the product and the reasoning behind it:

1. **Start on the home page.** Point out the Genova/Liguria focus, Italian/English switch, featured services, and the fact that public listings do not expose contact details.
2. **Browse services.** Search for tutoring or student housing and filter by category. Explain that cards come from active local provider profiles, and synthetic records carry a visible demo badge.
3. **Show request creation.** Sign in as a synthetic seeker, describe a practical need in natural language, and let the preview suggest structured fields. Edit one suggestion to show that the person, not the AI, has the final say.
4. **Explain safety.** Use a fictional, non-sensitive example. Mention that emergency and medical/legal-advice requests are screened out rather than treated as ordinary matching opportunities.
5. **Compare matches.** Show the eligibility result, score components, municipality distance, and any exact constraints. Clarify that the score is a ranking aid, not a success probability.
6. **Demonstrate mutual consent.** Open the provider workspace and accept a request. Return to the seeker account and confirm it. Only then show the contact-details page.
7. **Show the language tool.** Translate a fictional draft and point out that the original and translation are displayed for review; the platform does not send the message.
8. **Optional document demo.** Use only a synthetic letter image, select the separate consent, and show the summary/actions. Call out that the image is sent to the configured AI backend and is not stored by the app.

Seeded demo records are synthetic and tagged. Demo account passwords are configured locally through `COMMUNITYCONNECT_DEMO_PASSWORD`; no demo password is published here. Do not demonstrate with real names, contact details, or documents.

## Evaluation and Limitations

The repository includes bilingual synthetic requests, a grouped train/test matching review set, calibration data, and evaluation scripts. These materials help detect regressions and demonstrate how ranking can be measured; synthetic labels are not evidence of real-world service quality. The reported match score is not a probability, and the app does not independently verify providers or guarantee an outcome.

The app requires a working Supabase project for accounts and live listings. AI generation and translation can be unavailable; local rule-based intake remains available, but matching requires an embedding backend. The platform is a prototype for practical community support, not emergency response or professional advice.

## Tests

Run the automated test suite from the project root:

```powershell
python -m unittest discover -s tests
```

Request and matching evaluation scripts are in `scripts/`; their input data and evaluation sets are in `data/` and `tests/`.
