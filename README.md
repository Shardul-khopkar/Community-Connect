# CommunityConnect

CommunityConnect connects people seeking practical help with students, volunteers, and local organizations in Genova and Liguria.

The app includes local service discovery, multilingual request intake, transparent matching, consent-based registration, and private connections. Matching uses multilingual embeddings and structured eligibility/scoring rules. Optional AMD AI services can assist with intake and document summaries; matching can use the local CPU embedding fallback.

## Requirements

- Python 3.12 or newer
- A Supabase project
- Git

## Setup

Clone this project:

```powershell
git clone https://github.com/Shardul-khopkar/Community-Connect.git
cd Community-Connect
```

Create a virtual environment and install the app dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy `.env.example` to `.env`, then set `COMMUNITYCONNECT_SECRET_KEY`, `SUPABASE_URL`, `SUPABASE_PUBLISHABLE_KEY`, and `SUPABASE_SECRET_KEY`. Keep `.env` private; it is ignored by Git. Set `COMMUNITYCONNECT_DEMO_PASSWORD` if you plan to seed demo records.

Run `supabase/schema.sql` in the Supabase SQL Editor. Then initialize and seed optional demo data:

```powershell
python -m flask --app app init-db
python -m flask --app app seed-demo
```

Start the app:

```powershell
python app.py
```

Open <http://127.0.0.1:5000>. Italian is the default language; use the header switch for English.

## Optional Components

Install the local CPU embedding model when AMD embeddings are unavailable:

```powershell
pip install -r local_ai_requirements.txt
```

The AMD inference service is separate from the Flask app. See [docs/amd_deploy.md](docs/amd_deploy.md) for its deployment notes. Without an available embedding backend, requests remain usable but cannot be ranked.

## Tests

Run the test suite with:

```powershell
python -m unittest discover -s tests
```

Matching evaluation tools and synthetic datasets are under `scripts/`, `data/`, and `tests/`.

## Project Layout

```text
app.py                    Flask entry point
communityconnect.py       Routes and application workflows
core/                     AI clients, privacy, places, and scoring
data/                     Liguria place catalog and evaluation data
supabase/                 Database schema and migrations
templates/                Jinja templates
static/                   CSS and browser-side scripts
tests/                    Unit and security tests
```

## Security

Never commit `.env`, API keys, or real user records. The checked-in `.env.example` is only a template. Public account registration does not create administrators. Contact details are revealed only after the helper accepts and the seeker confirms.
