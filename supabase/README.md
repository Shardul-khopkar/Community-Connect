# Supabase setup

1. In Supabase, open the SQL Editor for the project and run `schema.sql`. For an existing project, also run `phase5_match_scoring_migration.sql` to add structured matching fields and ratings; it is additive and safe to rerun.
2. In project API settings, copy the publishable key and create a new secret key. Do not reuse any secret key that has been pasted into chat or committed.
3. Copy `.env.example` to `.env`, then set `COMMUNITYCONNECT_SECRET_KEY`, `SUPABASE_PUBLISHABLE_KEY`, and `SUPABASE_SECRET_KEY`. Generate the Flask secret with `python -c "import secrets; print(secrets.token_hex(32))"`. Keep `.env` local; it is ignored by Git.
4. Set `SUPABASE_URL` to the project URL shown in Supabase. The URL currently in `.env.example` is the project reference provided for this hackathon.
5. In Supabase Auth settings, configure the local site URL as `http://127.0.0.1:5000`. Email confirmation may be enabled; if enabled, users must confirm their email before password login.
6. Install dependencies and run `.\\.venv\\Scripts\\python.exe -m flask --app app init-db`, then `.\\.venv\\Scripts\\python.exe -m flask --app app seed-demo` to create the demo accounts.

The Flask server uses the secret key for server-side profile/request operations and the publishable key for Auth. Never put the secret key in HTML, JavaScript, logs, or version control. Row-level security is enabled and there are intentionally no public table policies; the app accesses tables only from the server.
