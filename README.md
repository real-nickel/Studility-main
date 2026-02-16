# Studility — Vercel + Postgres deployment notes

This project is a Flask app that uses SQLAlchemy. The repository has been updated to support connecting to a managed PostgreSQL database (as commonly used with Vercel via providers like Supabase or Neon).

What I changed
- `app.py` reads `DATABASE_URL` (or `DATABASE_URI`) from the environment and normalizes `postgres://` -> `postgresql+psycopg2://` for SQLAlchemy.
- `app.py` initializes `Flask-Migrate` so you can create and run Alembic migrations for Postgres.
- `requirements.txt` includes `psycopg2-binary` and `Flask-Migrate`.
- The app only runs `db.create_all()` automatically for local sqlite. For Postgres (managed DBs) you should apply migrations.

Quick local dev with Postgres
1. Install deps:

```bash
python3 -m pip install -r requirements.txt
```

2. Set environment variables (example for local Postgres):

```bash
export DATABASE_URL="postgresql+psycopg2://user:password@localhost:5432/studility"
export SECRET_KEY="your-secret"
```

3. Create migrations and apply them (first-time setup):

```bash
export FLASK_APP=app.py
flask db init
flask db migrate -m "initial"
flask db upgrade
```

4. Run the app:

```bash
python3 app.py
```

Deploying to Vercel (high level)
- On Vercel, set the Environment Variables in your Project settings:
  - `DATABASE_URL` — your full Postgres connection string (Vercel will provide it if you attach a DB add-on or you can use Supabase/Neon credentials).
  - `SECRET_KEY` — a strong random string for sessions.
  - Optionally `DB_SSLMODE=require` if your Postgres provider requires SSL connections.

- Vercel will detect `requirements.txt` and install dependencies. For the schema:
  - Either run migrations in CI before deployment (recommended), or use a one-time deploy step to run `flask db upgrade` against your Postgres database after deployment.

Notes
- Use migrations (Alembic/Flask-Migrate) rather than `create_all()` for production Postgres.
- If you want, I can scaffold an `alembic` folder and add a GitHub Action or Vercel Build Step to run migrations automatically.

If you want me to scaffold migrations and a CI step to run `flask db upgrade` on deploy, I can add that next.
