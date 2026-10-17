-- Read-only login for Power BI. Run in Supabase -> SQL Editor. Change the password first.
CREATE ROLE powerbi_reader WITH LOGIN PASSWORD 'change-me-to-a-long-password';
GRANT USAGE ON SCHEMA public TO powerbi_reader;
GRANT SELECT ON bi_jobs, bi_free_time, bi_deliveries, bi_volumes,
                bi_carriers, bi_destinations, bi_refresh TO powerbi_reader;
-- The feed recreates its tables; give the reader access to new tables automatically.
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO powerbi_reader;
