-- Read-only login for Power BI. Run in Supabase -> SQL Editor. Change the password first.
CREATE ROLE powerbi_reader WITH LOGIN PASSWORD 'change-me-to-a-long-password';
GRANT USAGE ON SCHEMA public TO powerbi_reader;
GRANT SELECT ON bi_jobs, bi_free_time, bi_deliveries, bi_volumes,
                bi_carriers, bi_destinations, bi_refresh TO powerbi_reader;
-- The bi_ tables have Row Level Security on, so the reader also needs a read policy.
-- The feed recreates its tables on every app start and then adds the grant and the
-- policy itself, so these lines are only needed until the next restart.
DO $$
DECLARE t text;
BEGIN
  FOREACH t IN ARRAY ARRAY['bi_jobs', 'bi_free_time', 'bi_deliveries', 'bi_volumes',
                           'bi_carriers', 'bi_destinations', 'bi_refresh'] LOOP
    EXECUTE format('DROP POLICY IF EXISTS bi_read ON %I', t);
    EXECUTE format('CREATE POLICY bi_read ON %I FOR SELECT TO powerbi_reader USING (true)', t);
  END LOOP;
END $$;
