-- Runs once, when the volume is first created: a separate database for the tests,
-- so running pytest never touches the ingested data.
CREATE DATABASE docsqa_test OWNER docsqa;
