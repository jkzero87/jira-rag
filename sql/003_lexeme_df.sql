-- lexeme_df: per-lexeme document-frequency over jira.issues.fts, from ts_stat.
-- Used by the hybrid search to keep only rare question lexemes (ndoc < 2% of issues)
-- in the OR keyword query, so common words (the/with/issue/...) don't dilute RRF.

DROP TABLE IF EXISTS jira.lexeme_df;

CREATE TABLE jira.lexeme_df (
    word  text PRIMARY KEY,
    ndoc  integer NOT NULL
);

INSERT INTO jira.lexeme_df (word, ndoc)
SELECT word, ndoc
FROM ts_stat('SELECT fts FROM jira.issues');
