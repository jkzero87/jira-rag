--
-- PostgreSQL database dump
--

\restrict szcBvFcDVp1U0DTh09X6Mgk2gCvblKXSIrTWH3Hw01WaynYad2Lr3CoFz4Oq46C

-- Dumped from database version 16.15 (Debian 16.15-1.pgdg12+2)
-- Dumped by pg_dump version 16.15 (Debian 16.15-1.pgdg12+2)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: jira; Type: SCHEMA; Schema: -; Owner: deuser
--

CREATE SCHEMA jira;


ALTER SCHEMA jira OWNER TO deuser;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: issue_chunks; Type: TABLE; Schema: jira; Owner: deuser
--

CREATE TABLE jira.issue_chunks (
    chunk_id bigint NOT NULL,
    issue_key text NOT NULL,
    strategy text NOT NULL,
    chunk_index integer NOT NULL,
    content text NOT NULL,
    token_count integer,
    embedding public.vector(1024),
    embedded_at timestamp with time zone
);


ALTER TABLE jira.issue_chunks OWNER TO deuser;

--
-- Name: issue_chunks_chunk_id_seq; Type: SEQUENCE; Schema: jira; Owner: deuser
--

CREATE SEQUENCE jira.issue_chunks_chunk_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE jira.issue_chunks_chunk_id_seq OWNER TO deuser;

--
-- Name: issue_chunks_chunk_id_seq; Type: SEQUENCE OWNED BY; Schema: jira; Owner: deuser
--

ALTER SEQUENCE jira.issue_chunks_chunk_id_seq OWNED BY jira.issue_chunks.chunk_id;


--
-- Name: issues; Type: TABLE; Schema: jira; Owner: deuser
--

CREATE TABLE jira.issues (
    issue_key text NOT NULL,
    issue_id bigint NOT NULL,
    project text NOT NULL,
    summary text,
    description text,
    issue_type text,
    priority text,
    status text,
    resolution text,
    created timestamp with time zone NOT NULL,
    updated timestamp with time zone,
    resolutiondate timestamp with time zone,
    raw jsonb,
    ingested_at timestamp with time zone DEFAULT now() NOT NULL,
    date_suspect boolean DEFAULT false NOT NULL
);


ALTER TABLE jira.issues OWNER TO deuser;

--
-- Name: issue_chunks chunk_id; Type: DEFAULT; Schema: jira; Owner: deuser
--

ALTER TABLE ONLY jira.issue_chunks ALTER COLUMN chunk_id SET DEFAULT nextval('jira.issue_chunks_chunk_id_seq'::regclass);


--
-- Name: issue_chunks issue_chunks_issue_key_strategy_chunk_index_key; Type: CONSTRAINT; Schema: jira; Owner: deuser
--

ALTER TABLE ONLY jira.issue_chunks
    ADD CONSTRAINT issue_chunks_issue_key_strategy_chunk_index_key UNIQUE (issue_key, strategy, chunk_index);


--
-- Name: issue_chunks issue_chunks_pkey; Type: CONSTRAINT; Schema: jira; Owner: deuser
--

ALTER TABLE ONLY jira.issue_chunks
    ADD CONSTRAINT issue_chunks_pkey PRIMARY KEY (chunk_id);


--
-- Name: issues issues_pkey; Type: CONSTRAINT; Schema: jira; Owner: deuser
--

ALTER TABLE ONLY jira.issues
    ADD CONSTRAINT issues_pkey PRIMARY KEY (issue_key);


--
-- Name: idx_chunks_strategy; Type: INDEX; Schema: jira; Owner: deuser
--

CREATE INDEX idx_chunks_strategy ON jira.issue_chunks USING btree (strategy);


--
-- Name: idx_issues_created; Type: INDEX; Schema: jira; Owner: deuser
--

CREATE INDEX idx_issues_created ON jira.issues USING btree (created);


--
-- Name: idx_issues_priority; Type: INDEX; Schema: jira; Owner: deuser
--

CREATE INDEX idx_issues_priority ON jira.issues USING btree (priority);


--
-- Name: idx_issues_project; Type: INDEX; Schema: jira; Owner: deuser
--

CREATE INDEX idx_issues_project ON jira.issues USING btree (project);


--
-- Name: idx_issues_updated; Type: INDEX; Schema: jira; Owner: deuser
--

CREATE INDEX idx_issues_updated ON jira.issues USING btree (updated);


--
-- Name: issue_chunks issue_chunks_issue_key_fkey; Type: FK CONSTRAINT; Schema: jira; Owner: deuser
--

ALTER TABLE ONLY jira.issue_chunks
    ADD CONSTRAINT issue_chunks_issue_key_fkey FOREIGN KEY (issue_key) REFERENCES jira.issues(issue_key) ON DELETE CASCADE;


--
-- PostgreSQL database dump complete
--

\unrestrict szcBvFcDVp1U0DTh09X6Mgk2gCvblKXSIrTWH3Hw01WaynYad2Lr3CoFz4Oq46C

