\set ON_ERROR_STOP 0
SELECT '=== version ===' AS step;
SELECT extname, extversion FROM pg_extension WHERE extname='vector';

SELECT '=== setup ===' AS step;
DROP SCHEMA IF EXISTS scratch_vec CASCADE;
CREATE SCHEMA scratch_vec;

SELECT '=== column dim: vector(16000) (expect success) ===' AS step;
CREATE TABLE scratch_vec.t16000 (id int, v vector(16000));

SELECT '=== column dim: vector(16001) (expect error) ===' AS step;
CREATE TABLE scratch_vec.t16001 (id int, v vector(16001));

SELECT '=== hnsw: vector(2000) index (expect success) ===' AS step;
CREATE TABLE scratch_vec.t2000 (id int, v vector(2000));
INSERT INTO scratch_vec.t2000 (v) VALUES (array_fill(0::float8, ARRAY[1:2000])::vector);
CREATE INDEX ON scratch_vec.t2000 USING hnsw (v vector_l2_ops);

SELECT '=== hnsw: vector(2001) index (expect error) ===' AS step;
CREATE TABLE scratch_vec.t2001 (id int, v vector(2001));
CREATE INDEX ON scratch_vec.t2001 USING hnsw (v vector_l2_ops);

SELECT '=== ivfflat: vector(2000) index (expect success) ===' AS step;
CREATE TABLE scratch_vec.t2000i (id int, v vector(2000));
INSERT INTO scratch_vec.t2000i (v) VALUES (array_fill(0::float8, ARRAY[1:2000])::vector);
CREATE INDEX ON scratch_vec.t2000i USING ivfflat (v vector_l2_ops) WITH (lists = 1);

SELECT '=== ivfflat: vector(2001) index (expect error) ===' AS step;
CREATE TABLE scratch_vec.t2001i (id int, v vector(2001));
CREATE INDEX ON scratch_vec.t2001i USING ivfflat (v vector_l2_ops) WITH (lists = 1);

SELECT '=== requested test: vector(2560) column + hnsw index ===' AS step;
CREATE TABLE scratch_vec.t2560 (id int, v vector(2560));
CREATE INDEX ON scratch_vec.t2560 USING hnsw (v vector_l2_ops);

SELECT '=== cleanup: drop scratch schema ===' AS step;
DROP SCHEMA scratch_vec CASCADE;
SELECT '=== final: scratch_vec should not exist ===' AS step;
SELECT nspname FROM pg_namespace WHERE nspname='scratch_vec';
