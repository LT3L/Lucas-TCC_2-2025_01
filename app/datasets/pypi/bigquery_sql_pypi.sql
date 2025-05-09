CREATE TABLE tcc2.pypi AS
SELECT
  timestamp,
  country_code,
  url,
  project,
  file
FROM
  `bigquery-public-data.pypi.file_downloads`
WHERE
  TIMESTAMP_TRUNC(timestamp, DAY) = TIMESTAMP("2023-01-01");