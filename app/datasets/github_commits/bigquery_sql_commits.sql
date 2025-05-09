CREATE TABLE tcc2.commits AS
SELECT
  commit AS commit,
  author.name AS author_name,
  message AS commit_message,
  repo_name AS repository,
  committer.date.seconds AS commit_time_utc
FROM
  `bigquery-public-data.github_repos.commits`;