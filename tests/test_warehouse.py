import pytest

from bleedtracker.warehouse import MissingTokenError, Source, query, read_token


def test_environment_variable_comes_first(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("HF_TOKEN=from-file\n", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "from-env")
    assert read_token(tmp_path) == "from-env"


def test_env_file_in_a_parent_folder(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    (tmp_path / ".env").write_text('# comment\nHF_TOKEN="from-file"\n', encoding="utf-8")
    inner = tmp_path / "notebooks"
    inner.mkdir()
    assert read_token(inner) == "from-file"


def test_empty_token_is_an_error(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    (tmp_path / ".env").write_text("HF_TOKEN=\n", encoding="utf-8")
    with pytest.raises(MissingTokenError):
        read_token(tmp_path)


def test_source_quotes_paths():
    assert Source(root="C:/it's").dim_clients() == "read_parquet('C:/it''s/dim_clients.parquet')"


class FakeResult:
    def df(self):
        return "frame"


class FakeConnection:
    def __init__(self, failures):
        self.failures = list(failures)
        self.calls = 0

    def execute(self, sql):
        self.calls += 1
        if self.failures:
            raise RuntimeError(self.failures.pop(0))
        return FakeResult()


def test_429_is_retried_with_growing_waits():
    waits = []
    con = FakeConnection(["HTTP 429 Too Many Requests", "HTTP 429 Too Many Requests"])
    assert query(con, "SELECT 1", sleep=waits.append) == "frame"
    assert waits == [20, 40]


def test_other_errors_are_not_retried():
    waits = []
    con = FakeConnection(["HTTP 403 Forbidden"])
    with pytest.raises(RuntimeError, match="403"):
        query(con, "SELECT 1", sleep=waits.append)
    assert waits == []
    assert con.calls == 1


def test_gives_up_after_six_tries():
    waits = []
    con = FakeConnection(["HTTP 429"] * 6)
    with pytest.raises(RuntimeError, match="429"):
        query(con, "SELECT 1", sleep=waits.append)
    assert waits == [20, 40, 60, 80, 100]
    assert con.calls == 6
