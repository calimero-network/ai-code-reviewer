from ai_reviewer.session import ReviewSession


def test_review_session_tracks_quota():
    session = ReviewSession(repo="o/r", head_sha="abc", github_budget=3)
    assert session.remaining_github_budget() == 3
    session.consume_github_request()
    session.consume_github_request()
    assert session.remaining_github_budget() == 1
    session.consume_github_request()
    assert session.remaining_github_budget() == 0
    assert session.is_github_budget_exhausted()


def test_review_session_file_cache():
    session = ReviewSession(repo="o/r", head_sha="abc", github_budget=10)
    assert session.cached_file("a.py") is None
    session.store_file("a.py", "print('hi')")
    assert session.cached_file("a.py") == "print('hi')"
