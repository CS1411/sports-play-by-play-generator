"""Plumbing check: the command line starts up."""


def test_the_command_line_runs(capsys):
    from app.cli import main
    main(["hello"])
    assert "is ready" in capsys.readouterr().out
