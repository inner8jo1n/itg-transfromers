from itg_transformers import main


def test_main_prints_greeting(capsys):
    main()
    assert "itg-transformers" in capsys.readouterr().out
