"""Module entry point. Allow ``python -m lstm_autoencoder`` to run the command-line interface."""

from lstm_autoencoder.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
