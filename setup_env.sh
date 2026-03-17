#!/bin/bash

read -p "Create virtual environment and install dependencies? [y/N] " answer

case "$answer" in
    [yY][eE][sS]|[yY])
        echo "Creating virtual environment..."
        python3 -m venv .venv
        echo "Installing dependencies..."
        .venv/bin/pip install --upgrade pip -q
        .venv/bin/pip install -r requirements.txt
        echo "Done. Activate with: source .venv/bin/activate"
        ;;
    *)
        echo "Aborted."
        ;;
esac
