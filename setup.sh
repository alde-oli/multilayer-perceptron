#!/bin/bash

# Check if virtualenv is installed, if not, install it
if ! pip show virtualenv > /dev/null 2>&1; then
    pip install --user virtualenv
fi

# Create a virtual environment using virtualenv
python3 -m virtualenv mlp

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source ./mlp/Scripts/activate
    pip install -r requirements.txt
    echo "L'installation est terminée ! Vous entrez maintenant dans l'environnement virtuel."

    cmd.exe /k "mlp\\Scripts\\activate.bat"
else
    source ./mlp/bin/activate
    alias py=python3
    pip install -r requirements.txt
    echo "L'installation est terminée ! Vous entrez maintenant dans l'environnement virtuel."
    clear
   exec "$SHELL"
fi