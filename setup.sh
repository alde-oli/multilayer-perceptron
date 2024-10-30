#!/bin/bash
python3 -m venv mlp

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source ./mlp/Scripts/activate
    pip install -r requirements.txt
    echo "L'installation est terminée ! Vous entrez maintenant dans l'environnement virtuel."

    cmd.exe /k "mlp\\Scripts\\activate.bat"
else
    source ./mlp/bin/activate
    pip install -r requirements.txt
    echo "L'installation est terminée ! Vous entrez maintenant dans l'environnement virtuel."
    
    exec "$SHELL" --init-file <(echo "source ./mlp/bin/activate")
fi
