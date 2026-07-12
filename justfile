# Affiche les commandes disponibles
default:
    @just --list

# Supprime les checkpoints PyTorch Lightning
clean-checkpoints:
    rm -rf checkpoints/*

# Supprime les logs Hydra
clean-logs:
    rm -rf outputs/ multirun/ logs/

# Nettoyage complet
clean: clean-checkpoints clean-logs
    @echo "✓ Projet nettoyé"

# Nettoyage + artefacts Python
clean-all: clean
    find . -type d -name __pycache__ -exec rm -rf {} +
    find . -name "*.pyc" -delete
    rm -rf .pytest_cache/ dist/ *.egg-info/

# Lancer un entraînement
train config='default':
    uv run python train.py --config-name={{config}}
