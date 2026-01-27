#!/bin/bash

# Glaucoma Screening System Runner

case "$1" in
    "app")
        echo "Starting Streamlit App..."
        ./venv/bin/streamlit run src/app/main.py
        ;;
    "train-seg")
        echo "Starting Segmentation Training..."
        export PYTHONPATH=$PYTHONPATH:.
        ./venv/bin/python src/training/segmentation_trainer.py
        ;;
    "train-cls")
        echo "Starting Classification Training..."
        export PYTHONPATH=$PYTHONPATH:.
        ./venv/bin/python src/training/classification_trainer.py
        ;;
    "setup-data")
        echo "Setting up data..."
        export PYTHONPATH=$PYTHONPATH:.
        ./venv/bin/python src/data/setup_data.py
        ;;
    *)
        echo "Usage: ./run.sh [app | train-seg | train-cls | setup-data]"
        exit 1
        ;;
esac
