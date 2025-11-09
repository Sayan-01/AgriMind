#!/bin/bash

# AgriMind ML Inference Quick Start Script
echo "🌱 AgriMind ML Inference Quick Start"
echo "===================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip first."
    exit 1
fi

echo "✅ Python and pip found"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check if Kaggle is configured
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "⚠️  Kaggle API not configured. Please:"
    echo "   1. Go to https://www.kaggle.com/account"
    echo "   2. Create API token"
    echo "   3. Download kaggle.json"
    echo "   4. Place it at ~/.kaggle/kaggle.json"
    echo "   5. Run: chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    echo "🔄 Continuing without dataset download..."
else
    echo "✅ Kaggle API configured"
fi

echo ""
echo "🚀 Setup complete! Choose an option:"
echo "1. Download datasets and run full pipeline"
echo "2. Start with preprocessing (if datasets already downloaded)"
echo "3. Start API server (if model is trained)"
echo "4. Run tests"
echo ""
read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo "🔄 Running full pipeline..."
        python src/main.py pipeline
        ;;
    2)
        echo "🔄 Starting preprocessing..."
        python src/main.py preprocess
        ;;
    3)
        echo "🔄 Starting API server..."
        python src/main.py serve
        ;;
    4)
        echo "🧪 Running tests..."
        python tests/test_ml_inference.py
        ;;
    *)
        echo "ℹ️  Manual commands available:"
        echo "  python src/main.py --help"
        echo "  python src/main.py download"
        echo "  python src/main.py preprocess"
        echo "  python src/main.py train"
        echo "  python src/main.py evaluate"
        echo "  python src/main.py serve"
        ;;
esac

echo ""
echo "✨ AgriMind ML Inference is ready!"
echo "📚 Check README.md for detailed usage instructions"
