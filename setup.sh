#!/bin/bash

echo "🌾 Setting up AgriMind..."

# Check if virtual environment exists, create if not
if [ ! -d ".venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source .venv/bin/activate

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install -r apps/api/requirements.txt
pip install -r apps/rag-script/requirements.txt  
pip install -r apps/ml-inference/requirements.txt
pip install -r packages/kb/requirements.txt

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
pnpm install

# Setup environment files
echo "⚙️ Setting up environment files..."
if [ ! -f "apps/api/.env" ]; then
    cp apps/api/.env.example apps/api/.env
    echo "📝 Created apps/api/.env from template"
fi

if [ ! -f "apps/rag-script/.env" ]; then
    cp apps/rag-script/.env.example apps/rag-script/.env
    echo "📝 Created apps/rag-script/.env from template"
fi

# Start Docker services
echo "🐳 Starting Docker services..."
docker compose -f infra/compose.yml up -d

# Wait for database to be ready
echo "⏳ Waiting for database to be ready..."
sleep 10

# Setup RAG system
echo "🧠 Setting up RAG system..."
cd apps/rag-script
python setup_db.py
python load_knowledge_base.py
cd ../..

echo "✅ Setup complete!"
echo ""
echo "🔑 IMPORTANT: Set your Gemini API key in apps/rag-script/.env"
echo "   Get your API key from: https://makersuite.google.com/app/apikey"
echo "   Edit the file and replace 'your_gemini_api_key_here' with your actual key"
echo ""
echo "🚀 Run 'pnpm dev' to start the application"
echo "🌐 Open http://localhost:3000 in your browser"
