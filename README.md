# AgriMind

An AI-powered agricultural assistance platform combining plant disease detection and intelligent knowledge retrieval for farmers and agricultural professionals.

## 🌟 Features

- **🔬 Plant Disease Detection**: AI-powered disease identification using Vision Transformer models
- **🧠 Agricultural Knowledge Assistant**: RAG-powered Q&A system for farming guidance
- **🌐 Web Interface**: Modern React frontend with image upload and voice input
- **🔗 Integrated Analysis**: Combined image + query processing for comprehensive insights
- **🌾 Crop-Specific Insights**: Specialized knowledge for Corn, Potato, Rice, and Wheat
- **📊 Market Intelligence**: Real-time market data and price information
- **🗺️ Regional Expertise**: Focused on West Bengal agriculture and practices

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

The easiest way to run AgriMind with all dependencies:

```bash
# Clone the repository
git clone <repository-url>
cd AgriMind

# Start everything with one command (includes databases!)
./start-agrimind.sh    # macOS/Linux
# OR
start-agrimind.bat     # Windows

# Or manually with docker-compose
docker-compose up --build
```

This will start:

- PostgreSQL database with pgvector extension
- Redis for caching
- AgriMind API server
- All services accessible at localhost

### Option 2: Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd AgriMind

# One-command setup (creates venv, installs dependencies, starts Docker)
pnpm setup        # macOS/Linux
# OR
pnpm setup:windows  # Windows

# Start the application
pnpm dev
```

Open `http://localhost:3000` in your browser and start analyzing your crops!

## 🔑 Environment Setup

**Important**: You need a Gemini API key for the AI assistant to work properly.

1. **Get your Gemini API key**:
   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key (free tier available)

2. **Set your API key**:

   ```bash
   # Edit the environment file
   nano apps/rag-script/.env

   # Replace this line:
   GEMINI_API_KEY=your_gemini_api_key_here

   # With your actual key:
   GEMINI_API_KEY=your_actual_api_key_here
   ```

3. **Optional configurations**:
   - `apps/api/.env` - API server settings (defaults work fine)
   - `apps/rag-script/.env` - RAG system settings (defaults work fine)

### Plant Disease Detection

Detect diseases in plant images with high accuracy:

```bash
# Detect disease in an image (human-readable output)
npm run detect-disease path/to/image.jpg

# Get JSON output for API integration
npm run detect-disease path/to/image.jpg -- --json

# Quiet mode (suppress loading messages)
npm run detect-disease path/to/image.jpg -- --quiet --json
```

**Supported diseases**: 13+ conditions across Corn, Potato, Rice, and Wheat including rusts, blights, spots, and healthy conditions.

### Agricultural Knowledge Assistant

Get intelligent answers to farming questions:

```bash
# Interactive mode - ask questions interactively
npm run ask-agrimind

# Single query with human-readable output
npm run ask-agrimind -- --query "What are the best crops for West Bengal during Kharif season?"

# Get JSON output for API integration
npm run ask-agrimind -- --query "Rice prices in Kolkata" --format json

# Market-specific queries
npm run ask-agrimind -- --query "Current vegetable prices" --type market

# Regional queries with filters
npm run ask-agrimind -- --query "Farming practices in Murshidabad" --region "Murshidabad"
```

**Knowledge base includes**: ICAR reports, market data, farming advisories, weather patterns, and crop recommendations.

## 📋 Requirements

- **Python 3.8+**
- **Node.js 18+** and pnpm
- **Docker**

That's it! The setup script handles everything else.

## 🔧 Manual Setup (if needed)

If you prefer manual setup or encounter issues:

1. **Create Python virtual environment**:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install all dependencies**:

   ```bash
   pnpm install
   pip install -r apps/api/requirements.txt
   pip install -r apps/rag-script/requirements.txt
   pip install -r apps/ml-inference/requirements.txt
   pip install -r packages/kb/requirements.txt
   ```

3. **Start Docker services**:

   **Option A: Complete Stack (Recommended)**

   ```bash
   # Start everything including databases
   docker-compose up --build
   ```

   **Option B: Infrastructure Only**

   ```bash
   docker compose -f infra/compose.yml up -d
   ```

4. **Initialize RAG system** (if using Option B):
   ```bash
   cd apps/rag-script && python setup_db.py && python load_knowledge_base.py && cd ../..
   ```

## 🏗️ Architecture

```
AgriMind/
├── apps/
│   ├── api/              # Backend API server
│   ├── frontend/         # Next.js web interface
│   ├── ml-inference/     # Disease detection service
│   └── rag-script/       # Knowledge retrieval service
├── packages/
│   ├── ui/               # Shared UI components
│   ├── kb/               # Knowledge base processing
│   └── typescript-config/ # Shared TypeScript configs
└── infra/
    └── compose.yml       # Docker services
```

## 🧪 Test Your Setup

```bash
# Check if environment variables are set correctly
pnpm check-env

# Test disease detection
npm run detect-disease apps/ml-inference/test_leaf.jpg

# Test RAG system (requires Gemini API key)
npm run ask-agrimind -- --query "What crops are good for West Bengal?"

# Check system health
npm run ask-agrimind -- --health-check
```

## � Docker Setup

AgriMind includes a complete Docker setup with databases for easy deployment:

### Quick Start with Docker

```bash
# Start the complete stack (recommended)
./start-agrimind.sh    # macOS/Linux
start-agrimind.bat     # Windows

# Or manually
docker-compose up --build
```

### Services Included

- **PostgreSQL**: Database with pgvector extension for embeddings
- **Redis**: Caching and session management
- **AgriMind API**: Complete backend with all services

### Access Points

- API Server: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health
- Database: localhost:5432 (user: agrimind, password: agrimind)
- Redis: localhost:6379

### Docker Commands

```bash
# Start services
docker-compose up --build

# View logs
docker-compose logs -f api

# Stop services
docker-compose down

# Rebuild specific service
docker-compose build api && docker-compose up -d api
```

For detailed Docker setup information, see [DOCKER_SETUP.md](DOCKER_SETUP.md).

## �🔧 Development

```bash
# Start all services in development mode
pnpm dev

# Run linting across all packages
pnpm lint

# Build all packages
pnpm build
```

## 📖 Documentation

- [Plant Disease Detection](./apps/ml-inference/README.md) - Detailed ML inference documentation
- [RAG System](./apps/rag-script/README.md) - Knowledge retrieval system guide
- [Knowledge Base Processing](./packages/kb/README.md) - Data processing pipeline
- [Frontend](./apps/frontend/README.md) - Web interface documentation
