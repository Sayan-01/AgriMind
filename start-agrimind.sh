#!/bin/bash

# AgriMind Docker Startup Script
# This script starts the entire AgriMind stack with databases

set -e

echo "🌱 Starting AgriMind with Docker Compose..."
echo "=========================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop first."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install Docker Compose."
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📋 Creating .env file from template..."
    cp .env.example .env
    echo "✅ Created .env file. You may want to customize it with your API keys."
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data models uploads logs

# Build and start services
echo "🚀 Building and starting services..."
echo "This may take a few minutes on the first run..."

docker-compose up --build -d

echo ""
echo "⏳ Waiting for services to be healthy..."

# Wait for database to be ready
echo "🗄️  Waiting for PostgreSQL..."
until docker-compose exec -T db pg_isready -U agrimind -d agrimind > /dev/null 2>&1; do
    printf "."
    sleep 2
done
echo " ✅ PostgreSQL is ready!"

# Wait for Redis to be ready
echo "🔄 Waiting for Redis..."
until docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; do
    printf "."
    sleep 1
done
echo " ✅ Redis is ready!"

# Wait for API to be ready
echo "🔌 Waiting for API server..."
until curl -f http://localhost:8000/health > /dev/null 2>&1; do
    printf "."
    sleep 2
done
echo " ✅ API server is ready!"

echo ""
echo "🎉 AgriMind is now running!"
echo "=========================================="
echo "📊 Services Status:"
docker-compose ps
echo ""
echo "🌐 Access Points:"
echo "   • API Server: http://localhost:8000"
echo "   • API Health: http://localhost:8000/health"
echo "   • API Docs: http://localhost:8000/docs"
echo "   • Database: localhost:5432 (agrimind/agrimind)"
echo "   • Redis: localhost:6379"
echo ""
echo "📋 Useful Commands:"
echo "   • View logs: docker-compose logs -f"
echo "   • Stop services: docker-compose down"
echo "   • Restart: docker-compose restart"
echo ""
echo "💡 For more information, see DOCKER_SETUP.md"
