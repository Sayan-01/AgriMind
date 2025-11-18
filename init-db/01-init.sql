-- AgriMind Database Initialization
-- This script sets up the database with necessary extensions and initial schema

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Set up database encoding and locale
-- This ensures proper text handling for agricultural data
ALTER DATABASE agrimind SET default_text_search_config = 'english';

-- Create initial schema structure
CREATE SCHEMA IF NOT EXISTS public;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE agrimind TO agrimind;
GRANT ALL ON SCHEMA public TO agrimind;

-- Create initial tables if they don't exist

-- Knowledge base documents table for RAG system
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    source VARCHAR(255),
    metadata JSONB DEFAULT '{}',
    embedding vector(1536), -- OpenAI ada-002 embedding size
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_documents_embedding ON documents USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);
CREATE INDEX IF NOT EXISTS idx_documents_title_gin ON documents USING gin(to_tsvector('english', title));
CREATE INDEX IF NOT EXISTS idx_documents_content_gin ON documents USING gin(to_tsvector('english', content));

-- Plant disease detection results table
CREATE TABLE IF NOT EXISTS plant_detections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    image_path VARCHAR(255) NOT NULL,
    prediction JSONB NOT NULL,
    confidence FLOAT,
    model_version VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for plant detections
CREATE INDEX IF NOT EXISTS idx_plant_detections_confidence ON plant_detections(confidence);
CREATE INDEX IF NOT EXISTS idx_plant_detections_created_at ON plant_detections(created_at);

-- User queries and responses log table
CREATE TABLE IF NOT EXISTS query_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query TEXT NOT NULL,
    response TEXT,
    context_used JSONB DEFAULT '[]',
    response_time_ms INTEGER,
    user_id VARCHAR(255),
    session_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for query logs
CREATE INDEX IF NOT EXISTS idx_query_logs_user_id ON query_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_query_logs_session_id ON query_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_query_logs_created_at ON query_logs(created_at);

-- Function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for documents table
DROP TRIGGER IF EXISTS update_documents_updated_at ON documents;
CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Insert some sample data for testing
INSERT INTO documents (title, content, source) VALUES
('Plant Disease Basics', 'Plant diseases can be caused by fungi, bacteria, viruses, and other pathogens. Early detection is crucial for effective treatment.', 'agricultural_handbook')
ON CONFLICT DO NOTHING;

-- Display initialization completion message
DO $$
BEGIN
    RAISE NOTICE 'AgriMind database initialized successfully with pgvector extension and initial schema.';
END
$$;
