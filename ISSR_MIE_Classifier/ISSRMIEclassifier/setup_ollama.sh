#!/bin/bash

# Setup script for Ollama MIE classification model

echo "🚀 Setting up Ollama MIE Classification Model..."

# Wait for Ollama to be ready
echo "Waiting for Ollama to be ready..."
while ! curl -s http://localhost:11434/api/tags > /dev/null; do
    echo "Ollama not ready yet, waiting..."
    sleep 2
done

echo "✅ Ollama is ready!"

# Check if MIE model already exists
if curl -s http://localhost:11434/api/tags | grep -q "mie-expert"; then
    echo "✅ MIE Expert model already exists"
else
    echo "📦 Creating MIE Expert model..."
    
    # Create the model using the Modelfile
    cd ollama/models
    ollama create mie-expert -f Modelfile
    
    if [ $? -eq 0 ]; then
        echo "✅ MIE Expert model created successfully!"
    else
        echo "❌ Failed to create MIE Expert model"
        exit 1
    fi
fi

# Test the model
echo "🧪 Testing MIE Expert model..."
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mie-expert",
    "prompt": "Test article: U.S. military conducted airstrikes in Syria. Is this an MIE?",
    "stream": false
  }'

echo ""
echo "🎉 Ollama setup complete!"
echo "You can now run the MIE classification system." 