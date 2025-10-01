#!/bin/bash

echo "🚀 Deploying Trading Advisor Backend..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create data directory
mkdir -p data/logs

echo "✅ Setup complete!"
echo "🔧 To start the server:"
echo "   source venv/bin/activate"
echo "   python server.py"
echo ""
echo "🌐 Or with gunicorn (production):"
echo "   gunicorn -w 4 -b 0.0.0.0:5000 server:app"
echo ""
echo "📱 Then open index.html in your browser"