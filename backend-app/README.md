## Development Setup

### Prerequisites
- Python 3.8 or higher
- UV package installer (recommended) or pip

### Installation with UV (Recommended)
1. Install UV:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create and activate virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
# OR
.venv\Scripts\activate     # On Windows
```

3. Install dependencies:
```bash
uv pip install -r requirements.txt
```

4. Run the application:
```bash
uv run main.py
```

