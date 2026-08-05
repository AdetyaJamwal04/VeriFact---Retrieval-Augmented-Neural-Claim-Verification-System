# Contributing

Thank you for your interest in contributing to VeriFact!

## Getting Started

1. Fork the repository.
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/VeriFact---Retrieval-Augmented-Neural-Claim-Verification-System.git
   cd VeriFact---Retrieval-Augmented-Neural-Claim-Verification-System
   ```
3. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows: .\.venv\Scripts\activate
   # macOS/Linux: source .venv/bin/activate
   ```
4. Install dependencies:
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
5. Copy the environment template and fill in any keys you need:
   ```bash
   copy .env.example .env
   # or: cp .env.example .env
   ```
6. Make your changes.
7. Run tests: `pytest tests/ -v`
8. Commit: `git commit -m "feat: Add your feature"`
9. Push: `git push origin feature/your-feature-name`
10. Open a Pull Request.

## Code Style

- Follow PEP 8 for Python code
- Use type hints for function parameters and returns
- Add docstrings to all public functions
- Keep functions focused and under 50 lines when possible

## Commit Messages

Use conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `refactor:` Code refactoring
- `test:` Adding tests

## Testing

All new features should include tests:

```bash
pytest tests/ -v
```

## Pull Request Process

1. Ensure all tests pass
2. Update documentation if needed
3. Add a clear description of changes
4. Link any related issues

## Questions?

Open an issue or reach out to the maintainers.
