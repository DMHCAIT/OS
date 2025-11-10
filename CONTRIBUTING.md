# Contributing to AI Lead Management & Voice Communication System

Thank you for your interest in contributing to the AI Lead Management & Voice Communication System! We welcome contributions from developers of all skill levels.

## 🚀 Getting Started

### Prerequisites

- Node.js 18+
- Python 3.11+
- Docker Desktop
- Git

### Setup Development Environment

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai-lead-management-system
   ```

2. **Run the setup script**
   ```bash
   ./scripts/setup.sh
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

4. **Start the development servers**
   ```bash
   ./scripts/start-dev.sh
   ```

## 🏗️ Project Structure

```
├── backend/                 # Python FastAPI backend
│   ├── app/
│   │   ├── api/            # API endpoints
│   │   ├── core/           # Core functionality
│   │   ├── ml/             # Machine learning models
│   │   ├── voice/          # Voice AI services
│   │   └── models/         # Data models
│   └── tests/              # Backend tests
├── frontend/               # React TypeScript frontend
│   ├── src/
│   │   ├── components/     # Reusable components
│   │   ├── pages/          # Page components
│   │   ├── services/       # API services
│   │   └── utils/          # Utility functions
│   └── public/             # Static assets
├── ml-models/              # Machine learning models
├── voice-ai/               # Voice AI components
├── database/               # Database schemas and migrations
└── scripts/                # Deployment and utility scripts
```

## 🛠️ Development Guidelines

### Code Style

**Python (Backend)**
- Follow PEP 8 style guidelines
- Use type hints for all functions
- Write docstrings for all modules, classes, and functions
- Use `black` for code formatting
- Use `flake8` for linting

**TypeScript/React (Frontend)**
- Use TypeScript for all components
- Follow React best practices and hooks patterns
- Use functional components with hooks
- Implement proper error boundaries
- Use Tailwind CSS for styling

### Commit Messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools

**Examples:**
```
feat(voice-ai): add real-time speech recognition
fix(api): resolve lead scoring calculation error
docs(readme): update installation instructions
```

### Testing

**Backend Testing**
```bash
cd backend
source venv/bin/activate
pytest tests/ -v --cov=app
```

**Frontend Testing**
```bash
cd frontend
npm test
npm run test:coverage
```

### API Development

When adding new API endpoints:

1. **Define the data model** in `backend/app/models/`
2. **Create the endpoint** in `backend/app/api/endpoints/`
3. **Add to router** in `backend/app/api/__init__.py`
4. **Write tests** in `backend/tests/`
5. **Update OpenAPI documentation**

### Frontend Development

When adding new React components:

1. **Create component** in appropriate directory
2. **Add TypeScript interfaces** for props and state
3. **Implement error handling**
4. **Add unit tests**
5. **Update Storybook stories** (if applicable)

## 🧪 Testing Strategy

### Backend Testing

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test API endpoints and database interactions
- **ML Model Tests**: Test machine learning model accuracy and performance
- **Voice AI Tests**: Test voice processing and generation

### Frontend Testing

- **Unit Tests**: Test individual components and utilities
- **Integration Tests**: Test component interactions and API calls
- **E2E Tests**: Test complete user workflows
- **Accessibility Tests**: Ensure WCAG compliance

### Testing Commands

```bash
# Run all backend tests
cd backend && pytest

# Run frontend tests
cd frontend && npm test

# Run E2E tests
npm run test:e2e

# Run all tests with coverage
npm run test:coverage
```

## 📝 Documentation

### Code Documentation

- **Python**: Use docstrings with type hints
- **TypeScript**: Use JSDoc comments for complex functions
- **API**: Document all endpoints with OpenAPI/Swagger
- **Components**: Document props and usage examples

### Writing Documentation

1. **API Documentation**: Auto-generated from FastAPI decorators
2. **Component Documentation**: Use Storybook for component docs
3. **User Guides**: Written in Markdown in `/docs` folder
4. **Architecture Docs**: High-level system design documentation

## 🔄 Pull Request Process

1. **Fork the repository** and create a feature branch
2. **Make your changes** following the coding guidelines
3. **Write tests** for new functionality
4. **Update documentation** as needed
5. **Run the test suite** and ensure all tests pass
6. **Create a pull request** with a clear description

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment details** (OS, browser, versions)
2. **Steps to reproduce** the issue
3. **Expected vs actual behavior**
4. **Screenshots or logs** if applicable
5. **Minimal reproduction case**

Use the bug report template in GitHub issues.

## 💡 Feature Requests

For new features:

1. **Check existing issues** to avoid duplicates
2. **Describe the problem** the feature would solve
3. **Propose a solution** with implementation details
4. **Consider alternatives** and their trade-offs
5. **Estimate complexity** and breaking changes

## 🚀 Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH`
- Major: Breaking changes
- Minor: New features (backward compatible)
- Patch: Bug fixes

### Release Steps

1. **Update version numbers** in package.json and __init__.py
2. **Update CHANGELOG.md** with release notes
3. **Create release tag** and GitHub release
4. **Deploy to staging** for testing
5. **Deploy to production** after approval

## 🤝 Community Guidelines

### Code of Conduct

- **Be respectful** and inclusive
- **Collaborate constructively**
- **Help others learn** and grow
- **Focus on the issue**, not the person
- **Give credit** where due

### Getting Help

- **GitHub Discussions**: For general questions and ideas
- **GitHub Issues**: For bug reports and feature requests
- **Discord/Slack**: For real-time chat (if available)
- **Documentation**: Check docs before asking questions

## 🎉 Recognition

Contributors will be:
- Listed in the CONTRIBUTORS.md file
- Mentioned in release notes for significant contributions
- Invited to join the core team for sustained contributions

## 📋 Development Checklist

Before submitting code:

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Commit messages follow convention
- [ ] No sensitive data in commits
- [ ] Performance impact considered
- [ ] Accessibility requirements met
- [ ] Security implications reviewed

Thank you for contributing to the AI Lead Management & Voice Communication System! 🙌