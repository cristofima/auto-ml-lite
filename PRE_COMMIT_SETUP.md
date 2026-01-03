# Pre-commit Setup Guide

Pre-commit is configured with industry best practices for PyPI packages, ensuring code quality and consistency before every commit.

## 📋 Tools Included

### 1. **Ruff** (⚡ Ultra-fast)
- **Replaces**: black, isort, flake8, pylint
- Auto-formats code and organizes imports
- Detects 800+ error types
- Written in Rust (10-100x faster than alternatives)

### 2. **Pre-commit Hooks**
- Removes trailing whitespace
- Fixes end-of-file formatting
- Validates YAML/TOML syntax
- Detects large files and merge conflicts
- Prevents debug statements in production

### 3. **Bandit**
- Security vulnerability scanner
- Detects common Python security issues
- Excludes test/example files automatically

### 4. **Mypy** (Optional - Currently Disabled)
- Static type checking
- Can be enabled in `.pre-commit-config.yaml` when type annotations are added

## 🚀 Installation & Usage

### First-time Setup
```bash
pip install -e ".[dev]"
pre-commit install
```

### Daily Usage
Pre-commit runs automatically on `git commit`. To run manually:

```bash
# Check all files
pre-commit run --all-files

# Check only staged files
pre-commit run

# Update hook versions
pre-commit autoupdate
```

### Bypass Hooks (Not Recommended)
```bash
git commit --no-verify
```

## ⚙️ Configuration

### [pyproject.toml](pyproject.toml)
- **Ruff**: PyPI-recommended rules with ~100 character line length
- **Bandit**: Security scanning with sensible excludes
- **Pytest**: Test coverage configuration

### [.pre-commit-config.yaml](.pre-commit-config.yaml)
- Pinned hook versions for reproducibility
- Windows-compatible configuration
- Optimized for Python 3.11+
- Mixed line ending hook disabled to avoid conflicts with partial commits

### Ruff Configuration Highlights
```toml
[tool.ruff.lint]
select = ["E", "W", "F", "I", "N", "UP", "B", "C4", "SIM", "TCH", "RUF", "PL", "PERF"]
ignore = ["E501", "PLR0913", "PLR2004"]  # Line length, args, magic values

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]  # Allow re-exports
```

## 📝 Useful Commands

```bash
# List installed hooks
pre-commit run --list

# Run specific hook
pre-commit run ruff --all-files
pre-commit run ruff-format --all-files

# Clean cache
pre-commit clean

# Skip specific hook once
SKIP=bandit git commit -m "message"
```

## 🔧 Troubleshooting

### Hook modifies files but commit fails
This is intentional. After hooks auto-fix files:
1. Review the changes with `git diff`
2. Stage the fixes with `git add .`
3. Commit again

### Conflicts with unstaged files
Pre-commit only validates staged files. Use:
```bash
git add <specific-files>  # Stage only what you want to commit
git commit -m "message"
```

## 🎯 CI/CD Integration

Add to GitHub Actions workflow:
```yaml
- name: Run pre-commit
  uses: pre-commit/action@v3.0.0
```

## 📚 Resources

- [Pre-commit Documentation](https://pre-commit.com/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Python Packaging Guide](https://packaging.python.org/)
