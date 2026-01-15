PYTHON ?= python3
TWINE ?= twine
DIST_DIR := dist
TEST_PYPI_REPO := https://test.pypi.org/legacy/
TEST_PYPI_URL  := https://test.pypi.org/project/pyreflect/

.PHONY: build clean upload-test bump-version

install:
	uv pip install -e .

build: clean ## Build source and wheel distributions under dist/
	$(PYTHON) -m build

upload: bump-version build ## Bump version, rebuild, and upload to Test PyPI
	$(PYTHON) -m twine upload --repository testpypi "$(DIST_DIR)"/*
	@echo "✅ Uploaded to TestPyPI. Check release at: $(TEST_PYPI_URL)"

bump-version: ## Increment the patch version inside pyproject.toml using sed
	@set -e; \
	CURRENT_VERSION=$$(sed -nE 's/^version = "([0-9]+\.[0-9]+\.[0-9]+)"/\1/p' pyproject.toml | head -n 1); \
	test -n "$$CURRENT_VERSION"; \
	IFS=. ; set -- $$CURRENT_VERSION; IFS=' \t\n'; \
	NEW_PATCH=$$(( $$3 + 1 )); \
	NEW_VERSION="$$1.$$2.$$NEW_PATCH"; \
	sed -i.bak -E 's/(version = ")([0-9]+\.[0-9]+\.[0-9]+)(")/\1'"$$NEW_VERSION"'\3/g' pyproject.toml; \
	rm -f pyproject.toml.bak; \
	printf 'Version bumped to %s\n' "$$NEW_VERSION"

clean: ## Remove local build artifacts
	rm -rf $(DIST_DIR) build *.egg-info
