# Developer Guide: How to Write Dev Logs & ADRs

This guide provides best practices for maintaining the **Nebium Developer Journey & Engineering Log**.

---

## 🛠️ Tooling & Local Preview

This documentation site is built with **MkDocs** and **Material for MkDocs**.

### 1. Start Local Live-Reload Server
Run the local preview server in your terminal:

```bash
uv run mkdocs serve
```
Then visit `http://localhost:8000` in your browser. Any change to markdown files will instantly trigger a live reload.

### 2. Build the Static Production Site
To compile the markdown documentation into static HTML files:

```bash
uv run mkdocs build
```
This generates the full documentation site into `docs/dev_journey/`, which is directly deployed via GitHub Pages and linked from `docs/index.html`.

---

## ✍️ How to Add a New Journal Entry

1. Make a copy of `dev_docs/journal/template.md`:
   ```bash
   cp dev_docs/journal/template.md dev_docs/journal/05-my-new-feature.md
   ```
2. Fill out the front matter:
   - Date, Author, Status, Tags
   - Context & Goal
   - What I tried, What failed, What worked
   - Today I Learned (TIL)
3. Add the new entry to the `nav` section in `mkdocs.yml`:
   ```yaml
   - "05 · My New Feature": "journal/05-my-new-feature.md"
   ```
4. Rebuild the site:
   ```bash
   uv run mkdocs build
   ```

---

## 📋 How to Propose or Document an ADR

1. Copy `dev_docs/decisions/adr-template.md`:
   ```bash
   cp dev_docs/decisions/adr-template.md dev_docs/decisions/ADR-005-my-decision.md
   ```
2. Document the context, considered options, pros/cons, and final decision.
3. Add the ADR to the `nav` section in `mkdocs.yml` and table in `dev_docs/decisions/index.md`.
4. Rebuild with `uv run mkdocs build`.
