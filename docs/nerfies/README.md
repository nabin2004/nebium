# Nebium Research Project Page

This directory contains the academic project website for **Nebium: A Causal Transformer for Self-Supervised Next-Move Prediction in Chess**. Built on the modern Nerfies / academic paper template.

## Preview Locally

You can open `index.html` directly in any web browser, or serve it locally with Python:

```bash
cd docs/nerfies
python -m http.server 8000
```
Then visit `http://localhost:8000`.

## Deploy to GitHub Pages

1. In your GitHub repository settings, navigate to **Pages**.
2. Select **Deploy from a branch**.
3. Choose your branch (e.g. `master` or `gh-pages`) and set the folder to `/docs/nerfies` (or move files to `/docs` for standard root hosting).
4. Save to deploy.

## Structure

```
docs/nerfies/
├── index.html              # Main single-page academic project site
├── static/
│   ├── css/style.css       # Clean, modern typography & design system
│   ├── js/main.js          # Interactive modals, clipboard & animations
│   └── images/
│       ├── teaser.svg      # Teaser diagram: Causal attention & candidate moves
│       ├── pipeline.svg    # System pipeline & architecture overview
│       └── qualitative.svg # Qualitative progression across training epochs
```
