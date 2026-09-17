# Nebium Research Project Page

This directory contains the academic project website for **Nebium: A Causal Transformer for Self-Supervised Next-Move Prediction in Chess**. Built on the modern Nerfies / academic paper template.

## Preview Locally

You can open `index.html` directly in any web browser, or serve it locally with Python:

```bash
cd docs
python -m http.server 8000
```
Then visit `http://localhost:8000`.

## Deploy to GitHub Pages

1. In your GitHub repository settings, navigate to **Pages**.
2. Select **Deploy from a branch**.
3. Choose your branch (e.g. `master` or `gh-pages`) and set the folder to `/docs`.
4. Save to deploy.

## Structure

```
docs/
├── index.html              # Main single-page academic project site (exact Nerfies template)
├── static/
│   ├── css/
│   │   ├── bulma.min.css         # Bulma CSS framework
│   │   ├── bulma-carousel.min.css # Carousel styling
│   │   ├── bulma-slider.min.css   # Slider styling
│   │   ├── fontawesome.all.min.css# FontAwesome icons
│   │   └── index.css             # Canonical Nerfies index.css
│   ├── js/
│   │   ├── bulma-carousel.min.js # Bulma carousel script
│   │   ├── bulma-slider.min.js   # Bulma slider script
│   │   ├── fontawesome.all.min.js# FontAwesome script
│   │   └── index.js              # Nerfies interactive script
│   └── images/
│       ├── favicon.svg           # Favicon
│       ├── teaser.svg            # Teaser diagram
│       ├── pipeline.svg          # System architecture overview
│       └── qualitative.svg       # Qualitative progression across training epochs
```
