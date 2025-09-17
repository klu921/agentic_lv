# Long-Context Video Understanding Blog

This is the GitHub Pages blog for our research on long-context video understanding.

## Setup Instructions

### To deploy to GitHub Pages:

1. Push this repository to GitHub
2. Go to Settings → Pages in your GitHub repository
3. Under "Source", select "Deploy from a branch"
4. Choose the branch (main/master) and select `/docs` folder
5. Click Save

Your blog will be available at: `https://[your-username].github.io/[repository-name]/`

### To test locally:

1. Install Ruby and Jekyll:
```bash
gem install bundler jekyll
```

2. Navigate to the docs folder:
```bash
cd docs
```

3. Install dependencies:
```bash
bundle install
```

4. Run the local server:
```bash
bundle exec jekyll serve
```

5. Open http://localhost:4000 in your browser

## Structure

- `_posts/`: Blog posts in Markdown format
- `_config.yml`: Jekyll configuration
- `index.md`: Home page
- `about.md`: About page
- `assets/css/`: Custom styles
- `Gemfile`: Ruby dependencies

## Adding New Posts

Create new posts in `_posts/` with the naming format: `YYYY-MM-DD-title.md`

Include the front matter:
```yaml
---
layout: post
title: "Your Post Title"
date: YYYY-MM-DD
categories: research
---
```