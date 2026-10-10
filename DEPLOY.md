# RobotKinematics Deployment

RobotKinematics is a static site. It does not need a Node server, database, Docker, or backup process like PhoneShopPOS.

## Final Entry File

`index.html` is the final entry file.

There is no separate `final.html` file. Static hosts serve the root `index.html` as the website.

## GitHub Pages

1. Push the root static files to the repository.
2. In GitHub repository settings, enable Pages.
3. Select the `main` branch and root folder.
4. Save and wait for GitHub Pages to publish.

Required root files:

- `index.html`
- `app.js`
- `styles.css`
- `favicon.svg`
- `robots.txt`
- `sitemap.xml`
- `googlea367074742ab2e86.html` when Google verification is needed

## Cloudflare Pages

Settings:

- Framework preset: none
- Build command: none
- Build output directory: `/` or repository root

## Nginx Static VPS

Copy the root files into the web root, for example:

```bash
/var/www/robotkinematics/
```

Then configure Nginx to serve that directory.

## Before Public Deploy

Preview first, then deploy only after approval.
