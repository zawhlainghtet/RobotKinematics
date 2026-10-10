# Robot Arm Kinematics Web Simulator

An interactive browser-based robotics simulator for studying robot arm kinematics.

## Current Status

- Live custom domain: `https://robotkinematics.study/`
- GitHub Pages style URL: `https://zawhlainghtet.github.io/RobotKinematics/`
- App type: static HTML/CSS/JavaScript
- Server/data required: none
- Build command: none
- Publish/output folder: repository root
- Final entry file: `index.html`

## Files To Host

Host these root files:

- `index.html`
- `app.js`
- `styles.css`
- `favicon.svg`
- `robots.txt`
- `sitemap.xml`
- `googlea367074742ab2e86.html` when Google Search Console verification is still needed

Optional project files:

- `README.md`
- `LICENSE`

Do not publish development-only media unless needed:

- `recording/`
- screenshot/output folders

## Features

- 2D forward kinematics and inverse kinematics
- Configurable 2-to-5 link planar robot arms
- DH parameter based 3D robot arm simulation
- 3D position inverse kinematics
- Up to 7-DOF robot arm experiments
- Jacobian-based singularity analysis and warnings
- 2D workspace visualization with live workspace-use feedback
- IK trajectory preview from a start point to the target
- Circular obstacle collision detection for active planar links
- Live planar transform matrix for the end-effector pose
- End-effector position, transform matrix, error values, and report exports

## Deploy

Use any static host:

- GitHub Pages
- Cloudflare Pages
- Netlify
- Nginx static root on a VPS

See `DEPLOY.md` for exact hosting notes.

## Release Note

Public deploys to `robotkinematics.study` should happen only after Sir Atee previews and approves the version.
