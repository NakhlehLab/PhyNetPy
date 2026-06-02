# PhyNetPy Project Site

This folder contains the static GitHub Pages project site for PhyNetPy.

## Publishing on GitHub Pages

In the GitHub repository settings:

1. Open **Settings > Pages**.
2. Set **Source** to **Deploy from a branch**.
3. Select the branch that contains this folder.
4. Select `/docs` as the publishing folder.
5. Save the settings.

GitHub Pages will publish the site at the repository project-site URL, such as:

```text
https://<organization-or-user>.github.io/PhyNetPy/
```

## Site Structure

- `index.html` is the landing page.
- `news.html` is for project updates and announcements.
- `releases.html` summarizes release notes and links to repository changelogs.
- `documentation.html` points users to generated API documentation in `src/docs`.
- `advertisements.html` is a community announcement board.
- `demos.html` is for tutorials, notebooks, scripts, and example workflows.
- `assets/css/site.css` contains shared styling.

## Maintenance Notes

The generated API reference currently lives in `src/docs`. Keep this site folder
focused on public-facing project content, and link into generated API pages when
the API reference is regenerated.
