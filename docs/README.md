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
- `documentation.html` points users to the generated API reference in `docs/api`.
- `advertisements.html` is a community announcement board.
- `demos.html` is for tutorials, notebooks, scripts, and example workflows.
- `assets/css/site.css` contains shared styling.

## Maintenance Notes

The API reference in `docs/api/` is generated -- do not edit those files by hand.
Regenerate them after changing any public docstring:

```bash
python generate_docs.py
```

The generator reads every non-underscore module in `src/`, plus each
subpackage. Four of those modules are facades whose surface is re-exported from
private implementation modules -- `infer`, `data`, `models`, and `criteria` --
so the generator folds each one's `__all__` onto its own page. That means it
has to *import* `phynetpy`, so run it with the interpreter the package is
installed into. Pages for modules that no longer exist are deleted.

Keep the hand-written pages in this folder focused on public-facing project
content and link into `api/` from there.
