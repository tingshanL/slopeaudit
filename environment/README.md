# Reproduction environments

## Python

The reference Python interpreter is recorded in
`python-version.txt`. Direct third-party dependencies used by the public
reproduction scripts are pinned in `requirements.txt`. Create a clean virtual
environment from the repository root with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r environment/requirements.txt
```

## R

The R environment uses `renv.lock`, `.Rprofile`, and `renv/activate.R`. Restore
the recorded library with:

```bash
Rscript -e 'renv::restore(prompt = FALSE)'
```

`renv-setup.R` is a maintainer utility. It records the direct R-package
versions used in the reference environment,
initializes the project-local renv library when needed, installs those exact
versions, and snapshots them with their recursive dependencies. Most users
should restore `renv.lock` rather than rerun `renv-setup.R`.

### Development dependency scope

`renv/settings.json` sets `snapshot.dev` to `true`. The lockfile includes the
package's development and check dependencies (including packages in `Suggests` and tools such as
`devtools` and `roxygen2`) in addition to runtime dependencies. This setting
makes `renv::status()` evaluate the same development-dependency scope recorded
by `renv-setup.R`; it does not add those tools to the
installed package's runtime dependencies.

## Pandoc

Building the R Markdown package vignettes requires a command-line Pandoc
installation in addition to the R packages recorded by `renv`. The reference
Pandoc version is recorded in `pandoc-version.txt`.

A fresh verification environment must make `pandoc` available on `PATH`;
`rmarkdown::pandoc_available()` must return `TRUE` before `R CMD build`.

## LaTeX and HTML validation

`R CMD check --as-cran` requires a LaTeX distribution to build the PDF
package manual. The reference environment uses TinyTeX.

A current HTML Tidy executable should also be available on `PATH` for
HTML-manual validation.

These are external executables rather than R-package dependencies and
therefore are not represented as packages in `renv.lock`.
