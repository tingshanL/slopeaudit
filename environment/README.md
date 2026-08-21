# Reproduction environments

## Python

The reference first-release Python interpreter is recorded in
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

The public release uses `renv.lock`, `.Rprofile`, and `renv/activate.R`. A
reviewer restores the frozen library with:

```bash
Rscript -e 'renv::restore(prompt = FALSE)'
```

`renv-setup.R` is a maintainer-only first-release freeze script. It records
the exact direct R-package versions established during release preparation,
initializes the project-local renv library when needed, installs those exact
versions, and snapshots them with their recursive dependencies. Reviewers do
not rerun `renv-setup.R`; they restore `renv.lock`.

### Development dependency scope

`renv/settings.json` sets `snapshot.dev` to `true` for this release project.
The frozen lockfile intentionally includes the package's development and
release-check dependencies (including packages in `Suggests` and tools such as
`devtools` and `roxygen2`) in addition to runtime dependencies. This setting
makes `renv::status()` evaluate the same development-dependency scope that was
deliberately frozen by `renv-setup.R`; it does not add those tools to the
installed package's runtime dependencies.

## Pandoc

Building the R Markdown package vignettes requires a command-line Pandoc
installation in addition to the R packages recorded by `renv`. The exact
release-preparation Pandoc version is recorded in `pandoc-version.txt`.

A fresh verification environment must make `pandoc` available on `PATH`;
`rmarkdown::pandoc_available()` must return `TRUE` before `R CMD build`.

## LaTeX and HTML validation

`R CMD check --as-cran` requires a LaTeX distribution to build the PDF
package manual. Release preparation uses TinyTeX and records the actual
`pdflatex` version in the private verification evidence.

A current HTML Tidy executable is also made available on `PATH` during
release verification so that HTML-manual validation is not skipped
because of local tooling.

These are external executables rather than R-package dependencies and
therefore are not represented as packages in `renv.lock`.
