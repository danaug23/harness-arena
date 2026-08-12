<#
.SYNOPSIS
  Convenience wrapper for harness-arena on Windows.

.DESCRIPTION
  Forwards everything to the harness-arena CLI. This exists only so the repo
  works before `pip install -e .`; once installed, `harness-arena <command>` is
  the same thing and works on every platform.

  Python is located in this order:
    1. $env:HARNESS_ARENA_PYTHON, if set
    2. an active virtualenv / conda env ($env:VIRTUAL_ENV, $env:CONDA_PREFIX)
    3. python / python3 on PATH

.EXAMPLE
  .\run.ps1 init
  Point the rig at your model server.

.EXAMPLE
  .\run.ps1 doctor
  Check Docker, Harbor and the endpoint before spending hours on a run.

.EXAMPLE
  .\run.ps1 bench -Subset stratified-25
  25 tasks, every harness, sequentially.

.EXAMPLE
  .\run.ps1 dash
  Live dashboard -- safe to leave open during a run.
#>
[CmdletBinding()]
param(
  [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
  [string[]]$Args
)

$ErrorActionPreference = 'Stop'

function Find-Python {
  if ($env:HARNESS_ARENA_PYTHON) {
    if (Test-Path $env:HARNESS_ARENA_PYTHON) { return $env:HARNESS_ARENA_PYTHON }
    throw "HARNESS_ARENA_PYTHON is set to '$($env:HARNESS_ARENA_PYTHON)' but that path does not exist."
  }

  foreach ($prefix in @($env:VIRTUAL_ENV, $env:CONDA_PREFIX)) {
    if (-not $prefix) { continue }
    foreach ($candidate in @(
        (Join-Path $prefix 'python.exe'),
        (Join-Path $prefix 'Scripts\python.exe'),
        (Join-Path $prefix 'bin/python')
      )) {
      if (Test-Path $candidate) { return $candidate }
    }
  }

  foreach ($name in @('python', 'python3')) {
    $found = Get-Command $name -ErrorAction SilentlyContinue
    if ($found) { return $found.Source }
  }

  throw @"
No Python found.

Create the environment first, then re-run:

  conda env create -f environment.yml
  conda activate harness-arena

or with a plain virtualenv:

  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -e .

or point this script at an interpreter directly:

  `$env:HARNESS_ARENA_PYTHON = 'C:\path\to\python.exe'
"@
}

$python = Find-Python
$root = $PSScriptRoot

Push-Location $root
try {
  # PYTHONPATH so `python -m bench` works from a bare checkout, with no install.
  $env:PYTHONPATH = if ($env:PYTHONPATH) { "$root;$($env:PYTHONPATH)" } else { $root }
  & $python -m bench @Args
  exit $LASTEXITCODE
} finally {
  Pop-Location
}
