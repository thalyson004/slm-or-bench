param(
    [string[]]$Models = @('all'),
    [ValidateSet('all', 'ollama', 'openrouter')]
    [string]$Provider = 'all',
    [ValidateSet('ComplexOR', 'LogiOR', 'IndustryOR')]
    [string[]]$Datasets = @('ComplexOR', 'LogiOR', 'IndustryOR'),
    [ValidateSet('direct', 'code', 'formulation_code')]
    [string[]]$Pipelines = @('direct', 'code', 'formulation_code'),
    [ValidateSet('zero_shot', 'few_shot')]
    [string[]]$ShotConditions = @('zero_shot', 'few_shot'),
    [int]$Repetitions = 1,
    [int]$RepetitionStart = 1,
    [int]$Limit = 1,
    [string]$OutputDir,
    [switch]$Full,
    [switch]$Execute
)

$ErrorActionPreference = 'Stop'
$repository = (Resolve-Path $PSScriptRoot).Path
$oldPythonPath = $env:PYTHONPATH
$sourcePath = Join-Path $repository 'src'
$separator = [IO.Path]::PathSeparator
$env:PYTHONPATH = if ($oldPythonPath) { "$sourcePath$separator$oldPythonPath" } else { $sourcePath }

if (-not $OutputDir) {
    $OutputDir = if ($Full) {
        Join-Path $repository 'results'
    }
    else {
        Join-Path $repository 'results-dev'
    }
}

try {
    $arguments = @(
        '-m', 'or_small_models.cli',
        '--models'
    ) + $Models + @('--provider', $Provider, '--datasets') + $Datasets + @('--pipelines') + $Pipelines + @('--shot-conditions') + $ShotConditions + @(
        '--repetitions', $Repetitions,
        '--repetition-start', $RepetitionStart,
        '--output-dir', $OutputDir
    )
    if ($Full) {
        $arguments += '--full'
    }
    else {
        $arguments += @('--limit', $Limit)
    }
    if (-not $Execute) {
        $arguments += '--dry-run'
    }

    & python @arguments
    $benchmarkExitCode = $LASTEXITCODE
    if ($benchmarkExitCode -eq 130) {
        Write-Host "Benchmark interrupted safely. Run the same command to resume from $OutputDir."
        exit 130
    }
    if ($benchmarkExitCode -eq 75) {
        Write-Warning "OpenRouter execution is waiting for credits. Add credits and run the same command to retry the unfinished observation from $OutputDir."
        exit 75
    }
    if ($benchmarkExitCode -ne 0) {
        throw "Benchmark command failed with exit code $benchmarkExitCode."
    }
}
finally {
    $env:PYTHONPATH = $oldPythonPath
}

if (-not $Execute) {
    Write-Host 'Plan only. Use -Execute to call models and -Full for the complete frozen matrix.'
}
