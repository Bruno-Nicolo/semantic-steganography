$ErrorActionPreference = "Stop"

function Get-EnvOrDefault {
    param(
        [string]$Name,
        [string]$Default
    )

    $Value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($Value)) {
        return $Default
    }
    return $Value
}

function Split-EnvList {
    param([string]$Value)

    return @($Value -split "\s+" | Where-Object { $_ -ne "" })
}

function Test-ReparsePoint {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }
    $Item = Get-Item -LiteralPath $Path -Force
    return [bool]($Item.Attributes -band [IO.FileAttributes]::ReparsePoint)
}

$IsRunningWindows = [Environment]::OSVersion.Platform -eq [PlatformID]::Win32NT
$DefaultPythonBin = if ($IsRunningWindows) { ".venv/Scripts/python.exe" } else { ".venv/bin/python" }

$ShardsRoot = Get-EnvOrDefault "SHARDS_ROOT" "data/coco/val2017_shards"
$PythonBin = Get-EnvOrDefault "PYTHON_BIN" $DefaultPythonBin
$RepetitionFactor = Get-EnvOrDefault "REPETITION_FACTOR" "3"
$OutputRoot = Get-EnvOrDefault "OUTPUT_ROOT" "outputs/stego_sweep_shards"
$CsvExportDir = Join-Path $OutputRoot "csv_exports"
$ShardInputsDir = Join-Path $OutputRoot "shard_inputs"
$PayloadBitsValues = Split-EnvList (Get-EnvOrDefault "PAYLOAD_BITS_VALUES" "8 64 128 512")
$AbsoluteDeltas = Split-EnvList (Get-EnvOrDefault "ABSOLUTE_DELTAS" "0.5 10 20 40 80")
$ProportionalDeltas = Split-EnvList (Get-EnvOrDefault "PROPORTIONAL_DELTAS" "0.05 0.1")

if (-not (Test-Path -LiteralPath $ShardsRoot -PathType Container)) {
    Write-Error "Shard directory not found: $ShardsRoot"
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
New-Item -ItemType Directory -Force -Path $CsvExportDir | Out-Null
New-Item -ItemType Directory -Force -Path $ShardInputsDir | Out-Null

$ShardDirs = @(Get-ChildItem -LiteralPath $ShardsRoot -Directory -Filter "shard_*" | Sort-Object Name)
if ($ShardDirs.Count -eq 0) {
    Write-Error "No shard directories found under: $ShardsRoot"
}

foreach ($ShardDir in $ShardDirs) {
    $ShardName = $ShardDir.Name
    $ShardOutputDir = Join-Path $OutputRoot $ShardName
    $ShardCocoRoot = Join-Path $ShardInputsDir $ShardName
    $ShardSplitDir = Join-Path $ShardCocoRoot "val2017"
    $ImageFiles = @(Get-ChildItem -LiteralPath $ShardDir.FullName -File)
    $ImageCount = $ImageFiles.Count

    if ($ImageCount -eq 0) {
        Write-Warning "Skipping empty shard: $($ShardDir.FullName)"
        continue
    }

    if (Test-Path -LiteralPath $ShardOutputDir) {
        Write-Error "Output directory already exists: $ShardOutputDir"
    }

    if ((Test-Path -LiteralPath $ShardSplitDir) -and -not (Test-ReparsePoint $ShardSplitDir)) {
        Write-Error "Shard split path already exists and is not a symlink/junction: $ShardSplitDir"
    }

    New-Item -ItemType Directory -Force -Path $ShardCocoRoot | Out-Null
    if (-not (Test-ReparsePoint $ShardSplitDir)) {
        try {
            New-Item -ItemType SymbolicLink -Path $ShardSplitDir -Target $ShardDir.FullName | Out-Null
        }
        catch {
            if ($IsRunningWindows) {
                New-Item -ItemType Junction -Path $ShardSplitDir -Target $ShardDir.FullName | Out-Null
            }
            else {
                throw
            }
        }
    }

    Write-Host "Running image-centric sweep on $ShardName ($ImageCount images)"
    & $PythonBin -m semantic_stego.cli.efficient_sweep_app `
        --coco-root $ShardCocoRoot `
        --split val2017 `
        --output-dir $ShardOutputDir `
        --max-images $ImageCount `
        --roi-strategies largest full_image smallest `
        --svd-bands mid_energy low_energy high_energy `
        --decoders non_blind blind `
        --attacks none gaussian_noise gaussian_blur jpeg_compression `
        --noise-sigmas 5 10 20 `
        --blur-kernels 3 5 7 `
        --jpeg-qualities 80 50 30 `
        --payload-bits-values $PayloadBitsValues `
        --absolute-deltas $AbsoluteDeltas `
        --proportional-deltas $ProportionalDeltas `
        --repetition-factor $RepetitionFactor `
        --seed 42 `
        --skip-no-detection

    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    Copy-Item -LiteralPath (Join-Path $ShardOutputDir "results.csv") -Destination (Join-Path $CsvExportDir "${ShardName}_results.csv")
}

Invoke-Item -LiteralPath $CsvExportDir
