param(
    [Parameter(Mandatory = $true)] [string]$DatasetRoot,
    [string]$DataDir = "$(Split-Path -Parent $PSScriptRoot)\DATA"
)

$ErrorActionPreference = "Stop"
$mapping = @{
    "imagenet-1k" = "imagenet-1k"
    "caltech-101" = "caltech-101"
    "oxford_pets" = "oxford_pets"
    "stanford_cars" = "stanford_cars"
    "flowers-102" = "flowers-102"
    "food101" = "food101"
    "fgvc_aircraft" = "fgvc_aircraft"
    "sun-397" = "sun-397"
    "dtd" = "dtd"
    "eurosat" = "eurosat"
    "ucf101" = "ucf101"
}

New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
foreach ($name in $mapping.Keys) {
    $source = Join-Path $DatasetRoot $mapping[$name]
    $target = Join-Path $DataDir $name
    if (-not (Test-Path -LiteralPath $source)) {
        Write-Warning "Missing source (not linked): $source"
        continue
    }
    if (Test-Path -LiteralPath $target) {
        Write-Host "Exists, leaving unchanged: $target"
        continue
    }
    New-Item -ItemType Junction -Path $target -Target $source | Out-Null
    Write-Host "Linked $target -> $source"
}
