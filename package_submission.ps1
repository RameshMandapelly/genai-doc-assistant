# Run this from your project root in PowerShell:
#   .\package_submission.ps1
#
# Produces genai-doc-assistant-submission.zip containing source code, README,
# requirements, Dockerfile, and the sample_outputs folder - excluding venv,
# .env, chroma_db, data, __pycache__, and app.log.

$zipName = "genai-doc-assistant-submission.zip"
$staging = "submission_staging"

if (Test-Path $staging) { Remove-Item -Recurse -Force $staging }
New-Item -ItemType Directory -Path $staging | Out-Null

$excludeDirs = @("venv", "chroma_db", "data", "__pycache__", $staging, ".git")
$excludeFiles = @(".env", "app.log", $zipName)

Get-ChildItem -Path . -Recurse -Force | Where-Object {
    $relative = $_.FullName.Substring((Get-Location).Path.Length + 1)
    $topLevel = $relative.Split('\')[0]
    -not ($excludeDirs -contains $topLevel) -and
    -not ($excludeFiles -contains $_.Name) -and
    -not ($relative -like "*\__pycache__\*") -and
    -not ($_.Name -eq "$staging")
} | ForEach-Object {
    $target = Join-Path $staging $_.FullName.Substring((Get-Location).Path.Length + 1)
    if ($_.PSIsContainer) {
        New-Item -ItemType Directory -Path $target -Force | Out-Null
    } else {
        New-Item -ItemType Directory -Path (Split-Path $target) -Force | Out-Null
        Copy-Item $_.FullName -Destination $target -Force
    }
}

Compress-Archive -Path "$staging\*" -DestinationPath $zipName -Force
Remove-Item -Recurse -Force $staging

Write-Host "Created $zipName"
