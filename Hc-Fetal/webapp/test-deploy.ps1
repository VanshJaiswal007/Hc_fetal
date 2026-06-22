# Simple test script to verify deploy.ps1 syntax
Write-Host "Testing deploy.ps1 syntax..." -ForegroundColor Cyan

try {
    # Parse the script to check for syntax errors
    $scriptContent = Get-Content "deploy.ps1" -Raw
    $errors = $null
    $tokens = $null
    $null = [System.Management.Automation.PSParser]::Tokenize($scriptContent, [ref]$errors)
    
    if ($errors.Count -eq 0) {
        Write-Host "[OK] deploy.ps1 syntax is valid!" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] Syntax errors found:" -ForegroundColor Red
        $errors | ForEach-Object { Write-Host $_.Message -ForegroundColor Red }
    }
} catch {
    Write-Host "[ERROR] Failed to parse script: $_" -ForegroundColor Red
}
