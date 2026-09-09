function Get-ValidFilePath($prompt) {
    while ($true) {
        $path = Read-Host $prompt
		
		if ([string]::IsNullOrWhiteSpace($path)) {
            Write-Host "No input provided. Please enter a valid file path." -ForegroundColor Yellow
            continue
        }

		
        if (Test-Path $path -PathType Leaf) {
            return $path
        }
		
        Write-Host "File not found. Please try again." -ForegroundColor Yellow
    }
}

function Get-ValidDirectoryPath($prompt) {
    while ($true) {
        $path = Read-Host $prompt
		
		if ([string]::IsNullOrWhiteSpace($path)) {
            Write-Host "No input provided. Please enter a valid file path." -ForegroundColor Yellow
            continue
        }
		
        if (Test-Path $path -PathType Container) {
            return $path
        }
		
        Write-Host "Directory not found. Please try again." -ForegroundColor Yellow
    }
}

$toolPath = Get-ValidFilePath "Enter full path to VectraDBTool_v1.8.exe"
$exportDir = Get-ValidDirectoryPath "Enter export directory"
$idFile = Get-ValidFilePath "Enter full path to interim ID list file (e.g. C:\path\ids.txt)"
$dbName = Read-Host "Enter database name"

# Read IDs from file and join them with commas
$exportIDs = (Get-Content $idFile | Where-Object { $_ -ne "" }) -join ","

# Build and run the command
$command = "& `"$toolPath`" -silentexport -exportdir `"$exportDir`" -exportids `"$exportIDs`" -cleanse -exportimages -dbname `"$dbName`""
Write-Host "Running command..."
Invoke-Expression $command