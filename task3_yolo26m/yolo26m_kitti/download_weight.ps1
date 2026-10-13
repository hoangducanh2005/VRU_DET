$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$TargetDirectory = Join-Path $RepoRoot "session\yolo26m-kitti"
$Target = Join-Path $TargetDirectory "best.pt"
$PartDirectory = Join-Path $TargetDirectory "parts"
$TotalBytes = 44052313
$PartCount = 8
$Uri = "https://huggingface.co/dronefreak/kitti-yolo26m/resolve/main/best.pt?download=true"
New-Item -ItemType Directory -Force -Path $TargetDirectory,$PartDirectory | Out-Null
$headers = curl.exe -sSIL $Uri
$line = @($headers | Select-String '^location:' | Select-Object -Last 1)[0].ToString()
$downloadUri = $line.Substring($line.IndexOf(':') + 1).Trim()
$partSize = [math]::Ceiling($TotalBytes / $PartCount)
for ($i=0; $i -lt $PartCount; $i++) {
  $start=[int64]($i*$partSize); $end=[math]::Min($TotalBytes-1,[int64](($i+1)*$partSize-1)); $part=Join-Path $PartDirectory ("part_{0:D2}.bin" -f $i); $expected=[int64]($end-$start+1)
  if ((Test-Path -LiteralPath $part) -and (Get-Item -LiteralPath $part).Length -eq $expected) { continue }
  & curl.exe -L --fail --http1.1 --retry 5 --retry-delay 2 --range ("{0}-{1}" -f $start,$end) -o $part $downloadUri
  if ($LASTEXITCODE -ne 0) { throw "Failed downloading byte range $start-$end" }
}
$out=[System.IO.File]::Open($Target,[System.IO.FileMode]::Create,[System.IO.FileAccess]::Write)
try { for($i=0;$i -lt $PartCount;$i++){ $part=Join-Path $PartDirectory ("part_{0:D2}.bin" -f $i); $bytes=[System.IO.File]::ReadAllBytes($part); $out.Write($bytes,0,$bytes.Length) } } finally { $out.Dispose() }
if ((Get-Item -LiteralPath $Target).Length -ne $TotalBytes) { throw "Unexpected checkpoint size" }
Get-Item -LiteralPath $Target | Select-Object FullName,Length
