# ClamAV runtime and database setup (Windows x64)

The Windows x64 ClamAV 1.5.4 runtime is included in the repository so users do
not need to install the scanner separately. Keep its original DLL layout beside
`clamscan.exe` and `freshclam.exe`. Do not commit the signature database,
generated local configuration or runtime logs.

The signature database is not present in a fresh checkout. From PowerShell at
the repository root, create a local database and updater configuration, then
fetch signatures:

```powershell
$ClamHome = (Resolve-Path "tools/clamav/1.5.4/windows-x64").Path
$Database = Join-Path $ClamHome "database"
$Config = Join-Path $ClamHome "freshclam.local.conf"
New-Item -ItemType Directory -Force $Database | Out-Null
@"
DatabaseDirectory "$Database"
CVDCertsDirectory "$(Join-Path $ClamHome 'certs')"
DatabaseMirror database.clamav.net
"@ | Set-Content -LiteralPath $Config -Encoding ascii
& (Join-Path $ClamHome "freshclam.exe") "--config-file=$Config"
```

Set these local `.env` values (the paths below match this checkout):

```ini
CLAMAV_BINARY=D:/garph_rag/tools/clamav/1.5.4/windows-x64/clamscan.exe
CLAMAV_DATABASE_DIRECTORY=D:/garph_rag/tools/clamav/1.5.4/windows-x64/database
```

Verify the scanner manually:

```powershell
& "$ClamHome\clamscan.exe" `
  "--database=$Database" `
  "D:\path\to\sample.pdf"
```

ClamAV returns `0` for clean, `1` for malware detected and `2` for a scan
error. The application treats scanner errors or missing signatures as a block;
it does not send those PDFs to parsing.

This repository's application environment is Windows PowerShell, not WSL2. If
you run the API in WSL2 or Linux, install a native ClamAV scanner there and set
the same two variables to Linux paths. Do not point a Linux process at Windows
executables.
