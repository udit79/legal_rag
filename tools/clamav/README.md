# Bundled ClamAV runtime

This repository includes the ClamAV 1.5.4 Windows x64 portable runtime from
[ClamAV](https://www.clamav.net/downloads). Only `clamscan`, `freshclam`,
their runtime DLLs, certificates, configuration examples, and license files
are retained. Debug symbols, static libraries, headers, manuals, and unused
utilities were removed.

The virus signature database is not committed. Create and update it once from
PowerShell in the repository root:

```powershell
$clam = (Resolve-Path "tools/clamav/1.5.4/windows-x64").Path
$database = Join-Path $clam "database"
$config = Join-Path $clam "freshclam.local.conf"
New-Item -ItemType Directory -Force -Path $database | Out-Null
@"
DatabaseDirectory "$database"
CVDCertsDirectory "$clam/certs"
DatabaseMirror database.clamav.net
"@ | Set-Content -LiteralPath $config -Encoding ascii
& (Join-Path $clam "freshclam.exe") "--config-file=$config"
```

The database directory and local `freshclam` config are ignored by Git. The
API automatically selects the bundled `clamscan.exe` and its sibling database
directory on Windows. Set `CLAMAV_SCAN_COMMAND` or
`CLAMAV_DATABASE_DIRECTORY` only to override those defaults. Updated signatures
are needed for reliable malware scanning; without them, G0 fails closed.

ClamAV is GPLv2; third-party license notices are retained in the adjacent
`COPYING.txt` and `COPYING/` files.
