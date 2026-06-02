<#
.SYNOPSIS
    Build and launch GeneNetWeaver (GNW) 3.0 from the source distribution under
    tools/gnw-3.0-src.

.DESCRIPTION
    The downloaded GNW 3.0 is the SOURCE distribution (tools/gnw-3.0-src) with no
    prebuilt jar, so this wrapper compiles it (JDK required) and runs it from the
    compiled classes + lib/*.jar.

    Entry points:
      CLI : ch.epfl.lis.gnw.GnwMain      (default)
      GUI : ch.epfl.lis.gnwgui.Main      (-Gui)

    SAFETY: running GnwMain with NO arguments starts DREAM5 benchmark generation
    (a large simulation). This wrapper therefore defaults to '--help' when no CLI
    arguments are supplied. Pass real arguments (e.g. -s settings.txt) explicitly.

.EXAMPLE
    .\tools\run_gnw.ps1 -Build              # compile from source
    .\tools\run_gnw.ps1                      # safe smoke test (prints CLI usage)
    .\tools\run_gnw.ps1 --settings my.txt    # run a benchmark from a settings file
    .\tools\run_gnw.ps1 -Gui                 # launch the GUI (opens a window)

.NOTES
    See docs/gnw_setup.md for the JDK 17 source patches and details.
#>
[CmdletBinding()]
param(
    [switch]$Gui,
    [switch]$Build,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$GnwArgs
)

$ErrorActionPreference = 'Stop'

$GnwRoot = Join-Path $PSScriptRoot 'gnw-3.0-src'
$Classes = Join-Path $GnwRoot 'build\classes'
$Lib     = Join-Path $GnwRoot 'lib'
$CliMain = 'ch.epfl.lis.gnw.GnwMain'
$GuiMain = 'ch.epfl.lis.gnwgui.Main'

if (-not (Test-Path $GnwRoot)) {
    throw "GNW source not found at $GnwRoot. Place the GNW 3.0 source distribution there."
}

function Resolve-JavaDir {
    # Prefer JAVA_HOME, then PATH, then a scan of common JDK install roots.
    if ($env:JAVA_HOME -and (Test-Path (Join-Path $env:JAVA_HOME 'bin\java.exe'))) {
        return (Join-Path $env:JAVA_HOME 'bin')
    }
    $onPath = Get-Command java.exe -ErrorAction SilentlyContinue
    if ($onPath) { return (Split-Path $onPath.Source) }

    $roots = @(
        "$env:ProgramFiles\Java", "${env:ProgramFiles(x86)}\Java",
        "$env:ProgramFiles\Eclipse Adoptium", "$env:ProgramFiles\Microsoft",
        "$env:ProgramFiles\Zulu", "$env:ProgramFiles\Amazon Corretto",
        "${env:ProgramFiles(x86)}\Android\openjdk", "$env:LOCALAPPDATA\Programs\Eclipse Adoptium"
    )
    foreach ($r in $roots) {
        if (Test-Path $r) {
            $found = Get-ChildItem $r -Recurse -Filter java.exe -Depth 4 -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($found) { return (Split-Path $found.FullName) }
        }
    }
    throw "No Java runtime found. Install a JDK (>=8) or set JAVA_HOME. See docs/gnw_setup.md."
}

$javaDir = Resolve-JavaDir
$javaExe = Join-Path $javaDir 'java.exe'
$javacExe = Join-Path $javaDir 'javac.exe'
Write-Host "Using Java: $javaExe"

if ($Build) {
    if (-not (Test-Path $javacExe)) {
        throw "javac not found in $javaDir. Building GNW needs a full JDK (a JRE cannot compile). See docs/gnw_setup.md."
    }
    # Apply the tracked, idempotent JDK 17 source-compat patches before compiling.
    $patchScript = Join-Path $PSScriptRoot 'gnw_patches\apply_jdk17_patches.ps1'
    if (Test-Path $patchScript) {
        Write-Host "Applying tracked JDK 17 compatibility patches ..."
        & $patchScript -GnwRoot $GnwRoot
    } else {
        Write-Warning "Patch script not found ($patchScript); build will fail on JDK >= 9 without it."
    }
    Write-Host "Compiling GNW source with $javacExe (--release 8) ..."
    New-Item -ItemType Directory -Force $Classes | Out-Null
    $srcs = Get-ChildItem -Recurse (Join-Path $GnwRoot 'src') -Filter *.java | ForEach-Object { $_.FullName }
    $argfile = Join-Path $GnwRoot 'build\sources.txt'
    Set-Content -Path $argfile -Value $srcs -Encoding ASCII
    # Run javac via Start-Process with OS-level stderr/stdout redirection so benign
    # notes (e.g. "uses unchecked or unsafe operations") never surface as failures;
    # real failures are detected via the process exit code.
    $compileLog = Join-Path $GnwRoot 'build\compile.log'
    $compileOut = Join-Path $GnwRoot 'build\compile.out'
    $argList = @('-encoding', 'UTF-8', '--release', '8', '-nowarn', '-d', $Classes, '-cp', (Join-Path $Lib '*'), "@$argfile")
    $proc = Start-Process -FilePath $javacExe -ArgumentList $argList -NoNewWindow -Wait -PassThru `
        -RedirectStandardError $compileLog -RedirectStandardOutput $compileOut
    if ($proc.ExitCode -ne 0) {
        Write-Host (Get-Content $compileLog -Raw)
        throw "javac failed (exit $($proc.ExitCode)). On JDK >= 9 two GUI files need the documented patches; see docs/gnw_setup.md."
    }
    # javac does not copy non-source resources (e.g. gnwguiLogSettings.txt); copy them in.
    robocopy (Join-Path $GnwRoot 'src') $Classes /E /XF *.java /XD .svn /NFL /NDL /NJH /NJS /NP | Out-Null
    Write-Host "Build complete: $Classes"
}

if (-not (Test-Path $Classes)) {
    throw "GNW is not built yet. Run:  .\tools\run_gnw.ps1 -Build"
}

$classpath = "$Classes;$(Join-Path $Lib '*')"

if ($Gui) {
    Write-Host "Launching GNW GUI ($GuiMain). Close the window to exit."
    & $javaExe -cp $classpath $GuiMain @GnwArgs
    exit $LASTEXITCODE
}

if (-not $GnwArgs -or $GnwArgs.Count -eq 0) {
    Write-Host "No CLI arguments given -> showing usage."
    Write-Host "(Bare GnwMain with no args would start DREAM5 benchmark generation; pass -s <settings> to run a benchmark.)"
    $GnwArgs = @('--help')
}

Write-Host "Running: java -cp <build\classes;lib\*> $CliMain $($GnwArgs -join ' ')"
& $javaExe -cp $classpath $CliMain @GnwArgs
exit $LASTEXITCODE
