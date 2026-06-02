<#
.SYNOPSIS
    Idempotently apply the JDK 17 source-compatibility patches to the (git-ignored)
    GeneNetWeaver 3.0 source tree.

.DESCRIPTION
    GNW 3.0 (2010) uses two JDK-internal APIs that were removed in Java 9+:
      * com.sun.image.codec.jpeg.{JPEGCodec,JPEGImageEncoder}  (GUI JPEG export)
      * com.sun.org.apache.xerces.internal.* / .xml.internal.serialize.*  (GUI XML export)
    This script rewrites the two affected GUI files to use the standard
    javax.imageio.ImageIO and javax.xml.transform APIs so GNW compiles on a modern
    JDK without touching anything the CLI/simulation code depends on.

    It is whitespace-tolerant and IDEMPOTENT: it matches the original (com.sun-based)
    code, so on an already-patched tree nothing matches and the files are left
    unchanged. This is the tracked, reproducible source of the patch; the GNW source
    tree itself stays git-ignored. tools/run_gnw.ps1 -Build runs this automatically.

.PARAMETER GnwRoot
    Path to the GNW source root (default: ../gnw-3.0-src relative to this script).
#>
[CmdletBinding()]
param(
    [string]$GnwRoot = (Join-Path (Split-Path $PSScriptRoot -Parent) 'gnw-3.0-src')
)

$ErrorActionPreference = 'Stop'

$guiDir = Join-Path $GnwRoot 'src\ch\epfl\lis\gnwgui'
$networkGraph = Join-Path $guiDir 'NetworkGraph.java'
$networkDesktopMap = Join-Path $guiDir 'NetworkDesktopMap.java'

function Invoke-FileReplacements {
    param([string]$Path, [object[]]$Replacements)
    if (-not (Test-Path $Path)) { throw "GNW source file not found: $Path (is tools/gnw-3.0-src present?)" }
    $original = [System.IO.File]::ReadAllText($Path)
    $content = $original
    foreach ($r in $Replacements) {
        $content = [regex]::Replace($content, $r.Pattern, $r.Replacement)
    }
    if ($content -ne $original) {
        [System.IO.File]::WriteAllText($Path, $content)
        return $true
    }
    return $false
}

# --- NetworkGraph.java: JPEG export via com.sun.* -> javax.imageio.ImageIO ---
$graphImportRepl = "// JDK17 compat (auto-patched): com.sun.image.codec.jpeg removed; using javax.imageio.ImageIO.`r`nimport javax.imageio.ImageIO;"
$networkGraphReplacements = @(
    @{ Pattern = '(?m)^[ \t]*import[ \t]+com\.sun\.image\.codec\.jpeg\.JPEGCodec;[ \t]*\r?\n[ \t]*import[ \t]+com\.sun\.image\.codec\.jpeg\.JPEGImageEncoder;[ \t]*'
       Replacement = $graphImportRepl }
    @{ Pattern = '(?m)^([ \t]*)JPEGImageEncoder[ \t]+encoder[ \t]*=[ \t]*JPEGCodec\.createJPEGEncoder\(out\);[ \t]*\r?\n[ \t]*encoder\.encode\(myImage\);'
       Replacement = '${1}ImageIO.write(myImage, "jpeg", out);' }
)

# --- NetworkDesktopMap.java: internal Xerces/serializer -> javax.xml.transform ---
$serializerBlock = @'
${1}javax.xml.transform.Transformer serializer = javax.xml.transform.TransformerFactory.newInstance().newTransformer();
${1}serializer.setOutputProperty(javax.xml.transform.OutputKeys.ENCODING, "ISO-8859-1");
${1}serializer.setOutputProperty(javax.xml.transform.OutputKeys.INDENT, "yes");
${1}serializer.setOutputProperty("{http://xml.apache.org/xslt}indent-amount", "1");
${1}serializer.transform(new javax.xml.transform.dom.DOMSource(xmldoc_), new javax.xml.transform.stream.StreamResult(fos));
'@
$networkDesktopMapReplacements = @(
    @{ Pattern = '(?m)^[ \t]*import[ \t]+com\.sun\.org\.apache\.xerces\.internal\.dom\.DocumentImpl;[ \t]*\r?\n[ \t]*import[ \t]+com\.sun\.org\.apache\.xml\.internal\.serialize\.OutputFormat;[ \t]*\r?\n[ \t]*import[ \t]+com\.sun\.org\.apache\.xml\.internal\.serialize\.XMLSerializer;[ \t]*'
       Replacement = '// JDK17 compat (auto-patched): com.sun.* internal Xerces replaced by javax.xml.transform below.' }
    @{ Pattern = 'new[ \t]+DocumentImpl\(\)'
       Replacement = 'DocumentBuilderFactory.newInstance().newDocumentBuilder().newDocument()' }
    @{ Pattern = '(?ms)^([ \t]*)OutputFormat[ \t]+of[ \t]*=.*?serializer\.serialize\(xmldoc_\.getDocumentElement\(\)\);'
       Replacement = $serializerBlock.TrimEnd("`r","`n") }
)

$g = Invoke-FileReplacements -Path $networkGraph -Replacements $networkGraphReplacements
Write-Host ("NetworkGraph.java    : " + ($(if ($g) { 'patched' } else { 'already patched / no change' })))
$d = Invoke-FileReplacements -Path $networkDesktopMap -Replacements $networkDesktopMapReplacements
Write-Host ("NetworkDesktopMap.java: " + ($(if ($d) { 'patched' } else { 'already patched / no change' })))

# sanity: no removed APIs should remain as actual imports (comments mentioning
# the old class names are fine).
$leftovers = @()
foreach ($f in @($networkGraph, $networkDesktopMap)) {
    if ((Get-Content $f -Raw) -match '(?m)^[ \t]*import[ \t]+com\.sun\.(image\.codec\.jpeg|org\.apache)') { $leftovers += $f }
}
if ($leftovers.Count -gt 0) { throw "Removed com.sun.* import still present after patching: $($leftovers -join ', ')" }
Write-Host "JDK17 compatibility patches OK (no removed com.sun.* APIs remain)."
