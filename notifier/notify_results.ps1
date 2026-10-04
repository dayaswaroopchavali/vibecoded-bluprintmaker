<#
  notify_results.ps1 - local file-change notifier (Windows PowerShell 5.1).
  Watches a directory for final result JSON files and shows a Windows notification
  to the USER. No AI calls, no network, no model polling. Message content is data:
  only the "id", "status" and "result" fields are read, as strings.
#>
param(
    [string]$WatchDir = (Join-Path $PSScriptRoot '..\a2a\from_claude'),
    [string]$LogFile  = (Join-Path $PSScriptRoot 'notifier.log'),
    [string]$SinkFile = '',        # if set: write notices to this file instead of showing a toast
    [switch]$DryRun,               # same as SinkFile but defaults the sink to <LogFile>.dryrun
    [int]$RetryCount = 4,          # read/parse attempts for a mid-write file
    [int]$RetryDelayMs = 300,
    [int]$RescanSeconds = 60,      # cheap local directory rescan safety net (0 = off)
    [int]$MaxRunSeconds = 0,       # 0 = run until stopped
    [string]$MutexName = ''        # default derived from WatchDir
)

$ErrorActionPreference = 'Stop'
$NotifyStatuses = @('completed', 'needs_review', 'blocked', 'failed')
$WatchDir = [System.IO.Path]::GetFullPath($WatchDir)
if ($DryRun -and -not $SinkFile) { $SinkFile = "$LogFile.dryrun" }

function Write-Log([string]$msg) {
    $line = (Get-Date).ToString('yyyy-MM-ddTHH:mm:ss.fff') + "`t" + $msg
    try { [System.IO.File]::AppendAllText($LogFile, $line + "`r`n") } catch { }
}

function Get-Clean([string]$s, [int]$max) {
    if ($null -eq $s) { return '' }
    $s = ($s -replace '[\x00-\x1F\x7F]+', ' ' -replace '\s+', ' ').Trim()
    if ($s.Length -gt $max) { $s = $s.Substring(0, $max - 3) + '...' }
    return $s
}

# ---- single instance guard ----
if (-not $MutexName) {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    $h = [BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($WatchDir.ToLowerInvariant()))).Replace('-', '').Substring(0, 16)
    $MutexName = "Local\ClaudeResultNotifier_$h"
}
$created = $false
$mutex = New-Object System.Threading.Mutex($true, $MutexName, [ref]$created)
if (-not $created) {
    Write-Log "second instance pid=$PID exiting: mutex $MutexName already held"
    exit 3
}

# ---- notification sinks ----
$script:balloon = $null
$script:sinkKind = 'toast'
function Send-Toast([string]$title, [string]$body) {
    [void][Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime]
    [void][Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime]
    $t = [System.Security.SecurityElement]::Escape($title)
    $b = [System.Security.SecurityElement]::Escape($body)
    $xml = New-Object Windows.Data.Xml.Dom.XmlDocument
    $xml.LoadXml("<toast><visual><binding template=`"ToastGeneric`"><text>$t</text><text>$b</text></binding></visual></toast>")
    $appId = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe'
    $toast = New-Object Windows.UI.Notifications.ToastNotification($xml)
    [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($appId).Show($toast)
}
function Send-Balloon([string]$title, [string]$body) {
    Add-Type -AssemblyName System.Windows.Forms, System.Drawing
    if (-not $script:balloon) {
        $script:balloon = New-Object System.Windows.Forms.NotifyIcon
        $script:balloon.Icon = [System.Drawing.SystemIcons]::Information
        $script:balloon.Visible = $true
    }
    $script:balloon.ShowBalloonTip(10000, $title, $body, [System.Windows.Forms.ToolTipIcon]::Info)
}
function Send-Notice([string]$title, [string]$body) {
    if ($SinkFile) {
        $rec = [pscustomobject]@{ title = $title; body = $body } | ConvertTo-Json -Compress
        [System.IO.File]::AppendAllText($SinkFile, $rec + "`r`n")
        return 'sinkfile'
    }
    try { Send-Toast $title $body; return 'toast' } catch { Write-Log "toast failed: $($_.Exception.Message)" }
    try { Send-Balloon $title $body; return 'balloon' } catch { Write-Log "balloon failed: $($_.Exception.Message)" }
    return 'log-only'
}

# ---- state ----
$known = @{}   # full path (lowercase) -> last seen content hash

function Read-Message([string]$path) {
    # returns @{ ok; hash; obj; err }
    $err = ''
    for ($i = 1; $i -le $RetryCount; $i++) {
        try {
            $fs = [System.IO.File]::Open($path, 'Open', 'Read', 'ReadWrite, Delete')
            try {
                $ms = New-Object System.IO.MemoryStream
                $fs.CopyTo($ms)
                $bytes = $ms.ToArray()
            } finally { $fs.Dispose() }
            $hash = [BitConverter]::ToString([System.Security.Cryptography.SHA256]::Create().ComputeHash($bytes)).Replace('-', '').ToLowerInvariant()
            $text = [Text.Encoding]::UTF8.GetString($bytes).TrimStart([char]0xFEFF)
            $obj = $text | ConvertFrom-Json
            if ($null -eq $obj) { throw 'empty json' }
            return @{ ok = $true; hash = $hash; obj = $obj }
        } catch {
            $err = $_.Exception.Message
            if ($i -lt $RetryCount) { Start-Sleep -Milliseconds $RetryDelayMs }
        }
    }
    return @{ ok = $false; err = $err }
}

function Process-File([string]$path, [bool]$baseline) {
    $name = [System.IO.Path]::GetFileName($path)
    if ($name -notlike '*.json' -or $name -like 'claim-*' -or $name -like '*.tmp') { return }
    if (-not (Test-Path -LiteralPath $path)) { return }
    $key = $path.ToLowerInvariant()
    if ($baseline) {
        try {
            $bytes = [System.IO.File]::ReadAllBytes($path)
            $known[$key] = [BitConverter]::ToString([System.Security.Cryptography.SHA256]::Create().ComputeHash($bytes)).Replace('-', '').ToLowerInvariant()
            return 1
        } catch { return }
    }
    $r = Read-Message $path
    if (-not $r.ok) { Write-Log "gave up reading $name after $RetryCount attempts (partial/invalid?): $($r.err)"; return }
    if ($known[$key] -eq $r.hash) { return }   # unchanged / duplicate content
    $known[$key] = $r.hash
    # data only: read three fields as strings
    $status = if ($r.obj.PSObject.Properties['status']) { [string]$r.obj.status } else { '' }
    $id     = if ($r.obj.PSObject.Properties['id'])     { [string]$r.obj.id }     else { '' }
    $result = if ($r.obj.PSObject.Properties['result']) { [string]$r.obj.result } else { '' }
    $status = Get-Clean $status 40
    if ($NotifyStatuses -notcontains $status.ToLowerInvariant()) {
        Write-Log "no-notify`t$name`t$status`t$($r.hash)"
        return
    }
    if (-not $id) { $id = ($name -replace '^(\d+).*', '$1') }
    $title = "Claude result $(Get-Clean $id 20): $status"
    $body = Get-Clean $result 200
    $sink = Send-Notice $title $body
    Write-Log "NOTIFIED($sink)`t$name`t$status`t$($r.hash)"
}

# ---- start ----
if (-not (Test-Path -LiteralPath $WatchDir)) { Write-Log "watch dir missing: $WatchDir"; exit 4 }
$n = 0
foreach ($f in Get-ChildItem -LiteralPath $WatchDir -Filter '*.json' -File) {
    if ((Process-File $f.FullName $true) -eq 1) { $n++ }
}
$fsw = New-Object System.IO.FileSystemWatcher($WatchDir, '*.json')
$fsw.NotifyFilter = [System.IO.NotifyFilters]'FileName, LastWrite, Size'
$fsw.EnableRaisingEvents = $true
$sid = "NotifierFsw_$PID"
foreach ($ev in 'Created', 'Changed', 'Renamed') { [void](Register-ObjectEvent -InputObject $fsw -EventName $ev -SourceIdentifier "$sid.$ev") }
$mode = if ($SinkFile) { "sinkfile=$SinkFile" } else { 'toast/balloon' }
Write-Log "started pid=$PID watch=$WatchDir baselined $n existing messages; sink=$mode"

$deadline = if ($MaxRunSeconds -gt 0) { (Get-Date).AddSeconds($MaxRunSeconds) } else { [datetime]::MaxValue }
$nextScan = (Get-Date).AddSeconds($(if ($RescanSeconds -gt 0) { $RescanSeconds } else { 86400 * 365 }))
try {
    while ((Get-Date) -lt $deadline) {
        $first = Wait-Event -Timeout 1
        if (-not $first -and (Get-Date) -lt $nextScan) { continue }
        $paths = [ordered]@{}
        if ($first) {
            Start-Sleep -Milliseconds 150   # let a burst of events settle
            foreach ($e in @(Get-Event)) {
                if ($e.SourceIdentifier -like "$sid.*") { $paths[$e.SourceEventArgs.FullPath] = 1 }
                Remove-Event -EventIdentifier $e.EventIdentifier -ErrorAction SilentlyContinue
            }
        }
        if ((Get-Date) -ge $nextScan) {
            foreach ($f in Get-ChildItem -LiteralPath $WatchDir -Filter '*.json' -File -ErrorAction SilentlyContinue) { $paths[$f.FullName] = 1 }
            $nextScan = (Get-Date).AddSeconds($RescanSeconds)
        }
        foreach ($p in @($paths.Keys)) {
            try { Process-File $p $false | Out-Null } catch { Write-Log "error processing $p : $($_.Exception.Message)" }
        }
        if ($script:balloon) { [System.Windows.Forms.Application]::DoEvents() }
    }
} finally {
    foreach ($ev in 'Created', 'Changed', 'Renamed') { Unregister-Event -SourceIdentifier "$sid.$ev" -ErrorAction SilentlyContinue }
    $fsw.Dispose()
    if ($script:balloon) { $script:balloon.Visible = $false; $script:balloon.Dispose() }
    Write-Log "stopped pid=$PID"
    $mutex.ReleaseMutex(); $mutex.Dispose()
}
