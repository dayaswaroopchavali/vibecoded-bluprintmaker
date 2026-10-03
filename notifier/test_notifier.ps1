# Self-test for notify_results.ps1 using a temp dir and a fake sink file. No real toasts.
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$tmp = Join-Path $env:TEMP ("notifier_test_" + [guid]::NewGuid().ToString('N').Substring(0, 8))
$dir = Join-Path $tmp 'from_claude'
New-Item -ItemType Directory -Force $dir | Out-Null
$sink = Join-Path $tmp 'sink.jsonl'
$log = Join-Path $tmp 'notifier.log'
$script:fail = 0

function Put([string]$name, $obj) { ($obj | ConvertTo-Json -Compress) | Set-Content -LiteralPath (Join-Path $dir $name) -Encoding UTF8 }
function Msg($id, $status, $result) { @{ id = $id; status = $status; result = $result } }
function Count { if (Test-Path $sink) { @(Get-Content $sink).Count } else { 0 } }
function Settle { Start-Sleep -Milliseconds 2500 }
function Check([string]$name, [bool]$ok, [string]$detail = '') {
    if ($ok) { "PASS  $name" } else { "FAIL  $name  $detail"; $script:fail++ }
}
function LastNotice { if (Test-Path $sink) { (Get-Content $sink | Select-Object -Last 1) } else { '' } }

# pre-existing messages (baseline)
Put '0001-old-completed.json' (Msg '0001' 'completed' 'old result')
Put '0002-old-review.json' (Msg '0002' 'needs_review' 'old review')
Put 'claim-0003.json' (Msg '0003' 'claimed' 'claim')

$args1 = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$here\notify_results.ps1`"", '-WatchDir', "`"$dir`"",
    '-LogFile', "`"$log`"", '-SinkFile', "`"$sink`"", '-RetryCount', '3', '-RetryDelayMs', '200', '-MaxRunSeconds', '120')
$proc = Start-Process powershell -ArgumentList $args1 -WindowStyle Hidden -PassThru
Start-Sleep -Seconds 4
try {
    # 1
    Check '1 pre-existing messages do not notify' ((Count) -eq 0) "count=$(Count)"
    Check '1b log shows baseline of 2 messages (claim excluded)' ((Get-Content $log -Raw) -match 'baselined 2 existing')

    # 2
    Put 'claim-0010.json' (Msg '0010' 'completed' 'claim should be ignored')
    Settle
    Check '2 claim file does not notify' ((Count) -eq 0) "count=$(Count)"

    # 3
    Put '0011-running.json' (Msg '0011' 'running' 'r'); Put '0012-claimed.json' (Msg '0012' 'claimed' 'c'); Put '0013-ready.json' (Msg '0013' 'ready' 'y')
    Settle
    Check '3 running/claimed/ready do not notify' ((Count) -eq 0) "count=$(Count)"

    # 4 unchanged / duplicate-content rewrite of a baselined file
    Put '0001-old-completed.json' (Msg '0001' 'completed' 'old result')
    Settle
    Check '4 duplicate-content rewrite does not notify' ((Count) -eq 0) "count=$(Count)"

    # 5 new final result, once
    Put '0020-new.json' (Msg '0020' 'completed' ('x' * 300))
    Settle
    $n5 = Count
    Check '5 new final result notifies exactly once' ($n5 -eq 1) "count=$n5"
    $notice = (LastNotice) | ConvertFrom-Json
    Check '5b title/body format, body truncated to <=200' (($notice.title -eq 'Claude result 0020: completed') -and ($notice.body.Length -le 200)) (LastNotice)
    Put '0020-new.json' (Msg '0020' 'completed' ('x' * 300))   # same content again
    Settle
    Check '4b rewrite of just-notified identical content does not notify again' ((Count) -eq 1) "count=$(Count)"

    # 6 changed final result
    Put '0020-new.json' (Msg '0020' 'failed' 'now failed')
    Settle
    Check '6 changed final result notifies once more' (((Count) -eq 2) -and ((LastNotice) -match 'failed')) "count=$(Count)"

    # 7 .tmp then rename
    $tmpf = Join-Path $dir '0021-flow.json.tmp'
    (Msg '0021' 'blocked' 'via rename' | ConvertTo-Json -Compress) | Set-Content -LiteralPath $tmpf -Encoding UTF8
    Settle
    Check '7a .tmp file alone does not notify' ((Count) -eq 2) "count=$(Count)"
    Move-Item -LiteralPath $tmpf -Destination (Join-Path $dir '0021-flow.json')
    Settle
    Check '7b rename to .json notifies exactly once' (((Count) -eq 3) -and ((LastNotice) -match '0021: blocked')) "count=$(Count)"

    # 8 half-written file
    $half = Join-Path $dir '0022-half.json'
    '{"id":"0022","status":"comp' | Set-Content -LiteralPath $half -Encoding UTF8 -NoNewline
    Settle
    Check '8a half-written file does not notify' ((Count) -eq 3) "count=$(Count)"
    Check '8b half-written file retried then logged' ((Get-Content $log -Raw) -match 'gave up reading 0022-half.json after 3 attempts')
    Put '0022-half.json' (Msg '0022' 'completed' 'now valid')
    Settle
    Check '8c becomes valid -> notifies once' (((Count) -eq 4) -and ((LastNotice) -match '0022: completed')) "count=$(Count)"

    # 9 second instance
    $args2 = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$here\notify_results.ps1`"", '-WatchDir', "`"$dir`"",
        '-LogFile', "`"$log`"", '-SinkFile', "`"$sink`"", '-MaxRunSeconds', '20')
    $p2 = Start-Process powershell -ArgumentList $args2 -WindowStyle Hidden -PassThru -Wait
    Check '9 second instance exits (code 3) due to mutex' (($p2.ExitCode -eq 3) -and ((Get-Content $log -Raw) -match 'second instance'))  "exit=$($p2.ExitCode)"
    Check '9b first instance still running' (-not $proc.HasExited)

    '--- log ---'
    Get-Content $log
} finally {
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
    Start-Sleep -Milliseconds 300
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
if ($script:fail -eq 0) { 'ALL TESTS PASSED' } else { "$($script:fail) TEST(S) FAILED"; exit 1 }
