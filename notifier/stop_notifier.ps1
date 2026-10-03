# Optional: stop any running notify_results.ps1 notifier processes.
$procs = Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
    Where-Object { $_.CommandLine -like '*notify_results.ps1*' -and $_.ProcessId -ne $PID }
foreach ($p in $procs) {
    Stop-Process -Id $p.ProcessId -Force
    "stopped notifier pid $($p.ProcessId)"
}
if (-not $procs) { 'no notifier running' }
