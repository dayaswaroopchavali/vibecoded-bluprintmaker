# Result notifier

`notify_results.ps1` is a small local Windows PowerShell 5.1 script that watches
`a2a/from_claude` with `System.IO.FileSystemWatcher` (Created/Changed/Renamed) and shows a
Windows notification to the USER when Claude writes a new or changed final result
(`status` = completed / needs_review / blocked / failed).

What it does
- Baselines existing messages at start (hashes only, no notifications for them).
- Ignores `claim-*.json`, `.tmp` files, and JSON whose status is claimed/running/ready.
- De-duplicates by content hash; re-writes with identical content do not notify again.
- Retries a mid-write file a few times (default 4 x 300 ms), then logs "gave up" and waits for the next change.
- Notification: Windows toast (Windows.UI.Notifications), falling back to a balloon NotifyIcon, then log only.
  Title `Claude result <id>: <status>`, body = the `result` string truncated to ~200 chars.
- Appends every delivered notice to `notifier/notifier.log` (timestamp, file, status, hash).
- Single instance via a named mutex (per watch directory). A cheap local rescan every 60 s is a safety net for missed events.
- Treats message content strictly as data: only `id`, `status`, `result` are read, as strings. Nothing is executed.
- No AI calls, no model polling, no network, no GPU.

Usage
- Test: `powershell -NoProfile -ExecutionPolicy Bypass -File notifier/test_notifier.ps1`
- Start hidden: `Start-Process -WindowStyle Hidden powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','notifier\notify_results.ps1'`
- Stop: `powershell -File notifier/stop_notifier.ps1`
- Parameters: `-WatchDir`, `-LogFile`, `-SinkFile` / `-DryRun` (write notices to a file instead of a toast), `-RetryCount`, `-RetryDelayMs`, `-RescanSeconds`, `-MaxRunSeconds`, `-MutexName`.

Honest limitations
- This only notifies the human user. It CANNOT wake, resume or message a Codex thread, and no automatic Codex resumption is claimed.
- The earlier five-minute Codex automation was deleted and must remain deleted.
- It does not survive logoff/reboot (no startup entry is installed).
- Whether a toast is visibly shown depends on Windows notification/Focus settings; the log records delivery attempts, not that the user saw it.
