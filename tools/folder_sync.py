"""Folder Sync - simple PyQt6 frontend around rsync.

Scans the local /24 network for machines with SSH open (port 22), lets you
pick one, then pushes or pulls a folder with rsync over SSH.

rsync backend is auto-detected:
  - native rsync on PATH (cwRsync / scoop / cygwin), or
  - rsync inside WSL (Windows paths are mapped to /mnt/<drive>/...)

SSH must be key-based (rsync runs non-interactively, so password prompts
can't be answered) - the "Set up SSH key" button opens a console window
running ssh-copy-id so you can enter the password once.

Usage: python tools/folder_sync.py
"""
import os
import shutil
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PyQt6.QtCore import Qt, QProcess, QSettings, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QSplitter,
    QVBoxLayout, QWidget,
)

SSH_OPTS = "-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=8"


# ---------------------------------------------------------------- backend

def find_backend():
    """Return ('native', rsync_path) or ('wsl', 'wsl') or (None, None)."""
    exe = shutil.which("rsync")
    if exe:
        return "native", exe
    if shutil.which("wsl"):
        try:
            r = subprocess.run(["wsl", "-e", "which", "rsync"],
                               capture_output=True, timeout=15)
            if r.returncode == 0 and r.stdout.strip():
                return "wsl", "wsl"
        except (OSError, subprocess.TimeoutExpired):
            pass
    return None, None


def to_posix(win_path, mode):
    """Map D:\\foo\\bar to /mnt/d/foo/bar (wsl) or /cygdrive/d/foo/bar (native)."""
    p = Path(win_path).resolve()
    if not p.drive:
        return str(p).replace("\\", "/")
    drive = p.drive[0].lower()
    rest = str(p)[len(p.drive):].replace("\\", "/")
    prefix = "/mnt/" if mode == "wsl" else "/cygdrive/"
    return f"{prefix}{drive}{rest}"


# ---------------------------------------------------------------- scanner

def local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # no traffic sent - just picks the route
        return s.getsockname()[0]
    finally:
        s.close()


class ScanWorker(QThread):
    """Probe every address in the local /24 for an open SSH port."""
    host_found = pyqtSignal(str, str)   # ip, hostname
    scan_done = pyqtSignal(int)         # number of hosts found

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop = False

    def stop(self):
        self._stop = True

    @staticmethod
    def _probe(ip):
        try:
            with socket.create_connection((ip, 22), timeout=0.4):
                return ip
        except OSError:
            return None

    def run(self):
        try:
            me = local_ip()
        except OSError:
            self.scan_done.emit(0)
            return
        prefix = me.rsplit(".", 1)[0]
        targets = [f"{prefix}.{i}" for i in range(1, 255) if f"{prefix}.{i}" != me]
        found = 0
        with ThreadPoolExecutor(max_workers=64) as pool:
            futures = {pool.submit(self._probe, ip): ip for ip in targets}
            for fut in as_completed(futures):
                if self._stop:
                    break
                ip = fut.result()
                if ip:
                    try:
                        name = socket.gethostbyaddr(ip)[0]
                    except OSError:
                        name = ""
                    found += 1
                    self.host_found.emit(ip, name)
        self.scan_done.emit(found)


# ---------------------------------------------------------------- remote browser

class RemoteBrowseDialog(QDialog):
    """Navigate the remote machine's directories over SSH and pick one."""

    def __init__(self, parent, mode, user, host, start="~"):
        super().__init__(parent)
        self.mode = mode
        self.target = f"{user}@{host}"
        self.current = None
        self.setWindowTitle(f"Browse {self.target}")
        self.resize(460, 480)

        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.returnPressed.connect(
            lambda: self.navigate(self.path_edit.text().strip() or "~"))
        row.addWidget(self.path_edit, stretch=1)
        self.hidden_chk = QCheckBox("Hidden")
        self.hidden_chk.toggled.connect(
            lambda: self.current and self.navigate(self.current))
        row.addWidget(self.hidden_chk)
        layout.addLayout(row)

        self.listing = QListWidget()
        self.listing.itemDoubleClicked.connect(self._descend)
        layout.addWidget(self.listing, stretch=1)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        btns = QHBoxLayout()
        btns.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        btns.addWidget(cancel)
        select = QPushButton("Select this folder")
        select.setDefault(True)
        select.clicked.connect(self.accept)
        btns.addWidget(select)
        layout.addLayout(btns)

        self.navigate(start or "~")

    def _ssh_list(self, path):
        """Return (ok, resolved_path, dir_names, error_text)."""
        flags = "-1pLA" if self.hidden_chk.isChecked() else "-1pL"
        if path in ("~", ""):
            cd = "cd"
        else:
            cd = "cd -- '" + path.replace("'", "'\\''") + "'"
        script = f"{cd} && pwd && ({{ ls {flags} -- . 2>/dev/null; }}; true)"
        cmd = ((["wsl"] if self.mode == "wsl" else [])
               + ["ssh"] + SSH_OPTS.split() + [self.target, script])
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=20,
                               creationflags=subprocess.CREATE_NO_WINDOW)
        except subprocess.TimeoutExpired:
            return False, None, [], "Timed out talking to the remote machine."
        if r.returncode != 0 or not r.stdout.strip():
            err = r.stderr.strip() or f"ssh exited with code {r.returncode}"
            if r.returncode == 255:
                err += "  (SSH auth? Use 'Set up SSH key…' in the main window.)"
            return False, None, [], err
        lines = r.stdout.splitlines()
        dirs = sorted((ln[:-1] for ln in lines[1:] if ln.endswith("/")),
                      key=str.lower)
        return True, lines[0].strip(), dirs, ""

    def navigate(self, path):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            ok, cwd, dirs, err = self._ssh_list(path)
        finally:
            QApplication.restoreOverrideCursor()
        if not ok:
            self.status.setText(err)
            return
        self.current = cwd
        self.path_edit.setText(cwd)
        self.listing.clear()
        if cwd != "/":
            self.listing.addItem("..")
        self.listing.addItems([d + "/" for d in dirs])
        self.status.setText(f"{len(dirs)} subfolder(s). Double-click to open, "
                            "then 'Select this folder'.")

    def _descend(self, item):
        name = item.text().rstrip("/")
        base = self.current.rstrip("/")
        self.navigate(f"{base}/{name}" if name != ".." else (base.rsplit("/", 1)[0] or "/"))

    def selected_path(self):
        return self.current


# ---------------------------------------------------------------- main window

class FolderSyncWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Folder Sync (rsync)")
        self.resize(980, 620)
        self.settings = QSettings("GL_Simple", "FolderSync")
        self.mode, self.rsync = find_backend()
        self.scan_worker = None
        self.proc = None
        self._build_ui()
        self._load_settings()
        if self.mode is None:
            self.log_line("ERROR: no rsync found. Install WSL (wsl --install) or "
                          "a native rsync (e.g. 'scoop install rsync'), then restart.")
            self.run_btn.setEnabled(False)
            self.dry_btn.setEnabled(False)
        else:
            self.log_line(f"rsync backend: {self.mode} ({self.rsync})")

    # ------------------------------------------------------------ UI

    def _build_ui(self):
        split = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(split)

        # left: network scan
        left = QWidget()
        ll = QVBoxLayout(left)
        self.scan_btn = QPushButton("Scan network for SSH machines")
        self.scan_btn.clicked.connect(self.start_scan)
        ll.addWidget(self.scan_btn)
        self.host_list = QListWidget()
        self.host_list.itemClicked.connect(self._host_picked)
        ll.addWidget(self.host_list)
        self.scan_status = QLabel("Not scanned yet.")
        ll.addWidget(self.scan_status)
        split.addWidget(left)

        # right: sync form + log
        right = QWidget()
        rl = QVBoxLayout(right)

        conn = QGroupBox("Connection")
        cl = QHBoxLayout(conn)
        cl.addWidget(QLabel("User:"))
        self.user_edit = QLineEdit()
        self.user_edit.setMaximumWidth(140)
        cl.addWidget(self.user_edit)
        cl.addWidget(QLabel("Host:"))
        self.host_edit = QLineEdit()
        self.host_edit.setPlaceholderText("pick from scan, or type IP/hostname")
        cl.addWidget(self.host_edit, stretch=1)
        self.key_btn = QPushButton("Set up SSH key…")
        self.key_btn.setToolTip("Opens a console running ssh-copy-id so you can "
                                "enter the password once; syncs then run without one.")
        self.key_btn.clicked.connect(self.setup_ssh_key)
        cl.addWidget(self.key_btn)
        rl.addWidget(conn)

        folders = QGroupBox("Folders (contents of one folder sync into the other)")
        fl = QVBoxLayout(folders)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Local folder:"))
        self.local_edit = QLineEdit()
        row1.addWidget(self.local_edit, stretch=1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_local)
        row1.addWidget(browse)
        fl.addLayout(row1)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Remote folder:"))
        self.remote_edit = QLineEdit()
        self.remote_edit.setPlaceholderText("/home/user/somefolder")
        row2.addWidget(self.remote_edit, stretch=1)
        browse_r = QPushButton("Browse…")
        browse_r.clicked.connect(self._browse_remote)
        row2.addWidget(browse_r)
        fl.addLayout(row2)
        rl.addWidget(folders)

        opts = QHBoxLayout()
        self.direction = QComboBox()
        self.direction.addItems(["Push  (this PC  →  Linux)",
                                 "Pull  (Linux  →  this PC)"])
        opts.addWidget(self.direction)
        self.mirror_chk = QCheckBox("Mirror (--delete: remove files missing from source)")
        opts.addWidget(self.mirror_chk)
        opts.addStretch(1)
        rl.addLayout(opts)

        btns = QHBoxLayout()
        self.dry_btn = QPushButton("Dry run (preview)")
        self.dry_btn.clicked.connect(lambda: self.start_sync(dry=True))
        btns.addWidget(self.dry_btn)
        self.run_btn = QPushButton("Sync now")
        self.run_btn.setStyleSheet("font-weight: bold;")
        self.run_btn.clicked.connect(lambda: self.start_sync(dry=False))
        btns.addWidget(self.run_btn)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_sync)
        btns.addWidget(self.stop_btn)
        btns.addStretch(1)
        rl.addLayout(btns)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Consolas", 9))
        self.log.setMaximumBlockCount(5000)
        rl.addWidget(self.log, stretch=1)

        split.addWidget(right)
        split.setSizes([300, 680])

    def log_line(self, text):
        self.log.appendPlainText(text)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    # ------------------------------------------------------------ scan

    def start_scan(self):
        if self.scan_worker and self.scan_worker.isRunning():
            return
        self.host_list.clear()
        self.scan_btn.setEnabled(False)
        try:
            self.scan_status.setText(f"Scanning {local_ip().rsplit('.', 1)[0]}.0/24 …")
        except OSError:
            self.scan_status.setText("Scanning…")
        self.scan_worker = ScanWorker(self)
        self.scan_worker.host_found.connect(self._add_host)
        self.scan_worker.scan_done.connect(self._scan_done)
        self.scan_worker.start()

    def _add_host(self, ip, name):
        label = f"{ip}    {name}" if name else ip
        item = QListWidgetItem(label)
        item.setData(Qt.ItemDataRole.UserRole, ip)
        self.host_list.addItem(item)

    def _scan_done(self, count):
        self.scan_btn.setEnabled(True)
        self.scan_status.setText(f"Found {count} machine(s) with SSH open.")

    def _host_picked(self, item):
        self.host_edit.setText(item.data(Qt.ItemDataRole.UserRole))

    def _browse_remote(self):
        user = self.user_edit.text().strip()
        host = self.host_edit.text().strip()
        if not (user and host):
            QMessageBox.warning(self, "Missing info", "Fill in user and host first.")
            return
        if self.mode is None:
            QMessageBox.warning(self, "No backend", "No rsync/ssh backend available.")
            return
        dlg = RemoteBrowseDialog(self, self.mode, user, host,
                                 start=self.remote_edit.text().strip() or "~")
        if dlg.exec() and dlg.selected_path():
            self.remote_edit.setText(dlg.selected_path())

    def _browse_local(self):
        d = QFileDialog.getExistingDirectory(self, "Local folder",
                                             self.local_edit.text() or str(Path.home()))
        if d:
            self.local_edit.setText(d)

    # ------------------------------------------------------------ sync

    def _validate(self):
        user = self.user_edit.text().strip()
        host = self.host_edit.text().strip()
        local = self.local_edit.text().strip()
        remote = self.remote_edit.text().strip()
        if not (user and host and local and remote):
            QMessageBox.warning(self, "Missing info",
                                "User, host, local folder and remote folder are all required.")
            return None
        if not Path(local).is_dir():
            QMessageBox.warning(self, "Bad folder", f"Local folder does not exist:\n{local}")
            return None
        return user, host, local, remote

    def _build_args(self, user, host, local, remote, dry):
        pull = self.direction.currentIndex() == 1
        args = ["-rltvz", "-s", "--info=progress2", "-e", f"ssh {SSH_OPTS}"]
        if not pull:
            args.insert(0, "-a")  # full metadata when the destination is Linux
        else:
            # Windows destination: chmod/chown on NTFS just produce noise
            args += ["--no-perms", "--no-owner", "--no-group", "-O"]
        if self.mirror_chk.isChecked():
            args.append("--delete")
        if dry:
            args.append("--dry-run")
        local_posix = to_posix(local, self.mode) + "/"
        remote_spec = f"{user}@{host}:{remote.rstrip('/')}/"
        args += [remote_spec, local_posix] if pull else [local_posix, remote_spec]
        return args

    def start_sync(self, dry):
        if self.proc:
            return
        v = self._validate()
        if not v:
            return
        user, host, local, remote = v
        self._save_settings()
        args = self._build_args(user, host, local, remote, dry)
        program = "wsl" if self.mode == "wsl" else self.rsync
        if self.mode == "wsl":
            args = ["rsync"] + args

        self.log_line("")
        self.log_line(("DRY RUN: " if dry else "SYNC: ") + program + " " + " ".join(args))
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._proc_output)
        self.proc.finished.connect(self._proc_done)
        self.run_btn.setEnabled(False)
        self.dry_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.proc.start(program, args)

    def _proc_output(self):
        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in data.replace("\r", "\n").split("\n"):
            if line.strip():
                self.log_line(line)

    def _proc_done(self, code, _status):
        if code == 0:
            self.log_line(">>> Done.")
        elif code == 255 or code == 12:
            self.log_line(f">>> FAILED (exit {code}). This is usually SSH auth: "
                          "click 'Set up SSH key…' to install your key on the remote "
                          "machine, then try again.")
        else:
            self.log_line(f">>> rsync exited with code {code}.")
        self.proc = None
        self.run_btn.setEnabled(self.mode is not None)
        self.dry_btn.setEnabled(self.mode is not None)
        self.stop_btn.setEnabled(False)

    def stop_sync(self):
        if self.proc:
            self.proc.kill()
            self.log_line(">>> Stopped.")

    # ------------------------------------------------------------ ssh key

    def setup_ssh_key(self):
        user = self.user_edit.text().strip()
        host = self.host_edit.text().strip()
        if not (user and host):
            QMessageBox.warning(self, "Missing info", "Fill in user and host first.")
            return
        target = f"{user}@{host}"
        if self.mode == "wsl":
            script = ("test -f ~/.ssh/id_ed25519 || ssh-keygen -t ed25519 -N '' "
                      "-f ~/.ssh/id_ed25519; "
                      f"ssh-copy-id -o StrictHostKeyChecking=accept-new {target}; "
                      "echo; read -p 'Press Enter to close…'")
            cmd = ["wsl", "bash", "-c", script]
        else:
            # Windows OpenSSH has no ssh-copy-id - append the key manually
            key = Path.home() / ".ssh" / "id_ed25519"
            if not key.exists():
                subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-f", str(key)])
            cmd = ["cmd", "/k",
                   f'type "{key}.pub" | ssh -o StrictHostKeyChecking=accept-new {target} '
                   '"mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && '
                   'chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys"']
        self.log_line(f"Opening console to install SSH key on {target} - "
                      "enter the Linux password there once.")
        subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)

    # ------------------------------------------------------------ settings

    def _load_settings(self):
        self.user_edit.setText(self.settings.value("user", ""))
        self.host_edit.setText(self.settings.value("host", ""))
        self.local_edit.setText(self.settings.value("local", ""))
        self.remote_edit.setText(self.settings.value("remote", ""))
        self.direction.setCurrentIndex(int(self.settings.value("direction", 0)))
        self.mirror_chk.setChecked(self.settings.value("mirror", "false") == "true")

    def _save_settings(self):
        self.settings.setValue("user", self.user_edit.text().strip())
        self.settings.setValue("host", self.host_edit.text().strip())
        self.settings.setValue("local", self.local_edit.text().strip())
        self.settings.setValue("remote", self.remote_edit.text().strip())
        self.settings.setValue("direction", self.direction.currentIndex())
        self.settings.setValue("mirror", "true" if self.mirror_chk.isChecked() else "false")

    def closeEvent(self, event):
        self._save_settings()
        if self.scan_worker and self.scan_worker.isRunning():
            self.scan_worker.stop()
            self.scan_worker.wait(2000)
        if self.proc:
            self.proc.kill()
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = FolderSyncWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
