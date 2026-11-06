import os, sys, time, paramiko, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

HOST = os.environ.get("REMOTE_HOST", "connect.westb.seetacloud.com")
PORT = int(os.environ.get("REMOTE_PORT", "27243"))
USER = os.environ.get("REMOTE_USER", "root")
PASS = os.environ["REMOTE_PASS"]


def run(cmd, timeout=120):
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PASS, timeout=20)
    chan = cli.get_transport().open_session()
    chan.exec_command(cmd)
    chan.settimeout(2)
    out, err = b"", b""
    deadline = time.time() + timeout
    while time.time() < deadline:
        got = False
        try:
            if chan.recv_ready():
                out += chan.recv(65536)
                got = True
        except Exception:
            pass
        try:
            if chan.recv_stderr_ready():
                err += chan.recv_stderr(65536)
                got = True
        except Exception:
            pass
        if chan.exit_status_ready() and not chan.recv_ready() and not chan.recv_stderr_ready():
            break
        if not got:
            time.sleep(0.3)
    cli.close()
    return out.decode("utf-8", "replace"), err.decode("utf-8", "replace")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "echo connected"
    out, err = run(cmd, timeout=int(sys.argv[2]) if len(sys.argv) > 2 else 120)
    print(out)
    if err.strip():
        print("STDERR:", err, file=sys.stderr)
