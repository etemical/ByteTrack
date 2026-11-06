"""打包本地代码并上传到远程服务器（不装依赖、不下载模型——那是 setup_remote.sh 的事）。"""
import io
import os
import tarfile
import warnings

import paramiko

warnings.filterwarnings("ignore")

HOST = os.environ.get("REMOTE_HOST", "connect.westb.seetacloud.com")
PORT = int(os.environ.get("REMOTE_PORT", "27243"))
USER = os.environ.get("REMOTE_USER", "root")
PASS = os.environ["REMOTE_PASS"]
REMOTE_DIR = "/root/autodl-tmp/voice-dialog-demo_v2"
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INCLUDE = ["server", "static", "requirements.txt", "run.sh", "setup_remote.sh",
           "README.md", "tools/test_client.py", "tools/make_test_wav.py", "tools/bench.py",
           "tools/transcribe.py"]


def build_tar() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for item in INCLUDE:
            path = os.path.join(PROJECT, item)
            if not os.path.exists(path):
                continue
            tar.add(path, arcname=item,
                    filter=lambda ti: None if "__pycache__" in ti.name else ti)
    return buf.getvalue()


def main():
    data = build_tar()
    print(f"tar.gz size: {len(data)/1024:.0f} KB")
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PASS, timeout=20)
    sftp = cli.open_sftp()
    with sftp.file("/tmp/voice-dialog-demo_v2.tar.gz", "wb") as f:
        f.write(data)
    sftp.close()
    _, out, err = cli.exec_command(
        f"mkdir -p {REMOTE_DIR} && tar -xzf /tmp/voice-dialog-demo_v2.tar.gz -C {REMOTE_DIR} && "
        f"chmod +x {REMOTE_DIR}/run.sh {REMOTE_DIR}/setup_remote.sh 2>/dev/null; ls {REMOTE_DIR}")
    print(out.read().decode())
    e = err.read().decode()
    if e.strip():
        print("STDERR:", e)
    cli.close()
    print("deployed ->", REMOTE_DIR)


if __name__ == "__main__":
    main()
