# Pi5 Hailo-10H YOLO26 Person Monitor

A lightweight **Raspberry Pi 5 + AI HAT+ 2 (Hailo-10H)** person-presence monitor. It reads JPEG snapshots from a USB camera served by [uStreamer](https://github.com/pikvm/ustreamer), runs a Hailo-10H-compiled **YOLO26n** model, and sends a **Telegram alert with the last detected person's snapshot** after the room remains empty for a configurable period.

The deployment documented here was developed on Raspberry Pi OS and adapted to Ubuntu Server on the Pi 5. It uses the **HailoRT 5.4.0 `InferModel` API directly**, with a single real input image per inference. **No fake batch, Torch inference, Ultralytics monkey patch, or cloud AI service is required at runtime.**

> **Compatibility scope:** This implementation expects a particular **YOLO26 `reg_max=1` raw-head HEF** compiled for **Hailo-10H**, with three matched box heads `(H,W,4)` and class-logit heads `(H,W,80)`. It is not a generic HEF/YOLO model runner. Hardware inference and Telegram delivery must be tested on your own device; repository CI only checks hardware-independent logic.

## Features

- HailoRT 5.4 single-frame hardware inference; no CPU YOLO or unnecessary Ultralytics install.
- Person detection (COCO class `0`) with confidence threshold and NMS.
- Configurable empty-room timeout; camera or inference failures never count as absence.
- Best recent single-person snapshot (sharpness, confidence, apparent size) as a Telegram photo.
- Retry after failed Telegram delivery; only one successful alert until someone returns for a configured period.
- Optional local event archive, with private file permissions.
- Optional user-level systemd service and 19:00–09:00 timers.

## 1. Hardware and OS

- Raspberry Pi 5 and **AI HAT+ 2 / Hailo-10H**. **AI HAT+ (Hailo-8/8L) is different hardware.**
- Raspberry Pi OS **64-bit Trixie or newer**, or Ubuntu Server arm64 with a compatible Pi kernel.
- USB camera exposed via an HTTP uStreamer `/snapshot` endpoint, either on the Pi or another machine/router (e.g. OpenWrt).
- Network access to `api.telegram.org` for alert delivery.

Check hardware enumeration and your actual running kernel before installing anything:

```bash
uname -m && uname -r
lspci -nn | grep -i hailo
```

You should see `Hailo-10H ... [1e60:45c4]` on PCIe. **An illuminated HAT LED only means power, not that the driver is working.**

## 2. Install HailoRT — choose ONE path

### A. Raspberry Pi OS: Raspberry Pi-maintained packages (simplest)

```bash
sudo apt update
sudo apt install dkms hailo-h10-all
sudo reboot
hailortcli scan
hailortcli --version
```

This is the [official Raspberry Pi AI HAT+ 2 package path](https://www.raspberrypi.com/documentation/computers/ai.html). The version supplied by the Raspberry Pi repository may differ from the **5.4.0** used in this project; older Raspberry Pi OS installations supplied the Hailo-10 stack at **5.1.1**. **Do not assume this step installs 5.4.0.** To use this project's proven API version, follow path B after checking/removing incompatible older runtime/driver packages.

### B. Ubuntu Server or HailoRT 5.4.0 on Raspberry Pi OS (project configuration)

Download the matching, official **HailoRT 5.4.0 arm64 runtime `.deb`**, **Hailo-10H PCIe DKMS driver 5.4.0 `.deb`**, **firmware supplied by that driver package**, and **HailoRT Python wheel** corresponding to your exact Python ABI (`cp311` vs `cp313`, aarch64). Obtain them from the [Hailo Developer Zone](https://hailo.ai/developer-zone/) / official [HailoRT releases](https://github.com/hailo-ai/hailort). Package *filenames* can differ: check with `dpkg-deb -f file.deb Package Version` instead of assuming a filename.

```bash
sudo apt update
sudo apt install -y dkms build-essential linux-headers-$(uname -r) python3-venv git
# Replace these paths with the actual downloaded Hailo 5.4.0 .deb files:
sudo apt install /path/to/hailort_5.4.0_arm64.deb /path/to/HAILO10H_PCIE_DRIVER_5.4.0.deb

# Register/build/install the DKMS driver for the *currently running* kernel:
dkms status
sudo dkms autoinstall -k "$(uname -r)"
sudo modprobe hailo1x_pci
hailortcli scan
hailortcli --version
```

For a manually installed `hailort`/`h10-hailort-pcie-driver` 5.4.0 pair, you may choose to protect the packages from being replaced by the Raspberry Pi package repository:

```bash
sudo apt-mark hold hailort h10-hailort-pcie-driver
apt-mark showhold
```

**Do not mix** a 5.1.1 `h10-hailort` userspace library with a 5.4.0 driver, or install both the `hailo-all` (Hailo-8) and `hailo-h10-all` (Hailo-10H) package sets. If upgrading a previously configured Pi OS installation, inspect `dpkg -l | grep -Ei 'hailo|hailort'` and remove the conflicting older packages *before* installing 5.4.0. Avoid blind `apt purge` commands on a Pi running other Hailo applications.

Verify the full device chain:

```bash
lspci -nnk | grep -A4 -i hailo
lsmod | grep hailo1x_pci
ls -l /dev/h1x-*
hailortcli scan
hailortcli fw-control identify
```

Expected Hailo-10H device: `/dev/h1x-0`, with `hailortcli scan` reporting `pci/0001:01:00.0` (PCI address can vary).

## 3. Install the Python runtime

```bash
cd ~
git clone https://github.com/Cope-233/Pi5-Hailo10H-YOLO-Monitor.git
cd ~/Pi5-Hailo10H-YOLO-Monitor
python3 -m venv ~/hailo-env/.venv
~/hailo-env/.venv/bin/python -m pip install -r requirements.txt

# Install the official HailoRT 5.4 Python wheel matching YOUR Python version:
~/hailo-env/.venv/bin/python -m pip install /path/to/hailort-5.4.0-<matching-python-abi>-linux_aarch64.whl

~/hailo-env/.venv/bin/python -c 'import hailo_platform; print("Hailo Python bindings OK")'
```

The wheel filename above is a **placeholder**, not a literal copy/paste command. Verify `~/hailo-env/.venv/bin/python --version` and pick the matching Hailo wheel. **`sudo pip install`, `--break-system-packages`, and system-level Python modifications are unnecessary.** If your venv was created with `uv` and has no pip, use `uv pip install --python ~/hailo-env/.venv/bin/python -r requirements.txt` and install the local wheel via the same `uv pip install --python ...` command.

`requirements.txt` intentionally does not contain `hailo_platform`: it comes from Hailo's architecture- and version-specific wheel. It also does **not** install PyTorch or Ultralytics.

## 4. Obtain a Hailo-10H YOLO26n HEF

Place your **Hailo-10H-compiled**, raw-output model at:

```text
Pi5-Hailo10H-YOLO-Monitor/
└── models/
    └── yolo26n.hef
```

Do **not** substitute the regular `yolo26n.pt`, a Hailo-8 HEF, or an arbitrary model-zoo HEF. Check the compiled model:

```bash
hailortcli parse-hef models/yolo26n.hef
```

If starting without a HEF, compile one in a **supported Hailo DFC 5.x build environment**, normally an appropriate Linux x86_64 workstation or an Ultralytics-supported export service. Ultralytics supports an export workflow of the form:

```python
from ultralytics import YOLO
YOLO("yolo26n.pt").export(format="hailo", name="hailo10h", imgsz=640)
```

See [Ultralytics' Hailo export guide](https://docs.ultralytics.com/integrations/hailo/) for DFC setup, calibration, HEF outputs, and export licensing. **For this direct decoder, ensure the exported HEF has *raw regression + raw classification logits* matching the shape assumptions above**, not fused NMS or an already-sigmoid head. `metadata.yaml` is useful for provenance but this monitor parses the HEF heads directly and does not require that file. We do not redistribute model weights or compiled HEFs in this repository.

## 5. Camera and Telegram configuration

The camera must serve individual JPEGs at `/snapshot` (not just a continuous MJPEG `/stream`). For example, a USB camera connected to OpenWrt can run uStreamer while the Pi fetches snapshots across the LAN. If uStreamer is already installed on your camera host, a simple USB-camera example is:

```bash
ustreamer --device=/dev/video0 --host=0.0.0.0 --port=8080
```

Bind or firewall the server to your private network; do **not** expose an unprotected camera feed to the public internet. Test from the Pi:

```bash
curl -fS http://CAMERA_IP:8080/snapshot -o /tmp/test-camera.jpg
file /tmp/test-camera.jpg
```

Create a bot with [@BotFather](https://t.me/BotFather), start a conversation with the bot, and obtain the chat ID via Telegram Bot API `getUpdates` (or your chosen trusted bot-management method).

```bash
cd ~/Pi5-Hailo10H-YOLO-Monitor
cp .env.example .env
chmod 600 .env
nano .env
```

Change `SNAPSHOT_URL`, `TELEGRAM_BOT_TOKEN`, and `TELEGRAM_CHAT_ID`; adjust `EMPTY_TIMEOUT_SECONDS` as needed. A 10-second timeout was used during development; **300 seconds** is the default here. `REQUIRE_PRIOR_PRESENCE=true` avoids alerts when the room was already empty when the monitor started.

For a manual smoke test:

```bash
set -a
source .env
set +a
~/hailo-env/.venv/bin/python monitor.py
```

Press Ctrl+C to stop. Test **person visible → leave frame → wait configured timeout → receive Telegram photo**, followed by **person returns for `REARM_PERSON_SECONDS` → another departure triggers a new alert**.

## 6. Run as a user systemd service (optional)

The supplied unit assumes the project was cloned to `~/Pi5-Hailo10H-YOLO-Monitor` and the venv to `~/hailo-env/.venv`. Edit the unit if using other paths. It uses **user systemd**, avoiding a hardcoded `pi`/`sonic` username.

```bash
mkdir -p ~/.config/systemd/user
cp systemd/* ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user start yolo-monitor.service
systemctl --user status yolo-monitor.service
journalctl --user -u yolo-monitor.service -f
```

After the manual test, optionally enable nightly **19:00 start / 09:00 stop** timers:

```bash
systemctl --user stop yolo-monitor.service
systemctl --user enable --now yolo-monitor-start.timer yolo-monitor-stop.timer
systemctl --user list-timers --all | grep yolo-monitor
sudo loginctl enable-linger "$USER"
```

Do **not** simultaneously enable `yolo-monitor.service` itself on boot if you want timer-only operation. Linger allows user timers to run without an interactive login. To turn off the schedule:

```bash
systemctl --user disable --now yolo-monitor-start.timer yolo-monitor-stop.timer
systemctl --user stop yolo-monitor.service
```

## 7. Troubleshooting — lessons learned

| Symptom | Root cause / action |
| --- | --- |
| `HAILO_OUT_OF_PHYSICAL_DEVICES(74)` | First stop other Hailo processes; check `hailortcli scan`, `lsmod`, and `/dev/h1x-0`. Device contention or an unavailable driver can produce this error. |
| `Hailo devices not found` after kernel upgrade | **We hit this on Ubuntu 7.0.0-1017 → 7.0.0-1020.** DKMS showed only `hailo1x_pci/5.4.0: added`, and the new kernel had no module. Run the DKMS recovery below. |
| `Module hailo1x_pci not found` | Install `linux-headers-$(uname -r)`, rebuild/install DKMS for the running kernel, and `modprobe hailo1x_pci`. |
| `HAILO_NOT_IMPLEMENTED(7)` | Often from **Ultralytics' legacy** `VDevice.configure()` / `InferVStreams` Hailo backend. This monitor **does not use that backend**; it calls `InferModel` directly. See `extras/` for an optional adapter. |
| Works with `apt`, breaks after upgrade | Avoid accidentally replacing HailoRT 5.4.0 with a Raspberry Pi-distributed 5.1.x package; use version checks and optionally `apt-mark hold`. Holding Hailo packages **does not hold kernel updates**. |
| `ModuleNotFoundError: hailo_platform` | Activate the correct Python venv and install the matching HailoRT 5.4.0 aarch64 Python wheel. `apt install python3-pip` does not install Hailo bindings into a separate venv. |
| JPEG `overread`, intermittent frame decoding | Prefer uStreamer's `/snapshot` endpoint instead of decoding its continuous MJPEG stream with OpenCV/FFmpeg. |
| Telegram never arrives | Verify bot token/chat ID/network access. The monitor retries failed sends and logs HTTP status without including credentials. |
| Too many or incorrect detections | Check output shapes and whether classification tensors are **logits**; the decoder applies sigmoid. Verify the HEF was compiled for Hailo-10H and compare against known images. |

**DKMS recovery after an Ubuntu kernel upgrade:**

```bash
uname -r
dkms status
sudo apt install dkms build-essential linux-headers-$(uname -r)
sudo dkms build hailo1x_pci/5.4.0 -k "$(uname -r)"
sudo dkms install hailo1x_pci/5.4.0 -k "$(uname -r)"
sudo depmod -a
sudo modprobe hailo1x_pci
hailortcli scan
```

If `dkms build` fails, check `/var/lib/dkms/hailo1x_pci/5.4.0/build/make.log` rather than repeatedly reinstalling HailoRT. If PCIe is missing from `lspci`, debug hardware power/cable/firmware first. If PCIe is visible but driver initialization fails, check `sudo dmesg | grep -i hailo | tail -80`; try a full poweroff/cold boot before invasive driver changes.

## Development and license

```bash
python -m compileall -q monitor.py extras/hailo10h_ultralytics_patch.py
python -m unittest discover -s tests -v
```

The tests cover decoding math, image preprocessing, NMS, empty-room state transitions, failure handling, and alert retry logic. **They do not emulate Hailo firmware, the camera, or Telegram delivery.** Always perform an end-to-end test on the Pi before unattended use.

Code is licensed under [MIT](LICENSE). HailoRT, Ultralytics, YOLO model weights and HEFs remain subject to their respective third-party licenses and distribution rules. Optional historical Ultralytics adaptation is in [`extras/`](extras/README.md).