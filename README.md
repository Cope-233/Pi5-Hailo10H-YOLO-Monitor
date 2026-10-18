# Pi5 Hailo-10H YOLO26 Monitor

A person-presence monitor for **Raspberry Pi 5 + AI HAT+ 2 (Hailo-10H)**. A directly connected **UVC USB webcam** supplies frames; **YOLO26n** runs on Hailo-10H and sends the last detected person's image to **Telegram** when the room has been empty for a configured time.

The monitor uses the **HailoRT 5.4.0 InferModel API** directly, one frame per inference. No Ultralytics installation or batch workaround is needed to run it.

## Features

- UVC camera input through OpenCV/V4L2 (`/dev/video0` or `/dev/v4l/by-id/…`).
- Person detection, empty-room countdown, and Telegram photo notification.
- Selection of a clear recent image; camera errors do not trigger empty-room alerts.
- Telegram retry, automatic rearming, and optional local snapshots.
- Optional systemd schedule: **19:00 start / 09:00 stop**.

## 1. Hardware and HailoRT

Required: Raspberry Pi 5, **AI HAT+ 2 / Hailo-10H**, a Linux UVC-compatible USB webcam, Raspberry Pi OS 64-bit or Ubuntu Server arm64, and network access to Telegram.

Install tools:

```bash
sudo apt update
sudo apt install -y git python3-venv v4l-utils dkms build-essential linux-headers-$(uname -r)
```

Download the **HailoRT 5.4.0 arm64 runtime**, **Hailo-10H PCIe driver 5.4.0** (`.deb` packages), and the **HailoRT 5.4.0 Python wheel** matching your Python version from [Hailo](https://hailo.ai/developer-zone/). Install the downloaded packages using their actual filenames:

```bash
sudo apt install /path/to/hailort_5.4.0_arm64.deb /path/to/hailort-pcie-driver_5.4.0_all.deb
sudo dkms autoinstall -k "$(uname -r)"
sudo modprobe hailo1x_pci
hailortcli scan
sudo apt-mark hold hailort h10-hailort-pcie-driver
```

Use matching **5.4.0 runtime, driver, and wheel**; replace older Raspberry Pi-distributed Hailo-10H packages where present. After future kernel upgrades, rebuild the DKMS driver for the new kernel with `sudo dkms autoinstall -k "$(uname -r)"`.

## 2. Install the monitor

```bash
cd ~
git clone https://github.com/Cope-233/Pi5-Hailo10H-YOLO-Monitor.git
cd Pi5-Hailo10H-YOLO-Monitor
python3 -m venv ~/hailo-env/.venv
~/hailo-env/.venv/bin/python -m pip install -r requirements.txt
# Replace with your downloaded HailoRT 5.4.0 wheel (correct Python ABI / aarch64):
~/hailo-env/.venv/bin/python -m pip install /path/to/hailort-5.4.0-...-linux_aarch64.whl
```

The repository already includes **`models/yolo26n.hef`**, compiled for Hailo-10H. The monitor loads this HEF automatically. It expects YOLO26n raw box and class heads, without integrated NMS.

## 3. Connect the USB webcam and Telegram

Plug the UVC camera **directly into the Raspberry Pi**. Check its video device:

```bash
v4l2-ctl --list-devices
ls -l /dev/v4l/by-id/
```

Use the image-capture node (normally `/dev/video0` or a `video-index0` symlink). The service user needs permission to access the camera; on systems using the `video` group, add your user with `sudo usermod -aG video "$USER"` and sign in again.

Create a Telegram bot through [@BotFather](https://t.me/BotFather), message the bot, and obtain your chat ID using Telegram's `getUpdates` API.

```bash
cp .env.example .env
chmod 600 .env
nano .env
```

Set `CAMERA_DEVICE`, `TELEGRAM_BOT_TOKEN`, and `TELEGRAM_CHAT_ID`. The defaults use 640×480 MJPEG at 15 FPS and send an alert after **300 seconds** without a person. Set `CAMERA_FOURCC=YUYV` if your webcam does not support MJPEG.

Test interactively:

```bash
set -a; source .env; set +a
~/hailo-env/.venv/bin/python monitor.py
```

Have someone enter the camera view, then leave. Once `EMPTY_TIMEOUT_SECONDS` has elapsed, the bot sends a photo. Press Ctrl+C to stop.

## 4. Run with systemd (optional)

The supplied **user service** expects the repository in `~/Pi5-Hailo10H-YOLO-Monitor` and Python in `~/hailo-env/.venv`.

```bash
mkdir -p ~/.config/systemd/user
cp systemd/* ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user start yolo-monitor.service
journalctl --user -u yolo-monitor.service -f
```

Enable the daily 19:00–09:00 schedule:

```bash
systemctl --user stop yolo-monitor.service
systemctl --user enable --now yolo-monitor-start.timer yolo-monitor-stop.timer
sudo loginctl enable-linger "$USER"
```

Stop the schedule:

```bash
systemctl --user disable --now yolo-monitor-start.timer yolo-monitor-stop.timer
systemctl --user stop yolo-monitor.service
```

## License

[MIT](LICENSE). Third-party HailoRT, model, and Ultralytics artifacts retain their respective licensing terms. A historical optional Ultralytics backend adapter is available under [`extras/`](extras/README.md); it is **not** used by the monitor.