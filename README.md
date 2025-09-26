````markdown
# 🌙 Lunar

**Lunar** is a neural-network-powered aim assist that uses real-time object detection accelerated with CUDA on NVIDIA GPUs.

---

## 📝 About

Lunar can be adapted to work with a variety of FPS games; however, it is currently configured for **Fortnite**.  
Unlike memory-injection tools, Lunar does **not** interact with the memory of other processes.  

Player detection is based on the [YOLOv5](https://github.com/ultralytics/yolov5) architecture written in **PyTorch**.

📺 A (currently outdated) demo video is available [here](https://www.youtube.com/watch?v=XDAcQNUuT84).

![thumbnail](https://user-images.githubusercontent.com/45726273/126563920-193ca8df-de70-4a91-81ec-d781ee961332.png)

---

## ⚙️ Installation

1. Install [Python 3.8 or later](https://www.python.org/downloads/).
2. In the root directory of the project, install the required dependencies:

```bash
pip install -r requirements.txt
````

---

## 🚀 Usage

Run Lunar:

```bash
python lunar.py
```

Update sensitivity settings:

```bash
python lunar.py setup
```

Collect image data for annotation/training:

```bash
python lunar.py collect_data
```

---

## ⚠️ Known Issues

* **Mouse movement**: The current [SendInput](https://github.com/zeyad-mansour/Lunar/blob/45e05373036f8bd072667313c155e55735cd7f57/lib/aimbot.py#L126) method is relatively slow, so the crosshair may lag behind moving targets. Increasing the [`pixel_increment`](https://github.com/zeyad-mansour/Lunar/blob/45e05373036f8bd072667313c155e55735cd7f57/lib/aimbot.py#L56) (e.g. to `4`) can reduce function calls and improve responsiveness.
* **False positives** may occur under certain lighting conditions.

---

## 🤝 Contributing

Pull requests are welcome. If you have suggestions, questions, or find issues, please open an [issue](https://github.com/zeyad-mansour/Lunar/issues) with details.
If you find this project interesting or helpful, consider starring the repository ⭐️.

---

## 📜 License

This project is distributed under the [GNU General Public License v3.0](https://github.com/zeyad-mansour/Lunar/blob/main/LICENSE).

```
