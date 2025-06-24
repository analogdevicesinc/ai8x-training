# Custom Keyword Spotting (KWS) on MAX78000 with Offline 3-Class Dataset

This project presents a customized configuration of the Keyword Spotting (KWS) pipeline designed for deployment on the MAX78000 platform. It demonstrates how to integrate a user-defined, offline dataset with three classes (`class1`, `class2`, `class3`) using the AI8X training framework, with minimal but essential modifications to the default workflow.

---

## 📌 Project Scope

This repository focuses on two key components:

1. **Offline Dataset Preparation Script**  
   A custom script (`adtotestandvalidation.py`) to process raw audio files and generate:
   - `validation_list.txt` and `testing_list.txt` (used for dataset indexing)

2. **Modified `kws20.py`**  
   The standard KWS dataset class has been streamlined to:
   - Support a **fixed n-class dataset**
   - Remove dependencies on the original Google Speech Commands dataset
   - Eliminate unused logic such as `_silence_`, `_unknown_`, `librispeech`, and dataset downloads

All other training and deployment instructions conform to the official Maxim documentation:

> 📘 [Official Guide – Making Your Own Audio and Image Classification Application](https://github.com/analogdevicesinc/MaximAI_Documentation/blob/main/Guides/Making%20Your%20Own%20Audio%20and%20Image%20Classification%20Application%20Using%20Keyword%20Spotting%20and%20Cats-vs-Dogs.md)

---

## 📁 Directory Structure

Organize your audio files as follows:

```
data/KWS/raw/
├── class1/
├── class2/
└── class3/
```

Each folder should contain `.wav` files corresponding to one of the target classes.

---

## 🧰 Usage Instructions

### 1. Dataset Preprocessing

Run the preprocessing script to generate training assets:

```bash
python3 adtotestandvalidation.py \
  --raw-dir data/KWS/raw/ \
  --output-dir data/KWS/processed/
```

This will create:
- `validation_list.txt`
- `testing_list.txt`



---

## 🛠 Modified Components

| File                      | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| `adtotestandvalidation.py` | Converts raw audio into  generates index lists              |
| `models/kws20.py`         | Simplified and adapted to support only 3 offline classes without downloads |

> All other framework logic, model training, quantization, and deployment tools remain unchanged and follow the official AI8X and MAX78000 documentation.

---

## 📄 License

This project is distributed under the Apache 2.0 License, in line with the AI8X training framework.

---

## 👤 Author & Maintainer

**Abdullah**  
Developer & Embedded ML Engineer  
Contributed the offline dataset integration and modified KWS dataset loader for MAX78000 deployment.

---

## 📣 Acknowledgements

Special thanks to the [Maxim Integrated AI](https://github.com/MaximIntegratedAI) team for providing the AI8X training framework and hardware SDK.
