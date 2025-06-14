# Improved Domain Adaptation Network Based on Wasserstein Distance for Motor Imagery EEG Classification


이 프로젝트는 뇌-컴퓨터 인터페이스(BCI) 데이터를 위한 도메인 적응 분류기를 구현한 것입니다. WGAN(Wasserstein GAN)과 도메인 적응을 결합하여 모터 이미지(Motor Imagery) 분류를 수행합니다.

## 프로젝트 흐름도

![image](https://github.com/user-attachments/assets/892cef6c-d2da-4c58-9859-38d9e8ba3b03)


## 프로젝트 구조

```
.
├── Datasets/          # BCI 데이터셋 저장 디렉토리
├── models/           # 모델 아키텍처 정의
│   ├── generator.py  # 특징 추출기와 분류기
│   └── discriminator.py  # 도메인 판별기
├── results/          # 실험 결과 저장
├── utils/           # 유틸리티 함수
├── main.py          # 메인 실행 파일
├── requirements.txt  # 의존성 패키지 목록
└── README.md        # 프로젝트 문서
```

## 주요 기능

- WGAN 기반 도메인 적응 학습
- 모터 이미지 분류 (4개 클래스)
- 교차 검증을 통한 성능 평가
- 자동화된 실험 결과 기록

## Installation

1. **Clone the Repository**
    ```bash
    git clone https://github.com/HurDuhwan/2025-1_BCI_Project.git
    cd 2025-1_BCI_Project
    ```

2. **(Optional) Pull Docker Image**
    ```bash
    docker pull pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel
    ```

3. **(Optional) Run Docker Container**
    ```bash
    docker run -it --rm --gpus all \
      -v "$(pwd)":/workspace -w /workspace \
      pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel /bin/bash
    ```

4. **Install Python Requirements**
    ```bash
    pip install -r requirements.txt
    ```

---

## Data Preparation

- Download the BCI Competition IV 2a dataset and place it in the `Datasets/` directory.
- If you have preprocessed or converted data, place it in the same directory as required by the code.

---

## Train

To train the model, run:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py
```
- You can specify the GPU number by changing the value after `CUDA_VISIBLE_DEVICES=`.
- Hyperparameters can be modified directly in `main.py` or in the relevant config files.

---

## Evaluation

- Training and evaluation logs will be saved in the `results/` directory.
- The best and average accuracy for each subject will be recorded in `results/sub_result.txt`.
- You can review per-subject logs in `results/log_subject{n}.txt`.

---

## Citation

If you use this code or find it helpful, please cite the original paper:

> **Improved Domain Adaptation Network Based on Wasserstein Distance for Motor Imagery EEG Classification**  
> [Add full citation here]

---

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---

## Contact

For questions or collaborations, please open an issue or contact [HurDuhwan](https://github.com/HurDuhwan). 
