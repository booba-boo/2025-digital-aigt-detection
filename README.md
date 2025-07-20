# 2025-digital-aigt-detection

## 데이터 경로

`./data/original_data` 폴더에 대회 데이터 원본 파일이 위치합니다:

- `train.csv`
- `test.csv`
- `sample_submission.csv`

---

## 사용한 Pretrained 오픈소스 모델

### **1. Kanana**
- **모델명**: `kakaocorp/kanana-1.5-8b-instruct-2505`  
- **모델 URL**: [https://huggingface.co/kakaocorp/kanana-1.5-8b-instruct-2505](https://huggingface.co/kakaocorp/kanana-1.5-8b-instruct-2505)  
- **Citation**:
```bibtex
@misc{kananallmteam2025kananacomputeefficientbilinguallanguage,
      title={Kanana: Compute-efficient Bilingual Language Models}, 
      author={Kanana LLM Team and Yunju Bak and Hojin Lee and Minho Ryu and Jiyeon Ham and Seungjae Jung and Daniel Wontae Nam and Taegyeong Eo and Donghun Lee and Doohae Jung and Boseop Kim and Nayeon Kim and Jaesun Park and Hyunho Kim and Hyunwoong Ko and Changmin Lee and Kyoung-Woon On and Seulye Baeg and Junrae Cho and Sunghee Jung and Jieun Kang and EungGyun Kim and Eunhwa Kim and Byeongil Ko and Daniel Lee and Minchul Lee and Miok Lee and Shinbok Lee and Gaeun Seo},
      year={2025},
      eprint={2502.18934},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.18934}, 
}
````

---

### **2. EXAONE**

* **모델명**: `LGAI-EXAONE/EXAONE-3.5-32B-Instruct`
* **모델 URL**: [https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-32B-Instruct](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-32B-Instruct)
* **Citation**:

```bibtex
@article{exaone-3.5,
  title={EXAONE 3.5: Series of Large Language Models for Real-world Use Cases},
  author={LG AI Research},
  journal={arXiv preprint arXiv:https://arxiv.org/abs/2412.04862},
  year={2024}
}
```

---

### **3. GEMMA**

* **모델명**: `google/gemma-3-12b-it`
* **모델 URL**: [https://huggingface.co/google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it)
* **Citation**:

```bibtex
@article{gemma_2025,
    title={Gemma 3},
    url={https://goo.gle/Gemma3Report},
    publisher={Kaggle},
    author={Gemma Team},
    year={2025}
}
```

---

### **4. QWEN3**

* **모델명**: `Qwen/Qwen3-14B`
* **모델 URL**: [https://huggingface.co/Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B)
* **Citation**:

```bibtex
@misc{qwen3technicalreport,
      title={Qwen3 Technical Report}, 
      author={Qwen Team},
      year={2025},
      eprint={2505.09388},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.09388}, 
}
```

---

## 사용 환경

* **NVIDIA H100 80GB** 환경에서 실험을 진행했습니다.
* **Jupyter Lab (python 3.11)**
* **OS: Ubuntu 22.04** 
---

## 코드 흐름 요약

1. **원본 데이터 경로**:
   `data/original_data` 폴더에 위치합니다.

2. **전처리 실행**:
   `data/data_preprocess.ipynb`를 실행하면,
   → `data/kfold_csv` 폴더에 전처리된 파일들이 저장됩니다.

3. **학습 및 추론 구조**:

   * 4개의 모델 × 4-fold stacking ensemble을 수행합니다.
   * 학습과 추론을 합쳐 총 **16개의 노트북**이 존재합니다.
   * 학습과 추론을 **분리하여 제출하는 것을 권장**하였으나, 최종 private score 기준에서는 학습과 추론 코드가 **함께 붙어있는 코드**의 점수가 더 높게 나왔습니다.
   * 따라서 `train + inference` 코드가 **하나의 `.ipynb` 파일에 포함**되어 있으며, 이 점 양해 부탁드립니다.

4. **각 모델의 실행 파일 경로**:

   ```
   train&inference/{model_name}/{fold}/{model_name}_{fold}.ipynb
   ```

   예시:

   ```
   gemma의 fold0 학습/추론 → train&inference/gemma/fold0/gemma_fold0.ipynb
   ```

5. **Ensemble 결과**:

   * 위 16개의 학습/추론 코드를 모두 실행하면,
     → `ensemble/data/val_ensemble_folding` 폴더에

     * **validation 추론 결과 16개**
     * **test 추론 결과 16개**가 저장됩니다.

6. **최종 제출 파일 생성**:

   * `ensemble/ensemble.ipynb` 실행
     → `final_submission.csv` 파일이 생성됩니다.

---

## 코드 실행 방식 안내

### 권장 방식: `.ipynb` 파일을 직접 열어 실행

* **Colab이나 JupyterLab에서 파일을 직접 열고 셀 단위로 실행**하세요.
* 이유: 경로 문제로 인해 `!ipython {절대경로}` 실행은 에러가 발생할 수 있습니다.

### 비권장 방식: `!ipython` 스크립트 실행

* 반드시 해당 `.ipynb` 파일이 있는 폴더로 이동한 후 실행해야 정상 작동합니다.
* 예시:

  ```bash
  # ❌ 잘못된 예
  /root 에서:
  !ipython 2025-digital-aigt-detection/train&inference/gemma/fold0/gemma_fold0.ipynb

  # ✅ 올바른 예
  cd 2025-digital-aigt-detection/train&inference/gemma/fold0
  !ipython gemma_fold0.ipynb
  ```
