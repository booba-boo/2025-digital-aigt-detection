# 2025-digital-aigt-detection

./data/original_data 에 대회 데이터 원본 파일(train.csv, test.csv, sample_submission.csv) 위치gi

**사용한 pretrained 오픈소스 모델 출처는 다음과 같습니다.**

**kanana**
모델명: kakaocorp/kanana-1.5-8b-instruct-2505
모델 url: https://huggingface.co/kakaocorp/kanana-1.5-8b-instruct-2505
citation:
@misc{kananallmteam2025kananacomputeefficientbilinguallanguage,
      title={Kanana: Compute-efficient Bilingual Language Models}, 
      author={Kanana LLM Team and Yunju Bak and Hojin Lee and Minho Ryu and Jiyeon Ham and Seungjae Jung and Daniel Wontae Nam and Taegyeong Eo and Donghun Lee and Doohae Jung and Boseop Kim and Nayeon Kim and Jaesun Park and Hyunho Kim and Hyunwoong Ko and Changmin Lee and Kyoung-Woon On and Seulye Baeg and Junrae Cho and Sunghee Jung and Jieun Kang and EungGyun Kim and Eunhwa Kim and Byeongil Ko and Daniel Lee and Minchul Lee and Miok Lee and Shinbok Lee and Gaeun Seo},
      year={2025},
      eprint={2502.18934},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.18934}, 
}

**EXAONE**
모델명: LGAI-EXAONE/EXAONE-3.5-32B-Instruct
모델 url: https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-32B-Instruct
citation:
@article{exaone-3.5,
  title={EXAONE 3.5: Series of Large Language Models for Real-world Use Cases},
  author={LG AI Research},
  journal={arXiv preprint arXiv:https://arxiv.org/abs/2412.04862},
  year={2024}
}

**GEMMA**
모델명: google/gemma-3-12b-it
모델 url: https://huggingface.co/google/gemma-3-12b-it
citation:
@article{gemma_2025,
    title={Gemma 3},
    url={https://goo.gle/Gemma3Report},
    publisher={Kaggle},
    author={Gemma Team},
    year={2025}
}

**QWEN3**
모델명: Qwen/Qwen3-14B
모델 url: https://huggingface.co/Qwen/Qwen3-14B
citation:
@misc{qwen3technicalreport,
      title={Qwen3 Technical Report}, 
      author={Qwen Team},
      year={2025},
      eprint={2505.09388},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.09388}, 
}
**사용 환경**

NVIDIA H100 80GB 환경에서 진행하였습니다.

**코드 흐름**

대회에서 제공된 기본 데이터 파일은 data/original_data 내부에 있습니다.
data/data_preprocess.ipynb 파일을 실행하면 data/kfold_csv 폴더에 전처리된 파일이 저장됩니다.

4개의 모델을 4fold stacking ensemble을 진행하기 때문에 학습 & 추론 파일이 총 16개입니다.
학습과 추론 파일을 별개로 제출하는 것을 권장하셨지만 최종 private score를 기록할 당시 코드는 학습과 추론 코드가 붙어있었고 따로 분리 후 연습 모드에서 제출해봤지만 점수가 같지 않고 오히려 올라가는 현상이 목격되었습니다.
private score 재현을 1순위로 두어 학습과 추론 코드가 전부 붙어 있는점 양해를 구합니다.

각 모델의 학습 & 추론 코드는 train&inference/{model_name}/{fold}/{model_name}_{fold}.ipynb 파일을 실행하시면 됩니다.
ex) gemma의 fold0 학습 추론 코드는 train&inference/gemma/fold0/gemma_fold0.ipynb 실행

16개의 학습 & 추론 코드를 돌리시면 ensemble/data/val_ensemble_folding 폴더에 16개의 validation data 추론 결과와
ensemble/data/val_ensemble_folding 폴더에 16개의 test data 추론 결과가 저장됩니다.

이후 ensemble 폴더의 ensemble.ipynb 파일을 실행하시면 최종 결과인 final_submission.csv가 저장됩니다.

제출한 zip 파일에는 
**코드 실행**
(권장) .ipynb 파일을 직접 열어서 실행
파일 경로 때문에 !ipython {절대경로} 와 같은 스크립트 실행으로 하면 에러가 날 수 있습니다.
.ipynb 파일을 직접 열어서 셀 단위로 실행하시는 것을 권장드립니다.

(비권장)!ipython {절대경로}와 같은 스크립트 실행
파일 경로를 맞추기 위해 반드시 실행 파일이 있는 폴더 위치에서 실행하셔야 에러 없이 작동합니다.
ex) gemma의 fold0 학습 추론 코드를 돌리기 위한 스크립트
    /root 에서 !ipython 2025-digital-aigt-detection/train&inference/gemma/fold0/gemma_fold0.ipynb -> X
    ../2025-digital-aigt-detection/train&inference/gemma/fold0 에서 !ipython gemma_fold0.ipynb    -> O








