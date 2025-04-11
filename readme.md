## Update Log (Refactor branch)

피드백
- Python 3.10 필요함을 명시하면 좋을 것 같음
    - Python 3.11에서 일부 패키지가 설치되지 않음

수정사항
- `requirements.txt` 에서 Windows 환경에서 설치되지 않는 패키지 일부가 있어 주석 처리함
- 코드 리뷰를 위해 Zenodo 데이터 다운로드 후 `/K-Emoworker` 경로에 올려서 작업하였고 gitignore에 추가함

- `Dataset Records.ipynb`
  - 전체 주석 영어로 변경
  - PHQ, PSS 등 설문 항목 관련 컬럼 접근을 하드코딩 방식에서 → `df.filter(like='PHQ', axis=1)` 방식으로 수정하여 확장성과 유지보수성 개선
  - 전체적인 시각화는 UID 1 example만 출력되도록 제한
  - 레이블 시각화 코드를 함수(plot_single_participant)로 리팩토링하여 재사용 가능하도록 구성
  - 센서 시각화 함수에서 `plt.suptitle` 대신 `axes[0].set_title`을 활용하여 subplot 개수와 관계없이 일정한 여백을 유지할 수 있도록 개선

- `Label_Analysis.ipynb`
  - 파일명 오타 수정 (기존 Lable)
  - 전체 주석 영어로 변경

- `ML_analysis.ipynb`
  - 기존의 `mean_audio`, `min_audio`, `max_audio`, `std_audio` 4개 함수를 `aggregate_audio()` 함수 하나로 통합하여 메모리 효율 및 처리 속도 개선
    - 이에 따라 `model_utils.py`에도 4개 함수 삭제하고 `aggregate_audio` 함수로 대체
    - `out_audio_customer_mean, out_audio_customer_min, out_audio_customer_max, out_audio_customer_std = aggregate_audio(audio_customer_data, audio_window, slide)`
  - `run_experiments()` 함수
    - 실행 시간이 길어 현재는 Decision Tree 모델까지 실행되는 것을 확인함
    - 코드를 각 기능 단위로 더 쪼개어 여러 개의 함수를 만들어서 실행하는 것도 좋을 것 같음 -> 필요시 작업 가능
  - Code (for Hide and seek)부터는 실행하지 않음