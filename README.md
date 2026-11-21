# Kimodo Maker

프롬프트로 Kimodo 모션을 생성하고 브라우저에서 SOMA 스켈레톤과 내 VRM에 적용해 보는 로컬 작업 도구다. 테이크별로 프롬프트, 길이, 추론 스텝, 시드와 제약을 보관한다. 설정을 다시 불러오거나 시드를 비워 다른 결과를 뽑을 수 있다.

## 구조

프론트엔드와 백엔드가 나뉘어 있다.

- `backend/` — Python API 서버. `server.py`가 `/api` 엔드포인트만 제공하고 `engine.py`로 GPU 추론을 돌린다. 테이크는 `backend/takes/`에 저장된다.
- `frontend/` — Vite + three.js 편집기. 빌드 결과는 `frontend/dist/`에 나온다.

프론트엔드는 기본적으로 같은 origin의 `/api`를 호출한다. 개발 중에는 Vite가 `/api`를 로컬 백엔드로 전달하고, 분리된 origin에서 서빙할 때는 환경 변수 `VITE_API_BASE`로 백엔드 주소를 지정한다.

## 실행

NVIDIA GPU와 CUDA용 PyTorch, Node.js 22.12 이상, Python 3.10 이상이 필요하다. Kimodo의 모션 후처리 모듈은 C++ 컴파일러와 CMake로 빌드한다.

### 백엔드 (Windows 네이티브)

Kimodo의 빌드 스크립트에 Windows 경로가 있다. 아래는 Python 3.10.11과 Visual Studio 2022 기준이다. Visual Studio Build Tools의 **C++를 사용한 데스크톱 개발** 워크로드와 Windows SDK를 설치하고 PowerShell을 연다. 다른 Visual Studio 버전을 쓰면 `cmake --help`에 나온 해당 버전의 생성기 이름을 사용한다. CMake는 Eigen과 pybind11을 찾지 못하면 내려받는다.

```powershell
git clone https://github.com/acidsound/kimodoMaker.git
cd kimodoMaker/backend
py -3.10 -m venv .venv
$python = ".\.venv\Scripts\python.exe"
& $python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
& $python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print(torch.cuda.get_device_name(0))"

$env:CMAKE_GENERATOR = "Visual Studio 17 2022"
& $python -m pip install -r requirements.txt
& $python -c "from motion_correction import motion_postprocess; print('MotionCorrection OK')"
```

[Meta Llama 3 8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) 접근 권한을 얻은 뒤 같은 PowerShell에서 실행한다.

```powershell
.\.venv\Scripts\hf.exe auth login
$env:TEXT_ENCODER_MODE = "local"
& $python server.py --host 127.0.0.1 --port 8765
```

`--host 0.0.0.0`을 주면 LAN 등 원격 접속을 허용한다. 첫 생성에서 모델과 텍스트 인코더를 내려받고 이후에는 서버가 모델을 재사용한다. NVIDIA의 [Kimodo 빠른 시작](https://research.nvidia.com/labs/sil/projects/kimodo/docs/getting_started/quick_start.html)은 모델과 텍스트 인코더를 GPU에서 함께 실행할 때 약 17 GB의 VRAM이 필요하다고 안내한다. VRAM이 부족하면 서버 실행 전에 환경 변수 `TEXT_ENCODER_DEVICE`를 `cpu`로 설정해 텍스트 인코더를 CPU로 옮길 수 있다.

Windows 네이티브 환경(RTX 4090, torch 2.10.0+cu128, 로컬 LLM2Vec 텍스트 인코더)에서 모델 로드부터 모션 생성, 파일 내보내기까지 전체 흐름을 확인했다. Llama 3 8B 기반이라 모델 접근 권한은 필수이고 권한 승인 후에는 별도 설정 없이 내려받기가 진행된다.

### 백엔드 (Linux / Ubuntu WSL2)

```bash
git clone https://github.com/acidsound/kimodoMaker.git
cd kimodoMaker/backend
sudo apt install build-essential cmake libeigen3-dev python3-dev python3-venv
python3 -m venv .venv
source .venv/bin/activate
```

1. [PyTorch 설치 페이지](https://pytorch.org/get-started/locally/)에서 Linux용 CUDA 빌드를 선택하고 가상 환경에 설치한다. `python -c "import torch; print(torch.cuda.is_available())"`가 `True`여야 한다.
2. Hugging Face에서 [Meta Llama 3 8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) 접근 권한을 얻고 읽기 토큰을 준비한다.

```bash
python -m pip install -r requirements.txt
hf auth login
TEXT_ENCODER_MODE=local python server.py
```

### 프론트엔드

```bash
cd frontend
npm ci
npm run dev
```

개발 서버는 `http://localhost:5173`을 열고 `/api` 요청을 `http://127.0.0.1:8765`로 전달한다.

배포할 때는 빌드 후 미리보기 서버를 쓴다. 백엔드와 origin이 다르므로 `VITE_API_BASE`로 API 주소를 지정한다.

```bash
VITE_API_BASE=http://localhost:8765 npm run build
npm run preview
```

`VITE_API_BASE`를 주지 않고 빌드하면 같은 origin의 `/api`를 호출하는 상태로 빌드된다.

### Tailscale 원격 접속

백엔드와 프론트엔드 각각 `tailscale serve`를 연결한다. tailnet 안쪽에서만 접속되고 인터넷에 노출되지 않는다. 백엔드는 서버를 띄운 채로:

```bash
tailscale serve --bg --https 8766 http://127.0.0.1:8765
```

프론트엔드는 `VITE_API_BASE`를 tailnet의 백엔드 주소로 빌드한 뒤:

```bash
cd frontend
VITE_API_BASE=https://<머신 이름>.<tailnet>.ts.net:8766 npm run build
npm run preview
tailscale serve --bg --https 8443 http://127.0.0.1:5173
```

같은 tailnet의 기기에서 `https://<머신 이름>.<tailnet>.ts.net:8443`을 열면 된다. 백엔드는 기본적으로 모든 origin의 CORS 요청을 허용하고, `--cors-origin`으로 허용 origin을 제한할 수 있다. 중지는 `tailscale serve --https=<포트> off`.

## 사용

1. 프롬프트와 길이를 입력하고 **모션 생성**을 누른다. 예를 들어 `A person takes two steps, plants both feet, and turns to the left.`처럼 한 테이크에 포함할 움직임을 적는다. 기존의 [Kimodo constraints.json](https://research.nvidia.com/labs/sil/projects/kimodo/docs/user_guide/constraints.html)이 있으면 함께 불러올 수 있다.
2. **VRM 불러오기**로 로컬 파일을 연다. SOMA 골격과 VRM을 나란히 놓고 재생하거나 프레임을 직접 이동해 비교할 수 있다. 루트 이동을 끄면 미리보기에서 수평 이동을 고정하고 높이 변화는 유지한다. VRM은 브라우저에서만 읽으며 파일을 서버로 보내지 않는다.
3. 다른 테이크를 선택하거나 **설정 불러오기**로 기존 프롬프트를 수정한다. 생성한 테이크는 기본적으로 `backend/takes/`에 저장되며 재시작 후에도 목록에 표시된다. `--data-dir`로 저장 위치를 지정할 수 있다. 목록에서 테이크를 삭제하거나 화살표로 순서를 바꿀 수 있고, 순서는 서버에 저장된다.

## 내보내기

세 파일은 같은 추론 결과에서 생성한다. 미리보기의 아바타 비율, 시점, 재생 속도, 루트 이동 토글과 무관하게 SOMA 원본 좌표를 저장한다.

| 파일 | 내용 |
| --- | --- |
| JSON | `kimodo-maker.soma-motion.v1`. SOMA-77 계층, 정지 자세의 로컬 오프셋, 재생 FPS, 미터 단위의 프레임별 골반 위치, 관절 순서로 펼친 로컬 쿼터니언 `xyzw`, 생성 설정. 축은 Y-up, +Z 전방. 골반의 정지 자세 높이는 발 관절 좌표를 기준으로 잡는다. |
| BVH | Kimodo 공식 내보내기 함수로 만든 SOMA-77 표준 T-pose BVH. BVH 계층에는 호환성을 위한 `Root` 노드가 추가되며 길이 단위는 센티미터다. |
| NPZ | Kimodo 공식 내보내기 함수로 저장한 모션 배열. Kimodo 데모에서 다시 불러올 수 있다. |

JSON은 프레임을 재생하거나 다른 리그에 리타기팅하는 데 쓸 수 있다. MotionBricks Live의 `kimodo_runtime.json`을 만들려면 액션 이름과 구간, 기물의 기준점과 접촉 정보 등 그 런타임이 요구하는 데이터를 별도로 구성해야 한다. VRM 미리보기는 휴머노이드 본 회전과 루트 이동을 전달한다. 캐릭터 비율이 다를 때 손·발·골반 접촉을 고정하려면 해당 장면에서 IK와 접촉 제약을 계산해야 한다.

## 개발

백엔드와 프론트엔드를 각각 실행한다. 프론트엔드 개발 서버는 `/api`를 로컬 백엔드로 전달한다.

```bash
cd backend && python server.py
cd frontend && npm run dev
```

테스트는 각 디렉터리에서 실행한다.

```bash
cd frontend && npm test
cd backend && python -m unittest discover -s tests
```
