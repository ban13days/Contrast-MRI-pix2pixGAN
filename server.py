import os, io, base64, uuid, time, zipfile, json
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import cv2

# ============================================================
# 설정
# ============================================================
CHECKPOINT_PATH = r"./0509/epoch_300.pt"   # ⚠️ 본인 경로로 수정
IMG_SIZE = 256

# 종양 강조 파라미터 (시각화용)
TUMOR_BOOST = 0.1
BACKGROUND_GRAY = 0.2

# 신뢰도/면적비율 임계값
MASK_THRESHOLD = 0.3          # sigmoid > 0.3 → 종양 픽셀 (낮춰서 가장자리도 포함)
BRAIN_THRESHOLD = 0.05        # 회색조 > 0.05 → 뇌 영역 (배경 제외)

# 뇌 영역 중앙 크롭 비율 (위아래 각 1/4, 좌우 각 1/8 제외)
CROP_TOP    = 0.25            # 위에서 25% 자름
CROP_BOTTOM = 0.25            # 아래에서 25% 자름
CROP_LEFT   = 0.125           # 왼쪽에서 12.5% 자름
CROP_RIGHT  = 0.125           # 오른쪽에서 12.5% 자름

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*60)
print("🚀 V-MAP 서버 시작 중...")
print(f"📁 체크포인트: {CHECKPOINT_PATH}")
print(f"💻 Device: {device}")
print("="*60)

# ============================================================
# 전처리 (학습/추론 코드와 동일)
# ============================================================
def apply_clahe(pil_img):
    """모델 입력용 CLAHE (시각화용 X)"""
    arr = np.array(pil_img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    arr = clahe.apply(arr)
    return Image.fromarray(arr)

tf_model = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,)),
])

tf_vis = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
])

def denorm(x):
    return (x * 0.5 + 0.5).clamp(0, 1)

# ============================================================
# 모델 정의 (학습 코드와 완전히 동일)
# ============================================================
def CBR(a, b):
    return nn.Sequential(
        nn.Conv2d(a, b, 3, 1, 1),
        nn.BatchNorm2d(b),
        nn.ReLU(),
    )

class MaskUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = nn.Sequential(CBR(1, 64), CBR(64, 64))
        self.p1 = nn.MaxPool2d(2)
        self.d2 = nn.Sequential(CBR(64, 128), CBR(128, 128))
        self.p2 = nn.MaxPool2d(2)
        self.d3 = nn.Sequential(CBR(128, 256), CBR(256, 256))
        self.p3 = nn.MaxPool2d(2)
        self.b  = nn.Sequential(CBR(256, 512), CBR(512, 512), nn.Dropout2d(0.5))
        self.u3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.c3 = nn.Sequential(CBR(512, 256), CBR(256, 256))
        self.u2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.c2 = nn.Sequential(CBR(256, 128), CBR(128, 128))
        self.u1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.c1 = nn.Sequential(CBR(128, 64), CBR(64, 64))
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.p1(d1))
        d3 = self.d3(self.p2(d2))
        b  = self.b(self.p3(d3))
        u3 = self.c3(torch.cat([self.u3(b),  d3], 1))
        u2 = self.c2(torch.cat([self.u2(u3), d2], 1))
        u1 = self.c1(torch.cat([self.u1(u2), d1], 1))
        return torch.sigmoid(self.out(u1))

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)

# ============================================================
# 종양 강조 (학습/추론 코드와 동일)
# ============================================================
def enhance_tumor_contrast(img, mask, tumor_boost=0.1, background_gray=0.2):
    mask = torch.clamp(mask, 0.0, 1.0)
    mask = mask ** 2
    bg = 1.0 - mask
    enhanced = img.clone()
    enhanced = enhanced * (1.0 - background_gray * bg) + 0.5 * background_gray * bg
    enhanced = enhanced + tumor_boost * mask
    enhanced = enhanced.clamp(0.0, 1.0)
    return enhanced

# ============================================================
# 모델 로드
# ============================================================
try:
    print("\n📦 체크포인트 로딩 중...")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"❌ 체크포인트 없음: {CHECKPOINT_PATH}")

    mask_net  = MaskUNet().to(device)
    generator = Generator().to(device)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    mask_net.load_state_dict(ckpt["mask_net"])
    generator.load_state_dict(ckpt["generator"])
    mask_net.eval()
    generator.eval()

    total = sum(p.numel() for p in mask_net.parameters()) + \
            sum(p.numel() for p in generator.parameters())
    print(f"📊 총 파라미터: {total:,}")
    print("✅ 모델 로드 완료")
    print("="*60 + "\n")

except Exception as e:
    print(f"\n❌ 모델 로딩 실패: {e}")
    import traceback; traceback.print_exc()
    exit(1)

# ============================================================
# 단일 이미지 추론 (배치 1장)
# ============================================================
def run_inference(pil_pre):
    """
    입력: PIL grayscale 이미지 (pre)
    출력: dict {
        'pre_b64', 'fake_b64', 'mask_b64',
        'area_ratio', 'confidence',
        'tumor_pixels', 'brain_pixels'
    }
    """
    # --- 텐서 변환 ---
    pre_clahe = apply_clahe(pil_pre)
    pre_tensor = tf_model(pre_clahe).unsqueeze(0).to(device)
    pre_vis_tensor = tf_vis(pil_pre).to(device)  # 시각화용 (CLAHE X)

    # --- 추론 ---
    with torch.no_grad():
        pred_mask = mask_net(pre_tensor)              # (1,1,H,W), sigmoid
        fake_post = generator(torch.cat([pre_tensor, pred_mask], dim=1))

    fake_denorm = denorm(fake_post)
    enhanced_fake = enhance_tumor_contrast(
        fake_denorm[0], pred_mask[0],
        tumor_boost=TUMOR_BOOST,
        background_gray=BACKGROUND_GRAY
    )

    # --- numpy 변환 ---
    pre_np  = pre_vis_tensor[0].cpu().numpy()              # (H,W) [0,1]
    fake_np = enhanced_fake[0].cpu().numpy()                    # (H,W) [0,1]
    mask_np = pred_mask[0, 0].cpu().numpy()                # (H,W) [0,1] sigmoid

    # --- 원본의 까만 배경을 AI 결과에도 적용 ---
    bg_mask = pre_np <= BRAIN_THRESHOLD
    fake_np = fake_np.copy()
    fake_np[bg_mask] = 0.0

    # --- 신뢰도: 마스크 영역(>0.5) 평균 sigmoid ---
    tumor_region = mask_np > MASK_THRESHOLD
    n_tumor = int(tumor_region.sum())
    if n_tumor > 0:
        confidence = float(mask_np[tumor_region].mean()) * 100.0
    else:
        confidence = 0.0

    # --- 면적 비율: (중앙 크롭 + 어두운 배경 제외) 영역 기준 ---
    H, W = pre_np.shape
    y0, y1 = int(H * CROP_TOP), int(H * (1 - CROP_BOTTOM))
    x0, x1 = int(W * CROP_LEFT), int(W * (1 - CROP_RIGHT))

    # 중앙 크롭 마스크 (해당 영역만 True)
    crop_mask = np.zeros_like(pre_np, dtype=bool)
    crop_mask[y0:y1, x0:x1] = True

    # 뇌 영역 = 중앙 크롭 ∩ 밝은 픽셀 (배경 제외)
    brain_region = crop_mask & (pre_np > BRAIN_THRESHOLD)
    n_brain = int(brain_region.sum())

    # 종양도 같은 영역 안에서만 세기 (공평한 비교)
    tumor_in_brain = tumor_region & brain_region
    n_tumor_in_brain = int(tumor_in_brain.sum())

    if n_brain > 0:
        area_ratio = (n_tumor_in_brain / n_brain) * 100.0
    else:
        area_ratio = 0.0

    # --- base64 인코딩 ---
    def to_b64(arr01):
        arr_u8 = (np.clip(arr01, 0, 1) * 255).astype(np.uint8)
        pil = Image.fromarray(arr_u8)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def to_b64_rgb(arr_rgb_u8):
        pil = Image.fromarray(arr_rgb_u8, mode="RGB")
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # --- 마스크 오버레이: 원본 위에 빨간색 종양 영역 표시 ---
    pre_u8 = (np.clip(pre_np, 0, 1) * 255).astype(np.uint8)
    overlay = np.stack([pre_u8, pre_u8, pre_u8], axis=-1)  # (H,W,3) RGB
    # 종양 영역만 빨강 강조 (반투명 알파블렌딩)
    alpha = 0.5
    overlay[tumor_region, 0] = (overlay[tumor_region, 0] * (1 - alpha) + 255 * alpha).astype(np.uint8)
    overlay[tumor_region, 1] = (overlay[tumor_region, 1] * (1 - alpha)).astype(np.uint8)
    overlay[tumor_region, 2] = (overlay[tumor_region, 2] * (1 - alpha)).astype(np.uint8)

    return {
        "pre":     to_b64(pre_np),
        "fake":    to_b64(fake_np),
        "mask":    to_b64(mask_np),
        "overlay": to_b64_rgb(overlay),
        "area_ratio": round(area_ratio, 2),
        "confidence": round(confidence, 2),
        "tumor_pixels": n_tumor,
        "brain_pixels": n_brain,
    }

# ============================================================
# Flask
# ============================================================
app = Flask(__name__, static_folder=None)
CORS(app)

# 세션별 결과 캐시 (간단한 메모리 저장)
SESSIONS = {}  # session_id -> {'patient_id', 'age', 'results': [...]}

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """
    폴더 단위 업로드 → 전체 이미지 추론 후 세션에 캐시
    multipart/form-data:
      - files[]: 이미지들 (pre 이미지만)
      - patient_id: str
      - age: str
    """
    try:
        patient_id = request.form.get("patient_id", "unknown")
        age = request.form.get("age", "")
        files = request.files.getlist("files")

        if not files:
            return jsonify({"error": "No files uploaded"}), 400

        print(f"\n{'='*60}")
        print(f"📥 환자 ID: {patient_id} / 나이: {age} / 이미지: {len(files)}장")

        results = []
        t0 = time.time()

        for i, f in enumerate(files):
            try:
                img = Image.open(f).convert("L")
                res = run_inference(img)
                res["filename"] = f.filename
                res["index"] = i
                results.append(res)
                if (i+1) % 10 == 0 or i+1 == len(files):
                    print(f"   ⏳ 진행 {i+1}/{len(files)}")
            except Exception as e:
                print(f"   ⚠️ {f.filename} 처리 실패: {e}")
                results.append({
                    "filename": f.filename,
                    "index": i,
                    "error": str(e),
                })

        elapsed = time.time() - t0
        print(f"✅ 추론 완료: {elapsed:.1f}s ({elapsed/len(files):.2f}s/장)")

        # 세션 저장
        session_id = str(uuid.uuid4())
        SESSIONS[session_id] = {
            "patient_id": patient_id,
            "age": age,
            "n_images": len(results),
            "results": results,
            "created_at": time.time(),
        }

        # 응답: 메타 + 첫 장만 (나머지는 슬라이더에서 /result로)
        return jsonify({
            "session_id": session_id,
            "patient_id": patient_id,
            "age": age,
            "n_images": len(results),
            "first_result": results[0] if results else None,
        })

    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/result/<session_id>/<int:index>", methods=["GET"])
def get_result(session_id, index):
    """슬라이더에서 N번 이미지 결과 가져오기"""
    sess = SESSIONS.get(session_id)
    if sess is None:
        return jsonify({"error": "Session not found"}), 404
    if index < 0 or index >= len(sess["results"]):
        return jsonify({"error": "Index out of range"}), 400

    return jsonify({
        "patient_id": sess["patient_id"],
        "age": sess["age"],
        "n_images": sess["n_images"],
        "index": index,
        "result": sess["results"][index],
    })

@app.route("/session/<session_id>/summary", methods=["GET"])
def session_summary(session_id):
    """다운로드/리포트용 - 모든 메타 정보 (이미지 제외)"""
    sess = SESSIONS.get(session_id)
    if sess is None:
        return jsonify({"error": "Session not found"}), 404

    summary = []
    for r in sess["results"]:
        if "error" in r:
            summary.append({"filename": r.get("filename"), "error": r["error"]})
        else:
            summary.append({
                "filename": r["filename"],
                "index": r["index"],
                "area_ratio": r["area_ratio"],
                "confidence": r["confidence"],
            })

    return jsonify({
        "patient_id": sess["patient_id"],
        "age": sess["age"],
        "n_images": sess["n_images"],
        "summary": summary,
    })

@app.route("/session/<session_id>/download_zip", methods=["GET"])
def download_zip(session_id):
    """모든 결과를 ZIP으로 다운로드"""
    sess = SESSIONS.get(session_id)
    if sess is None:
        return jsonify({"error": "Session not found"}), 404

    print(f"\n📦 ZIP 생성 중... (이미지 {sess['n_images']}장)")
    t0 = time.time()

    # 메모리 안에서 ZIP 생성
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # 통계 요약 JSON 추가
        summary = []
        for r in sess["results"]:
            if "error" in r:
                summary.append({"filename": r.get("filename"), "error": r["error"]})
            else:
                summary.append({
                    "filename": r["filename"],
                    "index": r["index"],
                    "area_ratio": r["area_ratio"],
                    "confidence": r["confidence"],
                })
        meta = {
            "patient_id": sess["patient_id"],
            "age": sess["age"],
            "n_images": sess["n_images"],
            "summary": summary,
        }
        zf.writestr("summary.json", json.dumps(meta, indent=2, ensure_ascii=False))

        # 이미지 추가 (AI 결과만)
        digits = len(str(sess["n_images"]))  # 자릿수 (예: 1000장 → 4자리)
        for r in sess["results"]:
            if "error" in r:
                continue
            idx_str = str(r["index"] + 1).zfill(digits)
            try:
                fake_bytes = base64.b64decode(r["fake"])
                zf.writestr(f"{idx_str}_ai_result.png", fake_bytes)
            except Exception as e:
                print(f"   ⚠️ {idx_str} 이미지 인코딩 실패: {e}")

    buf.seek(0)
    elapsed = time.time() - t0
    print(f"✅ ZIP 생성 완료: {elapsed:.1f}s, 크기: {len(buf.getvalue())/1024/1024:.1f}MB")

    filename = f"VMAP_{sess['patient_id']}_results.zip"
    return send_file(
        buf,
        mimetype="application/zip",
        as_attachment=True,
        download_name=filename,
    )

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "device": str(device),
        "img_size": IMG_SIZE,
        "n_sessions": len(SESSIONS),
    })

# ============================================================
# 정적 UI 서빙 (선택: 같은 서버에서 UI도 같이)
# ============================================================
@app.route("/")
def ui_index():
    return send_from_directory("static/ui_B", "index.html")

@app.route("/ui_A/<path:filename>")
def ui_a(filename):
    return send_from_directory("static/ui_A", filename)

@app.route("/ui_B/<path:filename>")
def ui_b(filename):
    return send_from_directory("static/ui_B", filename)

if __name__ == "__main__":
    print("\n🌐 V-MAP 서버 시작")
    print(f"📍 http://127.0.0.1:5000")
    print(f"📍 UI (재구성):    http://127.0.0.1:5000/ui_A/index.html")
    print(f"📍 UI (픽셀 고정): http://127.0.0.1:5000/ui_B/index.html")
    print(f"📍 API:")
    print(f"     POST /predict_batch")
    print(f"     GET  /result/<session_id>/<index>")
    print(f"     GET  /session/<session_id>/summary")
    print(f"     GET  /health")
    print("="*60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
