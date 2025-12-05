import os
import csv
from datasets import load_dataset, Audio
from tqdm import tqdm
import soundfile as sf

# ========== CONFIG ==========
HF_TOKEN = "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"
HF_DATASET = "JayLL13/dolly-audio-Mature-Woman"     # ví dụ: "librispeech_asr"
HF_SPLIT = "train"                       # train / validation / test
PARENT_DIR = "/mnt/d/tts_dataset/"                   # folder output

AUDIO_COLUMN = "audio"                   # cột audio trong HF dataset
TEXT_COLUMN = "text"                     # cột transcript
# ============================


def main():
    os.environ["HF_TOKEN"] = HF_TOKEN

    # Tạo thư mục output
    wavs_dir = os.path.join(PARENT_DIR, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)

    metadata_path = os.path.join(PARENT_DIR, "metadata.csv")
    writer = csv.writer(open(metadata_path, "w", newline="", encoding="utf-8"),
                        delimiter="|")

    print(f"📥 Loading dataset {HF_DATASET} (streaming)...")

    # 🔥 FIX QUAN TRỌNG: ép audio decode sang numpy
    dataset = load_dataset(
        HF_DATASET,
        split=HF_SPLIT,
        streaming=True,
        token=HF_TOKEN
    ).cast_column(AUDIO_COLUMN, Audio(decode=True))

    print("🎧 Bắt đầu tải và lưu audio...")

    idx = 1
    for sample in tqdm(dataset, desc="Processing"):
        audio_obj = sample[AUDIO_COLUMN]

        # HF trả về:
        # audio_obj["array"]  (numpy array)
        # audio_obj["sampling_rate"]

        array = audio_obj["array"]
        sr = audio_obj["sampling_rate"]

        # Tên file output
        filename = f"audio_{idx:06d}.wav"
        filepath = os.path.join(wavs_dir, filename)

        # Save bằng soundfile
        sf.write(filepath, array, sr)

        # Transcript
        text = sample[TEXT_COLUMN].replace("\n", " ").strip()

        # Ghi metadata
        writer.writerow([filename, text])

        idx += 1

    print("✅ DONE! Saved to:", PARENT_DIR)


if __name__ == "__main__":
    main()