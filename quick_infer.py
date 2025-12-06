"""
Quick Vietnamese TTS Inference
Simple script to test the finetuned model.
"""

from infer_vietnamese import VietnameseTTS, find_latest_checkpoint

# Configuration
CHECKPOINT_DIR = "./checkpoints/neutts-vietnamese"
REF_AUDIO = "/mnt/d/tts_dataset/wavs/audio_000002.wav"  # Change to your reference audio
REF_TEXT = "Tôi tạm thời xin được ở tạm trong khách điếm của chú."  # Change to match your reference audio
TEXT = "Hôm nay trời đẹp quá"  # Text to synthesize
OUTPUT = "output_vietnamese.wav"

def main():
    print("=" * 60)
    print("QUICK VIETNAMESE TTS INFERENCE")
    print("=" * 60)
    
    # Find latest checkpoint
    print("\n[1/3] Finding latest checkpoint...")
    checkpoint = find_latest_checkpoint(CHECKPOINT_DIR)
    
    # Load model
    print("\n[2/3] Loading model...")
    tts = VietnameseTTS(
        checkpoint_path=checkpoint,
        device="cuda",  # Change to "cpu" if no GPU
        codec_device="cuda",
    )
    sentences = [
        "Chào Anh HOÀNG VĂN NAM, Em là Trang gọi từ công ty tài chính ép e CREDIT, rất vui được kết nối với Anh. ",
        "Cho em xin ít phút để trao đổi về hợp đồng Thẻ tín dụng của mình nha Anh?",
        "Dạ, Anh HOÀNG VĂN NAM có hợp đồng Thẻ tín dụng tại ép e CREDIT.",
        "Ngày đến hạn thanh toán là ngày mười hai , tuy nhiên , hợp đồng đang trễ hạn hai ngày với số tiền cần thanh toán là mười hai ngìn đồng.",
        "Dạ, em xin cung cấp thông tin tới Anh.",
        "Hôm nay là ngày thứ tư và Anh cần đi thanh toán chậm nhất ngày mười hai nha Anh.",
        "Anh thu xếp đóng tiền sớm trong hôm nay cho bên em nha.",
        "Anh cố gắng thu xếp thanh toán trước ngày mười hai qua mô mô, hoặc thế giới di động.",
        "Thanh toán xong giữ lại biên lai giúp em.",
        "Cảm ơn và chúc Anh một ngày tốt lành."
    ]

    # Synthesize
    print("\n[3/3] Synthesizing...")
    for i, sentence in enumerate(sentences):
        tts.synthesize(
            text=sentence,
            ref_audio_path=REF_AUDIO,
            ref_text=REF_TEXT,
            output_path=f"output_vietnamese_{i}.wav",
    )
    
    print(f"\n✅ Done! Audio saved to: {OUTPUT}")


if __name__ == "__main__":
    main()

