"""
inference.py
=============
โหลดโมเดลที่เทรนแล้ว (บน HuggingFace Hub) มาทดสอบตอบคำถาม
ไฟล์นี้ "ไม่เทรน" — แค่โหลดมาใช้ทดสอบ/ตรวจสอบคำตอบเท่านั้น

ใช้ตรวจสอบว่า fine-tuning ที่ทำไปได้ผลดีแค่ไหน โดย:
1. ลองถามคำถามที่ "เคยเทรนไปแล้ว" (ดูว่าจำได้ตรงไหม)
2. ลองถามคำถามที่ "ไม่เคยเทรน" (ดูว่า generalize ได้ไหม หรือแค่ท่องจำ)

Run with:
    python inference.py
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ----------------------------
# Config
# ----------------------------
BASE_MODEL = "flax-community/gpt2-base-thai"
FINETUNED_MODEL = "MRlionman6/thai-traffic-law-gpt2"


def load_model():
    """โหลด base model + LoRA adapter ที่เทรนแล้วจาก HF Hub"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL)

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    # สำคัญ: ต้อง resize embeddings ให้ตรงกับตอนเทรน (train.py ทำขั้นตอนนี้
    # ก่อนผูก LoRA เช่นกัน) ไม่งั้นจะเจอ "size mismatch" ตอนโหลด adapter
    base_model.resize_token_embeddings(len(tokenizer))

    print("Loading fine-tuned LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL).to("cuda")

    return model, tokenizer


def ask(model, tokenizer, question, max_new_tokens=100):
    """ถามคำถามหนึ่งข้อ คืนค่าคำตอบที่โมเดล generate"""
    prompt = f"### คำถาม:\n{question}\n### คำตอบ:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.3,
        pad_token_id=tokenizer.pad_token_id,
    )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # ตัดเอาเฉพาะส่วนคำตอบ (หลัง "### คำตอบ:")
    answer = full_text.split("### คำตอบ:")[-1].strip()
    return answer


def main():
    model, tokenizer = load_model()

    print("\n" + "=" * 60)
    print("กลุ่มที่ 1: คำถามที่ 'เคยเทรน' ไปแล้ว (ดูว่าจำได้ตรงไหม)")
    print("=" * 60)
    trained_questions = [
        "ขับรถเร็วเกินกำหนดมีโทษอย่างไร",
        "ไม่คาดเข็มขัดนิรภัยผิดกฎหมายไหม",
        "ใบขับขี่รถยนต์ทำได้ตั้งแต่อายุเท่าไหร่",
    ]
    for q in trained_questions:
        answer = ask(model, tokenizer, q)
        print(f"\nQ: {q}")
        print(f"A: {answer}")

    print("\n" + "=" * 60)
    print("กลุ่มที่ 2: คำถามที่ 'ไม่เคยเทรน' (ดูว่า generalize ได้ไหม)")
    print("=" * 60)
    new_questions = [
        "ใบขับขี่หมดอายุกี่ปีต้องต่อใหม่?",   # ใกล้เคียงกับที่เคยเทรน แต่ถ้อยคำต่าง
        "ขับรถบรรทุกน้ำหนักเกินมีโทษอะไร",    # หัวข้อใหม่ ไม่เคยมีในข้อมูล
    ]
    for q in new_questions:
        answer = ask(model, tokenizer, q)
        print(f"\nQ: {q}")
        print(f"A: {answer}")

    print("\nDone.")


if __name__ == "__main__":
    main()
