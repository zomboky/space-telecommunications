import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------
# PARAMÈTRES
# -------------------------
MODEL_DIR = "phi3_fen_qLoRA"  # dossier où tu as sauvegardé ton modèle

# -------------------------
# CHARGEMENT TOKENIZER + MODÈLE
# -------------------------
print("Chargement du modèle...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,       # or float32 if CPU
    device_map="auto",               # GPU si dispo
    trust_remote_code=True
)

# -------------------------
# FONCTION DE GÉNÉRATION
# -------------------------
def ask_model(prompt, max_new_tokens=128):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# -------------------------
# TEST SIMPLE
# -------------------------
print("\n=== TEST DU MODÈLE ===")

prompt = "Évalue la position suivante en notation FEN : rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
response = ask_model(prompt)

print("\nRéponse du modèle :\n")
print(response)
