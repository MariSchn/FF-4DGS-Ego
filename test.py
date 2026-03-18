import torch
from diffsynth.auxiliary_models.worldmirror.models.models.worldmirror import WorldMirror

def test_new_setup():
    print("🚀 Inizializzazione modello con Hand Head...")
    
    # 1. Carica il modello (il freeze avviene nell'__init__ che abbiamo scritto)
    # In test.py, cambia questa riga:
    model = WorldMirror(enable_hand=True, freeze_backbone=True, enable_gs=False)
    model.eval() # Modalità inferenza
    
    # 2. Test caricamento pesi (strict=False è obbligatorio perché la testa è nuova!)
    print("📂 Caricamento pesi dal checkpoint...")
    ckpt_path = "models/reconstructor.ckpt"
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    # Gestione del dizionario se salvato con Lightning o simili
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    # Carichiamo i pesi del backbone ignorando quelli mancanti della testa
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"✅ Checkpoint caricato. Nota: {len(msg.missing_keys)} chiavi mancanti (normale per hand_head)")

    # 3. Dummy Forward Pass
    # Creiamo un'immagine finta [Batch=1, Frame=1, Canali=3, H=224, W=224]
    dummy_img = torch.randn(1, 1, 3, 224, 224)
    views = {'img': dummy_img}
    
    print("🏃 Esecuzione Forward Pass...")
    with torch.no_grad():
        preds = model(views, is_inference=True)

    # 4. Verifica finale
    if "hand_joints" in preds:
        print("🎉 SUCCESSO!")
        print(f"Forma output hand_joints: {preds['hand_joints'].shape}")
        # Ci aspettiamo qualcosa tipo [1, 1, 63, H_patch, W_patch] 
        # a seconda di come lavora DPTHead
    else:
        print("❌ ERRORE: 'hand_joints' non trovato nel dizionario preds.")

if __name__ == "__main__":
    test_new_setup()