import os
import json
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from diffsynth.utils.auxiliary import load_video
from torchvision.transforms import functional as TVF
import argparse

# Importiamo la classe WorldMirror dal tuo progetto
from diffsynth.auxiliary_models.worldmirror.models.models.worldmirror import WorldMirror

# ------------------------------------------------------------------
# 1. Dataset per HOT3D (Parsing dei file .jsonl di MANO)
# ------------------------------------------------------------------
class HOT3DHandDataset(Dataset):
    def __init__(self, seq_path, num_frames=16, res=(224, 224)):
        self.video_path = os.path.join(seq_path, "video_main_rgb.mp4")
        self.jsonl_path = os.path.join(seq_path, "hand_data/mano_hand_pose_trajectory.jsonl")
        self.res = res
        self.num_frames = num_frames
        
        self.gt_data = []
        if not os.path.exists(self.jsonl_path):
            raise FileNotFoundError(f"❌ Errore: Non trovo il file annotazioni in {self.jsonl_path}")

        print(f"📖 Caricamento annotazioni da: {self.jsonl_path}")
        with open(self.jsonl_path, 'r') as f:
            lines = f.readlines()
            # Leggiamo i primi 'num_frames' per questo test di overfitting
            for i in range(min(len(lines), self.num_frames)):
                data = json.loads(lines[i])
                hand0 = data["hand_poses"].get("0", {})
                if hand0:
                    pos = torch.tensor(hand0["wrist_xform"]["t_xyz"]) # 3
                    rot = torch.tensor(hand0["wrist_xform"]["q_wxyz"]) # 4
                    pose = torch.tensor(hand0["pose"]) # 15
                    vec = torch.cat([pos, rot, pose])
                    # Padding a 63 per matchare l'output della hand_head
                    padding = torch.zeros(63 - vec.shape[0])
                    self.gt_data.append(torch.cat([vec, padding]))
                else:
                    self.gt_data.append(torch.zeros(63))

    def __len__(self):
        return 1 

    def __getitem__(self, idx):
        # Caricamento video tramite utility interna
        pil_images = load_video(self.video_path, num_frames=self.num_frames, resolution=self.res)
        img_tensor = torch.stack([TVF.to_tensor(img) for img in pil_images], dim=0) # [S, 3, H, W]
        gt_joints = torch.stack(self.gt_data) # [S, 63]
        return {"img": img_tensor, "gt_joints": gt_joints}


# ------------------------------------------------------------------
# 2. Funzione Principale di Training
# ------------------------------------------------------------------
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_path", required=True, help="Percorso alla sequenza HOT3D")
    parser.add_argument("--checkpoint", default="models/reconstructor.ckpt", help="Checkpoint originale")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_frames", type=int, default=16)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- INIZIALIZZAZIONE MODELLO ---
    print(f"📦 Inizializzazione WorldMirror con Hand Head...")
    # Il freeze_backbone=True è gestito nel tuo __init__ modificato
    model = WorldMirror(enable_hand=True, freeze_backbone=True)
    
    print(f"📂 Caricamento pesi (Strict=False per ignorare la nuova testa)...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint.get("reconstructor", checkpoint))
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"✅ Modello pronto. Parametri mancanti (Hand Head): {len(missing)}")

    # Lasciamo il modello in Float32 per la stabilità delle LayerNorm
    model.to(device)
    model.train() 

    # --- OTTIMIZZATORE ---
    # Ottimizziamo solo i parametri che richiedono gradienti (Hand Head)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params, lr=args.lr)

    # --- DATALOADER ---
    dataset = HOT3DHandDataset(args.seq_path, num_frames=args.num_frames)
    dataloader = DataLoader(dataset, batch_size=1)

    print(f"🚀 Inizio Training su {device} con Automatic Mixed Precision (AMP)...")

    for epoch in range(args.epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            
            # Dati in Float32 (AMP gestirà la conversione interna)
            imgs = batch["img"].to(device) 
            gt = batch["gt_joints"].to(device)

            views = {
                "img": imgs, # [1, S, 3, H, W]
                "is_target": torch.zeros((1, args.num_frames), dtype=torch.bool, device=device),
                "timestamp": torch.arange(0, args.num_frames, device=device).unsqueeze(0)
            }

            # --- MIXED PRECISION FORWARD PASS ---
            # Risolve l'errore: expected scalar type Float but found BFloat16
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                preds = model(views, is_inference=False, use_motion=False)
                # Calcolo della Loss (MSE) tra i 63 valori predetti e il GT
                loss = F.mse_loss(preds["hand_joints"], gt)
            
            # Backpropagation (fuori dall'autocast)
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0:
                print(f"🔥 Epoch {epoch:03d}/{args.epochs} | Loss: {loss.item():.6f}")

    # --- SALVATAGGIO FINALE ---
    output_weights = "hand_head_weights.pt"
    torch.save(model.hand_head.state_dict(), output_weights)
    print(f"💾 Successo! Pesi della Hand Head salvati in: {output_weights}")

if __name__ == "__main__":
    train()