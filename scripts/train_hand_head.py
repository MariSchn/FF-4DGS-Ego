import os
import json
import yaml
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from diffsynth.utils.auxiliary import load_video
from torchvision.transforms import functional as TVF
import argparse

# Import the WorldMirror class from the project
from diffsynth.auxiliary_models.worldmirror.models.models.worldmirror import WorldMirror


# ------------------------------------------------------------------
# 1. Dataset for HOT3D (Parsing MANO .jsonl files)
# ------------------------------------------------------------------
class HOT3DHandDataset(Dataset):
    def __init__(self, seq_path, num_frames=16, res=(224, 224)):
        self.video_path = os.path.join(seq_path, "video_main_rgb.mp4")
        self.jsonl_path = os.path.join(seq_path, "hand_data/mano_hand_pose_trajectory.jsonl")
        self.res = res
        self.num_frames = num_frames

        self.gt_data = []
        if not os.path.exists(self.jsonl_path):
            raise FileNotFoundError(f"❌ Error: Cannot find annotations file at {self.jsonl_path}")

        print(f"📖 Loading annotations from: {self.jsonl_path}")
        with open(self.jsonl_path, 'r') as f:
            lines = f.readlines()
            # Read the first 'num_frames' for this overfitting test
            for i in range(min(len(lines), self.num_frames)):
                data = json.loads(lines[i])
                hand0 = data["hand_poses"].get("0", {})
                if hand0:
                    pos = torch.tensor(hand0["wrist_xform"]["t_xyz"]) # 3
                    rot = torch.tensor(hand0["wrist_xform"]["q_wxyz"]) # 4
                    pose = torch.tensor(hand0["pose"]) # 15
                    vec = torch.cat([pos, rot, pose])
                    # Padding to 63 to match the hand_head output
                    padding = torch.zeros(63 - vec.shape[0])
                    self.gt_data.append(torch.cat([vec, padding]))
                else:
                    self.gt_data.append(torch.zeros(63))

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        # Load video using internal utility
        pil_images = load_video(self.video_path, num_frames=self.num_frames, resolution=self.res)
        img_tensor = torch.stack([TVF.to_tensor(img) for img in pil_images], dim=0) # [S, 3, H, W]
        gt_joints = torch.stack(self.gt_data) # [S, 63]
        return {"img": img_tensor, "gt_joints": gt_joints}


# ------------------------------------------------------------------
# 2. Main Training Function
# ------------------------------------------------------------------
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_hand_head.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    data_cfg     = cfg["data"]
    model_cfg    = cfg["model"]
    training_cfg = cfg["training"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- MODEL INITIALIZATION ---
    print(f"📦 Initializing WorldMirror with Hand Head...")
    # freeze_backbone=True is handled in the modified __init__
    model = WorldMirror(enable_hand=model_cfg["enable_hand"], freeze_backbone=model_cfg["freeze_backbone"])

    print(f"📂 Loading weights (Strict=False to ignore the new head)...")
    checkpoint = torch.load(model_cfg["checkpoint"], map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint.get("reconstructor", checkpoint))

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"✅ Model ready. Missing parameters (Hand Head): {len(missing)}")

    # Keep the model in Float32 for LayerNorm stability
    model.to(device)
    model.train()

    # --- OPTIMIZER ---
    # Only optimize parameters that require gradients (Hand Head)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params, lr=float(training_cfg["lr"]))

    # --- DATALOADER ---
    res = tuple(data_cfg["resolution"])
    dataset = HOT3DHandDataset(data_cfg["seq_path"], num_frames=data_cfg["num_frames"], res=res)
    dataloader = DataLoader(dataset, batch_size=1)

    num_frames = data_cfg["num_frames"]
    epochs     = training_cfg["epochs"]
    log_every  = training_cfg["log_every"]

    print(f"🚀 Starting training on {device} with Automatic Mixed Precision (AMP)...")

    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()

            # Data in Float32 (AMP will handle internal conversion)
            imgs = batch["img"].to(device)
            gt = batch["gt_joints"].to(device)

            views = {
                "img": imgs, # [1, S, 3, H, W]
                "is_target": torch.zeros((1, num_frames), dtype=torch.bool, device=device),
                "timestamp": torch.arange(0, num_frames, device=device).unsqueeze(0)
            }

            # --- MIXED PRECISION FORWARD PASS ---
            # Resolves error: expected scalar type Float but found BFloat16
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                first_t = next(iter(views.values()))
                curr_b = first_t.shape[0] if isinstance(first_t, torch.Tensor) else 1
                curr_f = num_frames

                h_val, w_val = 512, 512
                if "images" in views:
                    h_val, w_val = views["images"].shape[-2:]

                views["is_static"] = torch.zeros((curr_b, curr_f), dtype=torch.bool, device=device)
                views["valid_mask"] = torch.ones((curr_b, curr_f, h_val, w_val), dtype=torch.bool, device=device)
                views["camera_poses"] = torch.eye(4).view(1, 1, 4, 4).expand(curr_b, curr_f, 4, 4).to(device)
                views["camera_intrs"] = torch.eye(3).view(1, 1, 3, 3).expand(curr_b, curr_f, 3, 3).to(device)
                views["depthmap"] = torch.ones((curr_b, curr_f, h_val, w_val), dtype=torch.float32, device=device)

                preds = model(views, is_inference=False, use_motion=False)
                # Compute MSE loss between the 63 predicted values and the GT
                loss = F.mse_loss(preds["hand_joints"], gt)

            # Backpropagation (outside autocast)
            loss.backward()
            optimizer.step()

            if epoch % log_every == 0:
                print(f"🔥 Epoch {epoch:03d}/{epochs} | Loss: {loss.item():.6f}")

    # --- SAVE FINAL WEIGHTS ---
    output_weights = training_cfg["output_weights"]
    torch.save(model.hand_head.state_dict(), output_weights)
    print(f"💾 Hand Head weights saved to: {output_weights}")

if __name__ == "__main__":
    train()
