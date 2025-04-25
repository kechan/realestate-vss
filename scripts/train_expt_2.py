from typing import List, Set, Dict, Tuple, Any, Optional, Iterator, Union, Callable, TypeVar, Generic
import time, math, random, PIL, gc, re, gzip, json, pyperclip, hashlib
from collections import defaultdict

from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm.auto import tqdm

from realestate_core.common.utils import save_to_pickle, load_from_pickle, join_df
from realestate_vision.common.utils import get_listingId_from_image_name

from realestate_vss.models.embedding import OpenClipTextEmbeddingModel, OpenClipImageEmbeddingModel

import open_clip
import faiss
from realestate_vss.data.index import FaissIndex

import torch
from torch import Tensor
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

from torch.utils.data import Dataset, DataLoader, IterableDataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as nn_utils
import numpy as np
from PIL import Image

# ensure reproducibility during development
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
torch.mps.manual_seed(GLOBAL_SEED)

def worker_init_fn(worker_id):
  seed = GLOBAL_SEED + worker_id
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.mps.manual_seed(seed)

project_name = 'NLImageSearch'
local_vol = Path('/Volumes/Samsung_T7')
local_home = local_vol/'jumptools_gdrive'
local_project_home = local_home/project_name
photos_dir = local_home/project_name/'photos'
remarks_dir = local_home/project_name/'structured_data_and_remarks'
eval_img_data_dir = Path("/Volumes/Samsung_T5/RLP/images/photos")

chkpt_dir = Path("/Volumes/Samsung_T7/jumptools_gdrive/NLImageSearch/distillation/experiment_3/distilled_models")  # for saving checkpoints

student_model_name = 'ViT-B-32'
student_pretrained='laion2b_s34b_b79k'

def instantiate_student_model(model_path=None, epoch=None, device=None) -> Tuple[nn.Module, Callable, Callable]:
  """
  Instantiate the student model and load the checkpoint if it exists.
  Args:
    epoch (int): The epoch number to load the checkpoint from. If None, the latest checkpoint will be loaded.
  Returns:
    student_model: The student model.
    student_preprocess: The preprocessing function for the student model.
    student_tokenizer: The tokenizer for the student model.
  """
  student_model, _, student_preprocess = open_clip.create_model_and_transforms(student_model_name, pretrained=student_pretrained)
  student_tokenizer = open_clip.get_tokenizer(student_model_name)

  if epoch is not None and model_path is not None:
    model_save_path = f'student_model_chkpt_set_{epoch}_epoch_1.pth'
    if (model_path/model_save_path).exists():
      print(f"Loading checkpoint from {model_save_path}")
      checkpoint = torch.load(model_path/model_save_path, map_location=device)
      student_model.load_state_dict(checkpoint['model_state_dict'])

  if device is not None:
    student_model = student_model.to(device)

  return student_model, student_preprocess, student_tokenizer


def build_retrieval_batch(
  teacher_embeddings: np.ndarray,
  student0_embeddings: np.ndarray,
  image_names: list[str],
  teacher_index: faiss.IndexFlatIP,
  batch_size: int = 32,
  pos_k: int = 10,
  neg_unif: int = 32,
  q_idxs: Optional[np.ndarray] = None
):
  """
  If `q_idxs` is None, will sample `batch_size` random queries.
  Otherwise must be an array of length == batch_size, which will be used
  verbatim every time for perfectly repeatable eval.

  Returns:
    img_paths:            list[str], length B*(1+pos_k+neg_unif)
    sim_teacher:          np.ndarray, shape (B, T)
    sim_student0:         np.ndarray, shape (B, T)
    q_idxs:               np.ndarray, shape (B,)
  """
  N, D = teacher_embeddings.shape

  # 1) set query indices
  if q_idxs is None:
    q_idxs = np.random.choice(N, batch_size, replace=False)
  else:
    assert len(q_idxs) == batch_size, "q_idxs length must == batch_size"

  all_paths = []
  sim_teacher = []
  sim_student0 = []

  for q in q_idxs:
    # 1) teacher positives
    q_emb_teacher = teacher_embeddings[q : q+1]  # (1, D)
    _, I_top = teacher_index.search(q_emb_teacher, pos_k + 1)
    pos_idxs = I_top[0]                           # (pos_k+1,)

    # 2) negatives
    pool = set(range(N)) - set(pos_idxs)
    neg_idxs = np.random.choice(list(pool), neg_unif, replace=False)

    # 3) gather batch indices
    batch_idxs = np.concatenate([pos_idxs, neg_idxs])  # (T,)

    # 4) teacher sims
    sims_t = (q_emb_teacher @ teacher_embeddings[batch_idxs].T).squeeze(0)  # (T,)

    # 5) original‑student sims
    q_emb_student0 = student0_embeddings[q : q+1]                          # (1, D)
    sims_s0 = (q_emb_student0 @ student0_embeddings[batch_idxs].T).squeeze(0)  # (T,)

    # 6) record paths & targets
    all_paths.extend(image_names[i] for i in batch_idxs)
    sim_teacher.append(sims_t.astype(np.float32))
    sim_student0.append(sims_s0.astype(np.float32))

  sim_teacher  = np.stack(sim_teacher,  axis=0)  # (B, T)
  sim_student0 = np.stack(sim_student0, axis=0)  # (B, T)

  return all_paths, sim_teacher, sim_student0, q_idxs


class RetrievalDataset(IterableDataset):
  def __init__(
    self,
    teacher_embeddings: np.ndarray,
    student0_embeddings: np.ndarray,
    image_names: list[str],
    teacher_index: faiss.IndexFlatIP,
    batch_size: int = 32,
    pos_k: int = 10,
    neg_unif: int = 32,
    preprocess: Optional[Callable] = None,
    data_dir: Optional[Path] = None
  ):
    # ensure float32 for FAISS & torch compatibility
    self.teacher_embeddings  = teacher_embeddings.astype("float32", copy=False)
    self.student0_embeddings = student0_embeddings.astype("float32", copy=False)
    self.image_names         = image_names
    self.teacher_index       = teacher_index
    self.batch_size          = batch_size
    self.pos_k               = pos_k
    self.neg_unif            = neg_unif

    self.preprocess          = preprocess   # if this is None, don't do preprocessing
    self.data_dir            = data_dir   # if this is None, don't do anything with paths

  def __iter__(self):
    while True:
      paths, sim_teacher, sim_student0, _ = build_retrieval_batch(
          teacher_embeddings=self.teacher_embeddings,
          student0_embeddings=self.student0_embeddings,
          image_names=self.image_names,
          teacher_index=self.teacher_index,
          batch_size=self.batch_size,
          pos_k=self.pos_k,
          neg_unif=self.neg_unif
      )

      # Generate a hash digest of the paths
      paths_digest = hashlib.md5("".join(paths).encode()).hexdigest()

      batch = None
      if self.preprocess is not None:
        imgs = [Image.open(self.data_dir/p).convert("RGB") for p in paths]
        batch = torch.stack([self.preprocess(img) for img in imgs], dim=0)

      yield {
        "paths": paths,                                # list[str], length B*T
        "paths_digest": paths_digest,                  # str
        "sim_teacher":  torch.from_numpy(sim_teacher),  # tensor (B, T)
        "sim_student0": torch.from_numpy(sim_student0),  # tensor (B, T)
        "batch": batch                                # tensor (B*T, 3, H, W)
      }


@torch.no_grad()
def embed_images_eval(
  model: torch.nn.Module, 
  preprocess: Callable, 
  paths: List[Path], 
  device="cpu",
  chunk_size=64) -> Tensor:
  """
  Load & preprocess images in chunks of chunk_size, then encode.
  Returns a tensor of shape (len(paths), D).
  """
  all_embs = []
  for i in tqdm(range(0, len(paths), chunk_size)):
    subpaths = paths[i : i + chunk_size]
    imgs = [Image.open(p).convert("RGB") for p in subpaths]
    batch = torch.stack([preprocess(img) for img in imgs], dim=0).to(device)
    embs = model.encode_image(batch, normalize=True)
    all_embs.append(embs)

  return torch.cat(all_embs, dim=0)


def embed_images_train(model: torch.nn.Module, preprocess: Callable, paths: List[Path], device="cpu") -> Tensor:
  imgs  = [Image.open(p).convert("RGB") for p in paths]
  batch = torch.stack([preprocess(img) for img in imgs], dim=0).to(device)
  return model.encode_image(batch, normalize=True)

# Load Data Sources (npy and pickles)
image_set_id = 1
# text_set_id = 1

train_image_names = load_from_pickle(Path.home()/'tmp'/f'train_image_names_{image_set_id}.pkl')
# val_image_names = load_from_pickle(Path.home()/'tmp'/'val_image_names.pkl')

train_teacher_image_embeddings = np.load(Path.home()/'tmp'/f'train_image_embeddings_{image_set_id}.npy')
# val_image_embeddings = np.load(Path.home()/'tmp'/'val_image_embeddings.npy')

# train_teacher_text_embeddings_df = pd.read_feather(Path.home()/'tmp'/f'train_text_embeddings_{text_set_id}_df')
# train_teacher_text_embeddings = np.stack(train_teacher_text_embeddings_df['embedding'].values)

# val_text_embeddings_df = pd.read_feather(Path.home()/'tmp'/'val_text_embeddings_df')
# val_text_embeddings = np.stack(val_text_embeddings_df['embedding'].values)

# train_image_index = faiss.IndexFlatIP(train_teacher_image_embeddings.shape[1])
# train_image_index.add(train_teacher_image_embeddings)
# print(f"Train Image Index size: {train_image_index.ntotal}")

# train_text_index = faiss.IndexFlatIP(train_teacher_text_embeddings.shape[1])
# train_text_index.add(train_teacher_text_embeddings)
# print(f"Train Text Index size: {train_text_index.ntotal}")

# print(f'val images size: {val_image_embeddings.shape[0]}')
# print(f'val texts size: {val_text_embeddings_df.shape[0]}')

train_student0_image_embeddings = np.load(Path.home()/'tmp'/f'train_student_0_image_embeddings_{image_set_id}.npy')
print(f'Train Student0 Image Embeddings size: {train_student0_image_embeddings.shape[0]}')

all_eval_teacher_image_embeddings = np.load(Path.home()/'tmp'/'eval_teacher_image_embeddings.npy')
eval_teacher_image_embeddings = all_eval_teacher_image_embeddings[:100_000]
all_eval_image_names = load_from_pickle(Path.home()/'tmp'/'eval_teacher_image_names.pkl')
eval_image_names = all_eval_image_names[:100_000]

print(f'Eval Teacher Image Embeddings size: {eval_teacher_image_embeddings.shape[0]}')

all_eval_student0_image_embeddings = np.load(Path.home()/'tmp'/f'eval_student_0_image_embeddings.npy')
eval_student0_image_embeddings = all_eval_student0_image_embeddings[:100_000]
print(f'Eval Student0 Image Embeddings size: {eval_student0_image_embeddings.shape[0]}')


# BUild FAISS Indexes
train_index = faiss.IndexFlatIP(train_teacher_image_embeddings.shape[1])
train_index.add(train_teacher_image_embeddings)

eval_index = faiss.IndexFlatIP(eval_teacher_image_embeddings.shape[1])
eval_index.add(eval_teacher_image_embeddings)

# Instantiate Student Model
student_model, student_preprocess, student_tokenizer = instantiate_student_model(epoch=None, device=device)
# student_model = student_model.to(device);

# configure Retrieval (pytorch) Dataset and DataLoader 
# pos_k = 10
# neg_unif = 32
pos_k = 20
neg_unif = 16

batch_size = 32

dataset = RetrievalDataset(
    train_teacher_image_embeddings,
    train_student0_image_embeddings,
    train_image_names,
    train_index,
    batch_size=batch_size,
    pos_k=pos_k,
    neg_unif=neg_unif
)

dataloader = DataLoader(
  dataset, 
  batch_size=None,
  # num_workers=0,
  # pin_memory=True,
  # prefetch_factor=1,
  # persistent_workers=False,
  # worker_init_fn=worker_init_fn
)

optimizer = torch.optim.AdamW(student_model.parameters(), lr=2e-6, weight_decay=1e-2)
criterion = torch.nn.MSELoss(reduction="mean")

# main training loop
# 7) Main training loop
num_steps = 100

# logging and early stopping params
log_interval = 5   # # evaluate every N steps
patience = 3       # how many evals to wait for a new best
min_delta = 1e-4   # smallest relative-improvement worth counting


# state
logs = defaultdict(list)
best_eval_rel_imp = -float('inf')   # highest rel-imp seen so far
best_step = -1
stale_counter = 0   # evals since last new best

running_loss = 0.0   # for distilled student
running_loss_0 = 0.0 # for original student

# Prepare eval set (which is fixed for every training step)
# number of eval anchors, choose once and then fixed them for eval

num_eval_anchors = 128
N_eval = eval_teacher_image_embeddings.shape[0]
q_idxs_eval = np.random.choice(N_eval, num_eval_anchors, replace=False)

# 1) Build the fixed eval batch once
paths_e, sim_te_e, sim_s0_e, _ = build_retrieval_batch(
  teacher_embeddings=eval_teacher_image_embeddings,
  student0_embeddings=eval_student0_image_embeddings,
  image_names=eval_image_names,
  teacher_index=eval_index,
  batch_size=num_eval_anchors,
  pos_k=pos_k,
  neg_unif=neg_unif,
  q_idxs=q_idxs_eval
)
B_e, T_e = sim_te_e.shape

eval_paths = [eval_img_data_dir/get_listingId_from_image_name(f)/f for f in paths_e]

# 2) Precompute student0 baseline eval loss
te_eval_tensor = torch.from_numpy(sim_te_e).to(device)
s0_eval_tensor = torch.from_numpy(sim_s0_e).to(device)
# eval_loss_0 = weighted_mse(s0_eval_tensor, te_eval_tensor, eval_weightings)
eval_loss_0 = criterion(s0_eval_tensor, te_eval_tensor)

student_model.train()

for step, batch in enumerate(dataloader, 1):
  start = time.perf_counter()
  print(f"Step {step} / {num_steps}")

  # DEBUG: inspect what this step is actually sampling
  print("paths digest:", batch["paths_digest"])
  
  paths = [photos_dir/f for f in batch["paths"]]                           # list length B*T    
  # imgs = batch["batch"].to(device, non_blocking=True)  # (B*T, 3, H, W)
  sim_teacher = batch["sim_teacher"].to(device)    # (B, T)
  sim_student0 = batch["sim_student0"].to(device)  # (B, T)
  B, T = sim_teacher.shape 

  # A) Embed all images at once
  embs = embed_images_train(student_model, student_preprocess, paths, device)  # (B*T, D)
  # embs = student_model.encode_image(imgs, normalize=True)  # (B*T, D)
  
  # B) Reshape into (B, T, D)
  # B = batch_size
  D = embs.size(-1)
  embs = embs.view(B, T, D)

  # C) Compute student similarities (B, T, T)
  stu_q = torch.bmm(embs, embs.transpose(1, 2))[:, 0, :]   # (B, T)
  # Query-to-all slice
  # stu_q = stu_sim[:, 0, :]  

  # D) Compute weighted MSE loss, and also the loss for the original student
  # loss = weighted_mse(stu_q, sim_teacher, training_weightings)
  loss = criterion(stu_q, sim_teacher)
  with torch.no_grad():
    # loss_0 = weighted_mse(sim_student0, sim_teacher, training_weightings)
    loss_0 = criterion(sim_student0, sim_teacher)

  # E) Backprop
  optimizer.zero_grad()
  loss.backward()

  # F) Gradient clipping
  nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
  
  optimizer.step()

  duration = time.perf_counter() - start
  print(f" step time: {duration:.2f}s loss: {loss.item():.4f}")

  # accumulate losses
  running_loss += loss.item()
  running_loss_0 += loss_0.item()


  # Logging and eval
  if step % log_interval == 0:
    avg_loss = running_loss / log_interval
    avg_loss_0 = running_loss_0 / log_interval
    rel_imp = (avg_loss_0 - avg_loss) / avg_loss_0   # 0 to start and the closer to 1 the better, -ve values are bad

    # reset accumulated losses
    running_loss = 0.0
    running_loss_0 = 0.0

    # eval batch on eval set
    student_model.eval()

    # distilled eval
    embs_e = embed_images_eval(student_model, student_preprocess, eval_paths, device)

    D_e      = embs_e.size(-1)
    embs_e   = embs_e.view(B_e, T_e, D_e)
    stu_q_e  = torch.bmm(embs_e, embs_e.transpose(1,2))[:, 0, :] # (B, T)

    # eval_loss = weighted_mse(stu_q_e, te_eval_tensor, eval_weightings)
    eval_loss = criterion(stu_q_e, te_eval_tensor)
    rel_imp_eval = (eval_loss_0 - eval_loss) / eval_loss_0

    # reset model to train mode
    student_model.train()

    # -------- checkpoint & early-stop logic ------------------------------------
    improved = rel_imp_eval - best_eval_rel_imp > min_delta
    if improved:
      best_eval_rel_imp = rel_imp_eval
      best_step = step
      stale_counter = 0

      # save checkpoints
      model_save_path = f'student_model_chkpt_set_{image_set_id}_epoch_1_{rel_imp_eval:.2%}.pth'
      save_checkpoint = {
        'model_state_dict': student_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # add any additional items here, like epoch number, loss, etc.
        'eval_rel_imp': best_eval_rel_imp
      }

      torch.save(save_checkpoint, chkpt_dir/model_save_path)

      print(f"✓ new best {best_eval_rel_imp:.4%}  →  saved {model_save_path}")
    else:
      stale_counter += 1
      print(f"no improvement ({stale_counter}/{patience})")

    # -------- logging ----------------------------------------------------------
    logs["step"].append(step)
    logs["train_mse"].append(avg_loss)
    logs["train_student0_mse"].append(avg_loss_0)
    logs["train_rel_imp"].append(rel_imp)
    logs["eval_mse"].append(eval_loss.item())
    logs["eval_student0_mse"].append(eval_loss_0.item())
    logs["eval_rel_imp"].append(rel_imp_eval.item())

    print(
      f"Step {step}: "
      f"train MSE={avg_loss:.4f}, "
      f"student0 MSE={avg_loss_0:.4f}, "
      f"rel imp={rel_imp:.2%} | "
      f"eval MSE={eval_loss:.4f}, "
      f"eval student0 MSE={eval_loss_0:.4f}, "
      f"eval rel imp={rel_imp_eval:.2%}"
    )

    # -------- stop if patience exhausted --------------------------------------
    if stale_counter >= patience:
      print(f"Early stopping at step {step} (patience={patience})")
      break

  if step >= num_steps:
    break
    
with open(chkpt_dir/f"student_model_chkpt_set_{image_set_id}_epoch_1_logs.json", "w") as f:
  json.dump(logs, f, indent=4)