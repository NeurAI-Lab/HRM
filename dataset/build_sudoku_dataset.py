from typing import Optional
import os
import csv
import json
import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from .common import PuzzleDatasetMetadata
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor

cli = ArgParser()



class SudokuImageRenderer:
    """
    Render Sudoku puzzles (flattened 81 ints: 0=blank, 1..9=digits) to RGB images.

    Args:
        output_size: final side length of output image (e.g., 32 -> 32x32).
        render_res: internal canvas side length for crisp glyphs (>= output_size).
        line_thick: base grid line thickness at render_res (thicker 3x3 lines auto-handled).
        font_scale: OpenCV font scale at render_res.
        font_thick: OpenCV font thickness at render_res.
    """
    def __init__(self,
                 output_size: int = 32,
                 render_res: int = 144,
                 line_thick: int = 2,
                 font_scale: float = 0.5,
                 font_thick: int = 1):
        assert render_res % 9 == 0, "render_res should be divisible by 9"
        self.output_size = int(output_size)
        self.render_res = int(render_res)
        self.cell = self.render_res // 9
        self.line_thick = int(line_thick)
        self.font_scale = float(font_scale)
        self.font_thick = int(font_thick)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def render_single(self, grid_flat: np.ndarray) -> np.ndarray:
        """Returns (H, W, 3) uint8 image with H=W=output_size."""
        img = np.full((self.render_res, self.render_res, 3), 255, np.uint8)

        # Grid lines (thicker for 3x3 boundaries)
        for i in range(10):
            t = self.line_thick + (1 if i % 3 == 0 else 0)
            y = i * self.cell
            x = i * self.cell
            cv2.line(img, (0, y), (self.render_res, y), (0, 0, 0), t)
            cv2.line(img, (x, 0), (x, self.render_res), (0, 0, 0), t)

        # Digits
        for r in range(9):
            for c in range(9):
                v = int(grid_flat[r * 9 + c])
                if v == 0 or v == 1: # The numbers are mapped from 1-->10 in this file, so ignore values < 2
                    continue
                text = str(v)
                (tw, th), _ = cv2.getTextSize(text, self.font, self.font_scale, self.font_thick)
                tx = c * self.cell + (self.cell - tw) // 2
                ty = r * self.cell + (self.cell + th) // 2
                cv2.putText(img, text, (tx, ty), self.font, self.font_scale, (0, 0, 0),
                            self.font_thick, cv2.LINE_AA)

        if self.output_size != self.render_res:
            img = cv2.resize(img, (self.output_size, self.output_size), interpolation=cv2.INTER_AREA)
        return img

    def render_batch(self, inputs_np: np.ndarray, num_workers: int = 32) -> np.ndarray:
        """
        Threaded batch renderer with tqdm.
        inputs_np: (N, 81) -> returns (N, 3, output_size, output_size) uint8
        """
        N = int(inputs_np.shape[0])

        # sequential fallback (still shows progress)
        if num_workers <= 1:
            out = np.empty((N, self.output_size, self.output_size, 3), dtype=np.uint8)
            for i in tqdm(range(N), desc="Creating images (1 thread)", unit="img"):
                out[i] = self.render_single(inputs_np[i])
            return np.transpose(out, (0, 3, 1, 2))

        # multithreaded path (order-preserving)
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            # chunksize is optional; ignored by ThreadPoolExecutor in some Python versions
            imgs_iter = ex.map(self.render_single, inputs_np, chunksize=64)
            imgs = list(imgs_iter)

        out = np.stack(imgs, axis=0)  # (N, H, W, 3)
        return np.transpose(out, (0, 3, 1, 2))



class DataProcessConfig(BaseModel):
    source_repo: str = "sapientinc/sudoku-extreme"
    output_dir: str = "data/sudoku-extreme-full"

    subsample_size: Optional[int] = None
    min_difficulty: Optional[int] = None
    num_aug: int = 0


def shuffle_sudoku(board: np.ndarray, solution: np.ndarray):
    # Create a random digit mapping: a permutation of 1..9, with zero (blank) unchanged
    digit_map = np.pad(np.random.permutation(np.arange(1, 10)), (1, 0))
    
    # Randomly decide whether to transpose.
    transpose_flag = np.random.rand() < 0.5

    # Generate a valid row permutation:
    # - Shuffle the 3 bands (each band = 3 rows) and for each band, shuffle its 3 rows.
    bands = np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])

    # Similarly for columns (stacks).
    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])

    # Build an 81->81 mapping. For each new cell at (i, j)
    # (row index = i // 9, col index = i % 9),
    # its value comes from old row = row_perm[i//9] and old col = col_perm[i%9].
    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])

    def apply_transformation(x: np.ndarray) -> np.ndarray:
        # Apply transpose flag
        if transpose_flag:
            x = x.T
        # Apply the position mapping.
        new_board = x.flatten()[mapping].reshape(9, 9).copy()
        # Apply digit mapping
        return digit_map[new_board]

    return apply_transformation(board), apply_transformation(solution)


def convert_subset(set_name: str, config: DataProcessConfig):
    # Read CSV
    inputs = []
    labels = []
    
    with open(hf_hub_download(config.source_repo, f"{set_name}.csv", repo_type="dataset"), newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for source, q, a, rating in reader:
            if (config.min_difficulty is None) or (int(rating) >= config.min_difficulty):
                assert len(q) == 81 and len(a) == 81
                
                inputs.append(np.frombuffer(q.replace('.', '0').encode(), dtype=np.uint8).reshape(9, 9) - ord('0'))
                labels.append(np.frombuffer(a.encode(), dtype=np.uint8).reshape(9, 9) - ord('0'))

    # If subsample_size is specified for the training set,
    # randomly sample the desired number of examples.
    if set_name == "train" and config.subsample_size is not None:
        total_samples = len(inputs)
        if config.subsample_size < total_samples:
            indices = np.random.choice(total_samples, size=config.subsample_size, replace=False)
            inputs = [inputs[i] for i in indices]
            labels = [labels[i] for i in indices]

    # Generate dataset
    num_augments = config.num_aug if set_name == "train" else 0

    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    puzzle_id = 0
    example_id = 0
    
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)
    
    for orig_inp, orig_out in zip(tqdm(inputs), labels):
        for aug_idx in range(1 + num_augments):
            # First index is not augmented
            if aug_idx == 0:
                inp, out = orig_inp, orig_out
            else:
                inp, out = shuffle_sudoku(orig_inp, orig_out)

            # Push puzzle (only single example)
            results["inputs"].append(inp)
            results["labels"].append(out)
            example_id += 1
            puzzle_id += 1
            
            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0)
            
        # Push group
        results["group_indices"].append(puzzle_id)
        
    # To Numpy
    def _seq_to_numpy(seq):
        arr = np.concatenate(seq).reshape(len(seq), -1)
        
        assert np.all((arr >= 0) & (arr <= 9))
        return arr + 1
    
    results = {
        "inputs": _seq_to_numpy(results["inputs"]),
        "labels": _seq_to_numpy(results["labels"]),
        
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    # # Render images --> Painfully slow and huge!! 
    # renderer = SudokuImageRenderer(output_size=224, render_res=288)
    # results["images"] = renderer.render_batch(results["inputs"])
    
    # Metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=81,
        vocab_size=10 + 1,  # PAD + "0" ... "9"
        
        pad_id=0,
        ignore_label_id=0,
        
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        sets=["all"]
    )

    # Save metadata as JSON.
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)
        
    # Save data
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
        
    # Save IDs mapping (for visualization only)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    convert_subset("train", config)
    convert_subset("test", config)


if __name__ == "__main__":
    cli()
