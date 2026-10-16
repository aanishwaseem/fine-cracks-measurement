"""
Run run_pipeline.py over a folder of frames, optionally every Nth frame.
Frames whose output already exists are skipped, so an interrupted run can be restarted.

Examples:
    python run_folder.py "datasets/B-1330-5.56_Cropped" --every 6 --iopaint --no-reference --out outputs/B1330
    python run_folder.py "datasets/C-1700-4.60_Cropped" --iopaint --reference references/C-1700-4_60_Cropped_ref.png

Any extra arguments after the known ones are passed to run_pipeline.py (e.g. --deepcrack-tile 200).
Run one copy at a time per checkout: the models share tile folders under models/.
"""
import argparse
import os
import re
import subprocess
import sys

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def frame_number(name):
    nums = re.findall(r"\((\d+)\)|(\d+)(?=\.\w+$)", name)
    flat = [int(a or b) for a, b in nums]
    return flat[-1] if flat else -1


def output_stem(path, iopaint):
    stem = re.sub(r"[^\w\-]", "_", os.path.splitext(os.path.basename(path))[0])
    return stem + ("_iopaint" if iopaint else "")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder")
    p.add_argument("--every", type=int, default=1, help="take every Nth frame (by frame number order)")
    p.add_argument("--offset", type=int, default=0, help="start at the Nth frame of the sorted list")
    p.add_argument("--out", default="outputs/batch")
    p.add_argument("--iopaint", action="store_true")
    p.add_argument("--no-reference", action="store_true")
    p.add_argument("--reference", default=None)
    args, extra = p.parse_known_args()

    files = [f for f in os.listdir(args.folder) if f.lower().endswith(IMAGE_EXTS)]
    files.sort(key=lambda f: (frame_number(f), f))
    picked = files[args.offset::args.every]
    print(f"{len(files)} frames in folder, running {len(picked)} (every {args.every}, offset {args.offset})")

    suffix = "_binary_no_reference.png" if args.no_reference else "_binary.png"
    for i, name in enumerate(picked, 1):
        path = os.path.join(args.folder, name)
        done = os.path.join(args.out, output_stem(path, args.iopaint) + suffix)
        if os.path.exists(done):
            print(f"[{i}/{len(picked)}] skip {name}")
            continue
        print(f"[{i}/{len(picked)}] {name}", flush=True)
        cmd = [sys.executable, "run_pipeline.py", "--image", path, "--out", args.out]
        if args.iopaint:
            cmd.append("--iopaint")
        if args.no_reference:
            cmd.append("--no-reference")
        elif args.reference:
            cmd += ["--reference", args.reference]
        subprocess.run(cmd + extra, check=False)


if __name__ == "__main__":
    main()
