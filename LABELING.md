# How the reference images are labelled

This explains how a reference mask (white cracks on black) is turned into a colour-coded, five-class label image.
It describes what was actually done for the two references labelled so far, and how to repeat it on another dataset.

Labelled so far:

| Reference | Built from | Size | Result files in `references/` |
|---|---|---|---|
| `B-1330-5_56_Cropped_ref.png` | frame 90 | 1950 x 716 | `..._ref_labeled.png`, `..._ref_classes.png`, `..._ref_labeled_on_frame90.png` |
| `B-868-7_14_Cropped_ref.png` | frame 450 | 2042 x 760 | `..._ref_labeled.png`, `..._ref_classes.png`, `..._ref_labeled_on_frame450.png` |

## 1. The classes

| Id | Class | Colour (RGB) | Meaning (from the project brief) |
|---|---|---|---|
| 1 | Compression crack | red (255, 0, 0) | originates from the top |
| 2 | Tension crack | cyan (0, 255, 255) | usually straight, angle 75 to 100 degrees |
| 3 | Shear Type 1 | orange (255, 140, 0) | originates from the left-bottom, right-bottom or the sides |
| 4 | Shear Type 2 | purple (160, 32, 240) | diagonal |
| 5 | Rebar detachment | green (0, 200, 0) | at the bottom |

Background reading (general reinforced-concrete beam literature, not these exact class names):
flexural cracks are vertical and start on the tension face; shear cracks are diagonal, about 45 degrees near supports,
and are either flexure-shear (a flexural crack that bends over) or web-shear (diagonal from the start, inside the web);
bond or splitting cracks run along the bottom bars; cracking and crushing in the compression zone appear at the top.
The mapping from that literature to the five classes is a judgement, so labels are best treated as a careful first pass for review.

## 2. Step 1: make a clean reference mask

The label is only as good as the mask under it. The mask is made first, by hand, from one late frame of the dataset
(late frames show the whole crack network):

1. Run the pipeline on that frame with no reference (`python run_pipeline.py --iopaint --no-reference --image "<frame>"`).
   This gives a binary mask that still contains the gridlines and surface noise.
2. Cut the frame into zoomed tiles (about 2.4x) with coordinate rulers and the mask outline drawn on top.
3. For every crack seen in the photo, trace its path as a polyline and keep only the model's pixels inside a brush corridor
   around that path. Gridlines that are not on a crack fall outside every corridor and disappear.
4. Erase the short gridline stubs that remain where a crack crosses a line, box by box, using positions read off the rulers.
5. Paint in any crack the model missed. Widen the mask slightly (7 x 7 pixels) so it is generous around the crack.
6. Check each tile again with the mask outline over the photo before saving as `references/<dataset folder>_ref.png`.

Nothing in this step removes gridlines automatically; every keep and erase decision is made by looking at the tiles.

## 3. Step 2: give every crack a class

Each crack is split into stretches, and each stretch gets one class. For every stretch, decide in this order and take the first rule that fits:

1. **Starts at the top edge and runs down** -> compression (red).
2. **Steep: 75 degrees or more from horizontal (up to about 100)** and at least about 40 px long -> tension (cyan).
   The angle is measured from the polyline points on the tile. A crack can change colour along its length where it gets steeper or shallower.
3. **Low in the frame (below about 85% of the height) and shallower than about 25 degrees** -> rebar detachment (green).
   If a diagonal crack flattens out near the bottom, it turns green from the point where it flattens.
4. **Inclined and starts at a side or at the bottom**, or is a branch of a crack that does -> Shear Type 1 (orange).
5. **Long diagonal (roughly 25 to 50 degrees) in the interior that does not start at an edge** -> Shear Type 2 (purple).

Branches take the class of the crack they split from unless a rule above (steep, or flat and low) says otherwise.

### How pixels get their class

Each stretch is a polyline with a brush width and a class. Painting happens only on pixels that are white in the reference,
so the label never adds or removes crack pixels; it only colours them. Later stretches overwrite earlier ones where brushes overlap.
Any reference pixel no brush reached takes the class of the nearest coloured pixel. In outline:

```python
label = zeros_like(reference)
for cls, width, points in stretches:           # hand-written list, one entry per stretch (not stored in the repo)
    brush = polyline_mask(points, width)
    label[brush & reference] = cls
label[unlabelled & reference] = class_of_nearest_labelled_pixel
```

### Outputs

For a reference `X_ref.png` (white = crack):

- `X_ref_labeled.png`: colour image on a black background.
- `X_ref_classes.png`: single channel, pixel value = class id (0 = background). Use this for training.
- `X_ref_labeled_on_frame<N>.png`: colours over the photo of frame N with a legend, for checking.

## 4. Worked examples

**B-1330-5.56, frame 90** (1950 x 716)

| Class | Pixels | What it is |
|---|---|---|
| Compression | 0 | no crack starts at the top edge in this frame |
| Tension | 3,006 | steep top of the left crack, the near-vertical lower part of a branch, a short hook at the end of the right arc |
| Shear Type 1 | 23,871 | the main left crack from the left side to the bottom, its branches, the right arc crack that starts at the right side |
| Shear Type 2 | 15,979 | the interior diagonals of the right-hand network and the faint chain in the upper middle |
| Rebar detachment | 10,719 | four shallow cracks low in the frame in the middle |

**B-868-7.14, frame 450** (2042 x 760)

| Class | Pixels | What it is |
|---|---|---|
| Compression | 2,902 | dark patch in the top-left corner that touches the top edge (may be a stain, not a crack) |
| Tension | 2,496 | vertical crack at the bottom (x about 1100) and the steep stretches of the left crack |
| Shear Type 1 | 969 | the inclined stretches of the left crack |
| Shear Type 2 | 9,149 | the long interior diagonal with its parallel strand and faint branch |
| Rebar detachment | 3,120 | the flattened tail of the diagonal and the shallow branches at the bottom right |

## 5. Known uncertainties

- Class boundaries use angles and heights read by eye from zoomed tiles, so they can be off by several degrees or pixels.
- Shallow cracks low in the frame can be rebar detachment or flat Shear Type 2; the rules above pick rebar detachment.
- A crack that crosses a gridline may keep a thin bar of gridline at the crossing.
- The labels were not checked by a structural engineer. Please review before using them for measurement or training.

## 6. Correcting a label

The list of stretches used to paint each label was written by hand for each reference and is not stored in this repo;
only the results are. To change a label, recolour the pixels directly in `X_ref_classes.png`, whose values are simply the class ids above,
and regenerate the colour image from it, or ask for the stretch to be relabelled.
`references/CLASSES.md` holds the short version of the class table.
