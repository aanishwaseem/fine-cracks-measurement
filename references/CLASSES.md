# Crack classes used for labelling references

Colour-coded reference images are saved next to the plain mask:

| Class | Id | Colour (RGB) | Rule used when labelling |
|---|---|---|---|
| Compression crack | 1 | red (255, 0, 0) | starts at the top edge and runs downward |
| Tension crack | 2 | cyan (0, 255, 255) | straight, near vertical, about 75-100 degrees from horizontal |
| Shear Type 1 | 3 | orange (255, 140, 0) | inclined crack that starts at the left or right side or at the bottom and runs inward |
| Shear Type 2 | 4 | purple (160, 32, 240) | diagonal crack inside the panel that does not start at an edge |
| Rebar detachment | 5 | green (0, 200, 0) | shallow crack low in the frame, roughly along the bottom |

Files for a labelled reference `X_ref.png`:

- `X_ref_labeled.png`: colour image on black (BGR when read with OpenCV: use the RGB values above).
- `X_ref_classes.png`: one channel, pixel value = class id (0 = background). Use this for training.
- `X_ref_labeled_on_frame<N>.png`: colours over the photo (frame N of the dataset), with a legend, for checking.

Background (general reinforced-concrete beam literature): flexural cracks are vertical and start on the tension face;
flexure-shear cracks start as flexural cracks and bend into a diagonal; web-shear cracks form diagonally in the web;
bond or splitting cracks run along the bottom bars; compression-zone cracks and crushing appear at the top.
The class names above are the project's own, so the mapping from literature to class is a judgement call.

Labelled so far: `B-1330-5_56_Cropped_ref` (frame 90) and `B-868-7_14_Cropped_ref` (frame 450).

How the rules were applied (hand-assigned per crack, from zoomed tiles):

- Steepness decides tension: stretches of 75 degrees or more from horizontal are tension (cyan); the same crack can switch colour along its length.
- A long diagonal (roughly 25-50 degrees) inside the panel is Shear Type 2; steeper inclined cracks that begin at a side or the bottom are Shear Type 1.
- A crack that flattens out low in the frame (below about 85% of the height, under about 25 degrees) is rebar detachment from that point on.
- A mark that touches the top edge is compression. In B-868 this is a dark patch in the top-left corner, which may be a stain rather than a crack.

Full description of the labelling method: `../LABELING.md`.
