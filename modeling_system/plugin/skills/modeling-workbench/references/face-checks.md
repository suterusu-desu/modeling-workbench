# Face construction audit

Before any mouth, viseme or expression work, and on every face shape after it, run the face audit
(`face_checks.run_face_audit`, the service operation `audit_face`). A face ships as single-frame blend shapes; three
commercial avatars studied shape by shape follow the same construction, and each rule below is a check. Numbers can
reject a construction; they never approve its appearance, so watch the shapes too. The eye has its own checks
([construction checks](construction-checks.md)).

## Inputs

Extract each object with `study_extract.py` (the skin and, as their own objects, the teeth and tongue: see
[study the avatars](study-avatars.md)). Shapes are named keys; a key applies to every object that has it, so a jaw
shape on the skin and the teeth is one shape. Units are mouth widths (the lips' extent across, or `unit`).

```json
{
  "skin": "Face",
  "objects": {"Face": {"path": "Face.npz"}, "Upper teeth": {"path": "Upper_teeth.npz"},
              "Lower teeth": {"path": "Lower_teeth.npz"}, "Tongue": {"path": "Tongue.npz"}},
  "sets": {"upper lip": [...], "lower lip": [...], "lid margin": [...], "mouth area": [...],
           "lower face": [...], "outer skin": [...], "attachment ring": [...]},
  "lips": {"upper": "upper lip", "lower": "lower lip"},
  "shapes": {"A": {"region": "lower face", "still": ["lid margin"]},
             "U": {"region": "mouth area", "still": ["lid margin"], "boundary": 2.0, "seams": ["attachment ring"]}},
  "controls": {"mouth_open": {"samples": [0, 0.25, 0.5, 0.75, 1], "weights": {"A": [0, 0.25, 0.5, 0.75, 1]}}},
  "jaw": {"shape": "A", "rigid": ["Lower teeth", "Tongue"], "surface": "outer skin", "upper_lip_share": 0.3},
  "visemes": {"slots": ["vrc.v_sil", "vrc.v_pp", "...15 names in the descriptor's order..."],
              "basis": ["A", "I", "U", "E", "O"], "pp": "vrc.v_pp", "ff": "vrc.v_ff", "upper_teeth": "Upper teeth"},
  "pairs": [{"both": "smile", "left": "smile_L", "right": "smile_R", "split": "feathered"},
            {"both": "blink", "left": "blink_L", "right": "blink_R", "split": "hard"}],
  "teeth": ["Upper teeth", "Lower teeth"],
  "combinations": [{"name": "smile + aa", "weights": {"smile": 1, "vrc.v_aa": 1}},
                   {"name": "pp", "weights": {"vrc.v_pp": 1}, "press": 0.03}],
  "clips": [{"name": "happy", "weights": {"smile": 0.8, "brow_up": 0.5}}],
  "expression_shapes": ["smile", "brow_up"]
}
```

- `sets`: vertex indices of the skin (inline or `{"path", "key"}`); `lips` the two seam rows (points of the two lips
  facing each other at rest, paired within `pair_distance`, default .05). `up`, `forward` and `across` default to +z,
  -y and +x.
- `controls`: a control's weight for each key it drives, sampled from 0 to 1 (read them off the rig's drivers).
  `correctives` names the keys a bake drives by a bump of the main weight (`{"blink_mid": "mid"}`; `mid`, `early` or
  `late`, as [bake](bake.md) writes them): such a key passes `phase_gated_keys` when its weights follow that bump, and
  `path_deviation` is measured without it, the arc it carries (a hinge's roll) reported as `arc_with_correctives`.
  Undeclared bumps and late ramps, or a declared key off its bump, still fail.
- `jaw.surface`: the outer skin when the mouth bag or teeth share the skin mesh; `chin`, `min_behind` (default 1.5
  mouth widths) and `min_above` (0) are optional.
- `visemes.slots`: the descriptor's list in VRChat's order, or a mapping from slot to key.
- `combinations`: allowed sums (expression plus viseme, plus blink); `press` allows a deliberate press of the lips.
- `limits`: overrides of the defaults below, also per shape inside `shapes`.

## The checks

| Rule | Check | Measures | Default |
|---|---|---|---|
| C1 region | `region_outside_share` | share of a shape's skin motion outside its region | .05 |
| | `still_travel_share` | largest travel of a still set (the lid margin), share of the shape's largest | .02 |
| C2 one path | `path_deviation` | largest distance of a control's path from its straight chord, share of travel | .05 |
| | `path_reversals` | moving vertices whose progress along the chord falls back by more than 2 % | 0 |
| C2/C10 | `phase_gated_keys` | keys driven by a weight not proportional to the control (bumps, late ramps) | 0 |
| C3 jaw | `jaw_rigid_residual` | RMS distance of each rigid part from its best rigid turn, mouth widths | .01 |
| | `jaw_hinge_position` | how much less far behind / above the lips the hinge is than required | 0 |
| C4 skin | `jaw_skin_residual` | median share of the skin's motion below the lips off the jaw turn | .25 |
| | `jaw_weight_monotone` | largest rise of the median jaw weight going away from the chin | .1 |
| | `jaw_drags_upper_face` | share of the moving skin above the lips following the jaw by more than a quarter | .1 |
| | `upper_lip_share_error` | the upper lip's share of the opening against the declared target | .05 |
| C5 falloff | `falloff_beyond_boundary` | largest travel farther from the seam than the boundary, share of the largest | .05 |
| | `falloff_on_seam` | largest travel on a declared topology seam (a stitched ring), share | .05 |
| C6 visemes | `viseme_slot_mapping` | slots playing a key named for another slot | 0 |
| | `pp_gap` | widest lip gap at PP along the mouth (closest pair at each place) | .01 |
| | `ff_gap` | closest approach of the lower lip to the upper teeth at FF | .02 |
| C7 left/right | `lr_sum_error` | largest \|L + R - both\|, share of both | .03 |
| | `lr_midline` | hard split: feather beyond .02; feathered: shortfall from .1 mouth widths | 0 |
| | `lr_crease` | largest fall of the left share across the midline (feathered) | .05 |
| C8 combinations | `lips_cross` | how far the lips pass through each other (plus `press`) | .01 |
| | `teeth_behind_lips` | how far teeth come in front of the skin covering them | 0 |
| C9 production | `single_frame_keys` | keys stored against another key instead of the basis | 0 |
| | `clip_full_state` | expression shapes a clip leaves unset | 0 |

The report adds the jaw's angle, hinge position, chord sag at half weight (whether the jaw needs in-betweens) and
upper-lip share, and each viseme's nearest non-negative mix of the basis with its residual.

## What the avatars showed, and the failing mouth

On the three avatars: mouth shapes moved the lid margin by 0.00; lip shapes fell to about 0 inside a boundary 1.6-2.2
mouth widths from the seam while jaw shapes carried past it; the lower teeth turned 7-9.4 degrees rigidly about an axis
2.4 mouth widths behind and .5-.85 above the lips; the skin below the lips followed by a weight (median .10-.23); the
upper lip gave a third of the opening; most visemes were mixes of five vowels; eye and brow splits were hard and mouth
splits feathered over about .1 mouth widths; every shape had one frame.

Audit results: one avatar passed every declared check (jaw 9.4 degrees, hinge 2.36 behind and .49 above, skin residual
.12, PP pressing .026, feathered smile .22 wide, eye and brow splits hard). Another's descriptor played the wrong keys
in two slots (E and ih), found from the descriptor itself, and its FF stopped .025 short of the upper teeth; the
third's FF did not reach them (.035) and its PP is empty (the rest keeps the lips within .008, a pass). A character's rejected mouth, one control driving the open shape plus guide keys gated to phases of
it, failed where the study predicted: four phase-gated keys, paths up to .55 of their travel off the chord (36 % of the
moving vertices over .1), 36 reversing vertices, teeth not rigid (.043), a hinge only 1.24 mouth widths behind the lips
and skin off the jaw turn (.43). Two one-sided sculpts declared as splits failed `lr_sum_error`: declare only real
splits.

## Limits

- Paths are sampled at the declared control values; region, still and seam sets are the declaration's.
- `teeth_behind_lips` compares each tooth point with the front of the skin within .1 mouth widths of it in the front
  view; points seen through an open mouth are not constrained.
- The audit does not judge likeness: expressions and visemes are still compared with their guides by overlap.
- Motion added after the shape keys (subdivision fields, node fields, drivers doing the shaping) is not in an
  extraction: bake the shipped motion and round-trip the export ([bake](bake.md)). Whether a turn needs in-betweens
  is reported as the jaw's chord sag at half weight, not gated.
