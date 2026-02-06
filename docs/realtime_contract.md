# Realtime JSON Contract

This document defines structured fields shared by `/api/realtime/turn`, `/api/speak`, and the face UI.

## EmotionState

```json
{
  "mood": "neutral|happy|sad|angry|thinking|surprised",
  "intensity": 0.0,
  "confidence": 0.0,
  "source": "heuristic|..."
}
```

- `mood`: canonical discrete state.
- `intensity`: `0.0` to `1.0`, controls expression strength.
- `confidence`: `0.0` to `1.0`, reliability of inferred emotion.
- `source`: where emotion was produced (`heuristic`, model, etc.).

## ThoughtEvent

```json
{
  "text": "short thought bubble text",
  "category": "response|planning|warning|...",
  "importance": 0.0
}
```

- `importance`: `0.0` to `1.0`, visual prominence hint.

## EmojiDirective

```json
{
  "emoji": "😄",
  "placement": "left_eye|right_eye|status|speech_cloud",
  "intensity": 0.0,
  "fallback_text": "optional plain-text fallback"
}
```

- `emoji` allowed set: `🙂 😄 😢 😠 🤔 😮 ✨ ⚙️ 💡`.
- `placement` controls deterministic badge location.
- `intensity` range `0.0` to `1.0`.

## ExpressionDirectives

```json
{
  "emotion_state": {"...": "EmotionState"},
  "thought_event": {"...": "ThoughtEvent"},
  "emoji_directive": {"...": "EmojiDirective"},
  "face_motion": {
    "sway": 0.0,
    "breath": 0.0,
    "drift": 0.0
  },
  "compat": { "mood": "neutral" }
}
```

- `face_motion` values are normalized `0.0` to `1.0`.
- `compat.mood` preserves compatibility for clients using only `mood`.

## Endpoint behavior

- `/api/realtime/turn` always returns:
  - `emotion_state`
  - `thought_event`
  - `expression_directives`
  - `emoji_directive`
  - legacy `mood` (compat shim)
- `/api/speak` accepts either:
  - legacy `mood`, or
  - structured `expression_directives` / `emotion_state`.

When both are provided, structured directives take precedence.

## Digital Being Layers (`/api/realtime/turn`)

`layers.digital_being` provides a high-level observe → think → act/speak trace:

- `perception`: normalized observation features (token density, punctuation energy, modality presence, uncertainty).
- `appraisal`: continuous social/affect dimensions (safety, urgency, novelty, social_closeness, certainty).
- `cognition`: intent and blended reasoning weights.
- `identity`: stable-to-dynamic posture state with smoothing (`previous`, `target`, `current`).
- `action_policy`: normalized policy mix across exploratory/supportive/instructional actions.
- `expression`: output-facing tone, cadence, and policy blend hints.

`layers.sequential_task_list` reports progression through six implementation sections:

1. Perceptual layer
2. Appraisal layer
3. Cognitive layer
4. Self/identity layer
5. Action policy layer
6. Expressive layer

Each step includes a `status` of `pending`, `in_progress`, or `completed`.
