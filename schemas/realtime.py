"""Structured realtime/speech contracts for Pi-Guy."""

from copy import deepcopy

ALLOWED_MOODS = {'neutral', 'happy', 'sad', 'angry', 'thinking', 'surprised'}
ALLOWED_EMOJIS = {'🙂', '😄', '😢', '😠', '🤔', '😮', '✨', '⚙️', '💡'}
ALLOWED_PLACEMENTS = {'left_eye', 'right_eye', 'status', 'speech_cloud'}


def _clamp(value, low, high, default):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, numeric))


def _clean_mood(value, fallback='neutral'):
    mood = (value or fallback or 'neutral').strip().lower()
    return mood if mood in ALLOWED_MOODS else fallback


def build_emotion_state(raw=None, fallback_mood='neutral'):
    payload = raw if isinstance(raw, dict) else {}
    mood = _clean_mood(payload.get('mood') or payload.get('primary'), fallback=fallback_mood)
    return {
        'mood': mood,
        'intensity': _clamp(payload.get('intensity', 0.5), 0.0, 1.0, 0.5),
        'confidence': _clamp(payload.get('confidence', 0.5), 0.0, 1.0, 0.5),
        'source': str(payload.get('source', 'heuristic')),
    }


def build_thought_event(raw=None):
    payload = raw if isinstance(raw, dict) else {}
    return {
        'text': str(payload.get('text', '')).strip(),
        'category': str(payload.get('category', 'response')),
        'importance': _clamp(payload.get('importance', 0.5), 0.0, 1.0, 0.5),
    }


def build_emoji_directive(raw=None, fallback_mood='neutral'):
    payload = raw if isinstance(raw, dict) else {}
    fallback_text = str(payload.get('fallback_text', '')).strip()
    emoji = str(payload.get('emoji', '')).strip()
    if not emoji:
        mood_to_emoji = {
            'happy': '😄',
            'sad': '😢',
            'angry': '😠',
            'thinking': '🤔',
            'surprised': '😮',
            'neutral': '🙂',
        }
        emoji = mood_to_emoji.get(_clean_mood(fallback_mood), '🙂')
    if emoji not in ALLOWED_EMOJIS:
        emoji = '🙂'
    placement = str(payload.get('placement', 'status')).strip().lower()
    if placement not in ALLOWED_PLACEMENTS:
        placement = 'status'
    return {
        'emoji': emoji,
        'placement': placement,
        'intensity': _clamp(payload.get('intensity', 0.6), 0.0, 1.0, 0.6),
        'fallback_text': fallback_text,
    }


def build_expression_directives(raw=None, fallback_mood='neutral', fallback_text=''):
    payload = raw if isinstance(raw, dict) else {}
    emotion_state = build_emotion_state(payload.get('emotion_state'), fallback_mood=fallback_mood)
    thought_event = build_thought_event(payload.get('thought_event'))
    if not thought_event['text'] and fallback_text:
        thought_event['text'] = str(fallback_text).strip()

    emoji_directive = build_emoji_directive(
        payload.get('emoji_directive'),
        fallback_mood=emotion_state['mood'],
    )
    face_motion = payload.get('face_motion') if isinstance(payload.get('face_motion'), dict) else {}
    motion = {
        'sway': _clamp(face_motion.get('sway', emotion_state['intensity']), 0.0, 1.0, emotion_state['intensity']),
        'breath': _clamp(face_motion.get('breath', 0.5), 0.0, 1.0, 0.5),
        'drift': _clamp(face_motion.get('drift', emotion_state['intensity']), 0.0, 1.0, emotion_state['intensity']),
    }
    return {
        'emotion_state': emotion_state,
        'thought_event': thought_event,
        'emoji_directive': emoji_directive,
        'face_motion': motion,
        'compat': {
            'mood': emotion_state['mood'],
        },
    }


def build_realtime_turn_contract(reply, mood, layers=None, directives=None):
    expression_directives = build_expression_directives(
        directives,
        fallback_mood=mood,
        fallback_text=reply,
    )
    payload_layers = deepcopy(layers) if isinstance(layers, dict) else {}
    return {
        'reply': reply,
        'mood': expression_directives['emotion_state']['mood'],
        'emotion_state': expression_directives['emotion_state'],
        'thought_event': expression_directives['thought_event'],
        'expression_directives': expression_directives,
        'emoji_directive': expression_directives['emoji_directive'],
        'layers': payload_layers,
    }


def parse_speech_directives(data):
    payload = data if isinstance(data, dict) else {}
    raw_mood = str(payload.get('mood', '')).strip().lower()

    expression_block = payload.get('expression_directives')
    emotion_block = payload.get('emotion_state')
    if not isinstance(expression_block, dict):
        expression_block = {}
    if isinstance(emotion_block, dict):
        expression_block = dict(expression_block)
        expression_block['emotion_state'] = emotion_block

    directives = build_expression_directives(
        expression_block,
        fallback_mood=_clean_mood(raw_mood or 'neutral'),
        fallback_text=payload.get('text', ''),
    )
    return directives
