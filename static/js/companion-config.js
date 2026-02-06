(function bootstrapCompanionConfig(globalScope) {
    const MOOD_ORDER = ['neutral', 'happy', 'sad', 'angry', 'thinking', 'surprised', 'worried'];
    const MOOD_COLOR_MAP = {
        neutral: '#00ffff',
        happy: '#00ff66',
        sad: '#6688ff',
        angry: '#ff2244',
        thinking: '#ffdd00',
        surprised: '#ff6600',
        worried: '#88b2ff',
    };

    function normalizeMood(mood, fallback = 'neutral') {
        const normalized = (mood || '').toString().toLowerCase();
        return MOOD_ORDER.includes(normalized) ? normalized : fallback;
    }

    globalScope.PiGuyCompanionConfig = {
        moods: MOOD_ORDER,
        moodColorMap: MOOD_COLOR_MAP,
        normalizeMood,
    };
})(window);
