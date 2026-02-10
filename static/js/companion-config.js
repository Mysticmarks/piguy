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

    const runtimeConfig = globalScope.PiGuyRuntimeConfig || globalScope.__PIGUY_CONFIG__ || {};
    const localApiKey = globalScope.localStorage?.getItem('piguy-api-key') || '';
    const configuredApiKey = (runtimeConfig.apiKey || runtimeConfig.api_key || localApiKey || '').toString().trim();

    globalScope.PiGuyCompanionConfig = {
        moods: MOOD_ORDER,
        moodColorMap: MOOD_COLOR_MAP,
        normalizeMood,
        apiKey: configuredApiKey,
    };
})(window);
