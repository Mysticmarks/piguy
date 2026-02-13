(function () {
  'use strict';

  const MANIFEST_URL = '/static/models/transformers/manifest.json';

  function canUseCdnFallback() {
    return window.ALLOW_CDN_FALLBACK === true;
  }

  function markRemoteFallback(label) {
    window.__PIGUY_REMOTE_FALLBACK_ACTIVE__ = true;
    window.__PIGUY_REMOTE_FALLBACK_USED__ = window.__PIGUY_REMOTE_FALLBACK_USED__ || [];
    if (label && !window.__PIGUY_REMOTE_FALLBACK_USED__.includes(label)) {
      window.__PIGUY_REMOTE_FALLBACK_USED__.push(label);
    }
    window.dispatchEvent(new CustomEvent('piguy:remote-fallback-used', { detail: { label } }));
  }

  async function resourceExists(url) {
    try {
      const response = await fetch(url, { method: 'HEAD', cache: 'no-store' });
      return response.ok;
    } catch (_err) {
      return false;
    }
  }

  async function loadManifest() {
    const response = await fetch(MANIFEST_URL, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`Unable to load model manifest: ${response.status}`);
    }
    return response.json();
  }

  async function resolveModelFile(modelId, relativePath) {
    const manifest = await loadManifest();
    const model = (manifest.models || []).find((entry) => entry.id === modelId);
    if (!model) {
      throw new Error(`Unknown model id: ${modelId}`);
    }

    const localUrl = `${model.local_base}/${relativePath}`;
    if (await resourceExists(localUrl)) {
      return { url: localUrl, source: 'local' };
    }

    if (!canUseCdnFallback()) {
      throw new Error(`Local model file missing (${localUrl}) and ALLOW_CDN_FALLBACK is disabled.`);
    }

    const fallbackUrl = `${model.cdn_base}/${relativePath}`;
    markRemoteFallback(`model:${model.id}`);
    return { url: fallbackUrl, source: 'cdn' };
  }

  async function summarizeModelAvailability() {
    const manifest = await loadManifest();
    const summary = [];

    for (const model of manifest.models || []) {
      let localCount = 0;
      for (const filePath of model.files || []) {
        const localUrl = `${model.local_base}/${filePath}`;
        if (await resourceExists(localUrl)) localCount += 1;
      }
      summary.push({
        modelId: model.id,
        localFiles: localCount,
        totalFiles: (model.files || []).length,
        fallbackCdn: model.cdn_base
      });
    }

    return summary;
  }

  window.PiGuyModelLoader = {
    loadManifest,
    resolveModelFile,
    summarizeModelAvailability
  };
})();
