/**
 * Gradient Serverless Chat 前端逻辑（TypeScript in JS 风格）。
 * 采用模块脚本直接运行，无第三方依赖。
 */

const DEFAULT_SETTINGS = {
  baseUrl: 'https://inference.do-ai.run',
  model: 'llama3.3-70b-instruct',
  temperature: 0.2,
  topP: 0.95,
  maxTokens: 512,
  persistKey: false,
  systemPrompt: '',
};

/** @typedef {{ prompt_tokens?: number; completion_tokens?: number; total_tokens?: number; }} TokenUsage */
/** @typedef {{ role: 'user' | 'assistant' | 'system'; content: string; }} ChatHistoryItem */
/**
 * @typedef MessageRecord
 * @property {string} id
 * @property {'user' | 'assistant' | 'system'} role
 * @property {string} content
 * @property {number} createdAt
 * @property {boolean=} streaming
 * @property {boolean=} queued
 * @property {boolean=} isAuto
 * @property {string=} finishReason
 * @property {TokenUsage=} usage
 * @property {string=} requestId
 * @property {string=} error
 * @property {string=} note
 */

/**
 * 生成唯一 ID。
 * @param {string} prefix
 * @returns {string}
 */
function createId(prefix) {
  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

/**
 * 解析数字表单值。
 * @param {string} value
 * @param {number} fallback
 * @param {number} min
 * @param {number} max
 * @returns {number}
 */
function parseNumberInput(value, fallback, min, max) {
  const numeric = Number(value);
  if (Number.isFinite(numeric)) {
    return Math.min(max, Math.max(min, numeric));
  }
  return fallback;
}

/**
 * 指数退避工具。
 */
class Backoff {
  /**
   * 计算指数退避延迟（毫秒）。
   * @param {number} attempt 从 1 开始计数
   * @returns {number}
   */
  static exponential(attempt) {
    const cappedAttempt = Math.max(1, Math.min(attempt, 5));
    return Math.min(8000, 1000 * Math.pow(2, cappedAttempt - 1));
  }

  /**
   * 延迟一段时间。
   * @param {number} ms
   * @param {AbortSignal=} signal
   * @returns {Promise<void>}
   */
  static wait(ms, signal) {
    return new Promise((resolve, reject) => {
      if (signal && signal.aborted) {
        reject(new DOMException('Aborted', 'AbortError'));
        return;
      }
      const timeout = window.setTimeout(() => {
        cleanup();
        resolve();
      }, ms);
      const onAbort = () => {
        cleanup();
        reject(new DOMException('Aborted', 'AbortError'));
      };
      const cleanup = () => {
        window.clearTimeout(timeout);
        if (signal) {
          signal.removeEventListener('abort', onAbort);
        }
      };
      if (signal) {
        signal.addEventListener('abort', onAbort, { once: true });
      }
    });
  }
}

/**
 * API 错误。
 */
class ApiError extends Error {
  /**
   * @param {string} message
   * @param {number} status
   * @param {number=} retryAfterMs
   * @param {string=} requestId
   */
  constructor(message, status, retryAfterMs, requestId) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.retryAfterMs = retryAfterMs;
    this.requestId = requestId || null;
  }
}

/**
 * 指标采集。
 */
class Metrics {
  constructor() {
    /** @type {number} */
    this.totalPromptTokens = 0;
    /** @type {number} */
    this.totalCompletionTokens = 0;
    /** @type {number} */
    this.totalTokens = 0;
    /** @type {{ timestamp: number; value: number; }[]} */
    this.ttftLog = [];
    /** @type {number[]} */
    this.recent429 = [];
    /** @type {number[]} */
    this.recentRetries = [];
    /** @type {Map<string, number>} */
    this.failureCounts = new Map();
  }

  /**
   * 累加 usage。
   * @param {TokenUsage|undefined|null} usage
   */
  recordUsage(usage) {
    if (!usage) {
      return;
    }
    if (typeof usage.prompt_tokens === 'number') {
      this.totalPromptTokens += usage.prompt_tokens;
    }
    if (typeof usage.completion_tokens === 'number') {
      this.totalCompletionTokens += usage.completion_tokens;
    }
    if (typeof usage.total_tokens === 'number') {
      this.totalTokens += usage.total_tokens;
    } else {
      const total = (usage.prompt_tokens || 0) + (usage.completion_tokens || 0);
      this.totalTokens += total;
    }
  }

  /**
   * 记录 TTFT。
   * @param {number} ms
   */
  recordTtft(ms) {
    const value = Math.max(0, Math.round(ms));
    const entry = { timestamp: Date.now(), value };
    this.ttftLog.push(entry);
    if (this.ttftLog.length > 20) {
      this.ttftLog.shift();
    }
  }

  /**
   * 记录 429。
   */
  record429() {
    this.recent429.push(Date.now());
    this.pruneWindow(this.recent429, 5 * 60 * 1000);
  }

  /**
   * 记录重试。
   */
  recordRetry() {
    this.recentRetries.push(Date.now());
    this.pruneWindow(this.recentRetries, 5 * 60 * 1000);
  }

  /**
   * 记录失败类型。
   * @param {string} type
   */
  recordFailure(type) {
    const current = this.failureCounts.get(type) || 0;
    this.failureCounts.set(type, current + 1);
  }

  /**
   * 剔除过期样本。
   * @param {number[]} list
   * @param {number} windowMs
   */
  pruneWindow(list, windowMs) {
    const cutoff = Date.now() - windowMs;
    while (list.length && list[0] < cutoff) {
      list.shift();
    }
  }

  /**
   * 统计窗口。
   * @param {number[]} list
   * @returns {number}
   */
  countWindow(list) {
    this.pruneWindow(list, 5 * 60 * 1000);
    return list.length;
  }

  /**
   * 计算 p95。
   * @returns {number|null}
   */
  getTtftP95() {
    if (!this.ttftLog.length) {
      return null;
    }
    const values = this.ttftLog.map((item) => item.value).slice().sort((a, b) => a - b);
    const idx = Math.max(0, Math.ceil(values.length * 0.95) - 1);
    return values[idx];
  }

  /**
   * 导出 NDJSON。
   * @returns {string}
   */
  exportNdjson() {
    const lines = [];
    lines.push(JSON.stringify({
      type: 'totals',
      timestamp: Date.now(),
      prompt_tokens: this.totalPromptTokens,
      completion_tokens: this.totalCompletionTokens,
      total_tokens: this.totalTokens,
    }));
    for (const entry of this.ttftLog) {
      lines.push(JSON.stringify({ type: 'ttft', timestamp: entry.timestamp, value_ms: entry.value }));
    }
    lines.push(JSON.stringify({ type: 'window', timestamp: Date.now(), code429_5m: this.countWindow(this.recent429), retries_5m: this.countWindow(this.recentRetries) }));
    for (const [failure, count] of this.failureCounts.entries()) {
      lines.push(JSON.stringify({ type: 'failure', failure, count }));
    }
    return lines.join('\n');
  }

  /**
   * 获取快照。
   * @returns {{ prompt: number; completion: number; total: number; ttftList: number[]; ttftP95: number|null; code429: number; retries: number; failures: [string, number][]; }}
   */
  getSnapshot() {
    return {
      prompt: this.totalPromptTokens,
      completion: this.totalCompletionTokens,
      total: this.totalTokens,
      ttftList: this.ttftLog.map((entry) => entry.value),
      ttftP95: this.getTtftP95(),
      code429: this.countWindow(this.recent429),
      retries: this.countWindow(this.recentRetries),
      failures: Array.from(this.failureCounts.entries()),
    };
  }
}

/**
 * 持久化与状态。
 */
class Store {
  constructor() {
    this.settingsKey = 'gradient-settings-v1';
    this.sessionKeyKey = 'gradient-session-key';
    /** @type {ReturnType<Store['loadSettings']>} */
    this.settings = this.loadSettings();
    /** @type {string} */
    this.apiKey = '';
    this.restoreSessionKey();
  }

  /**
   * @returns {{ baseUrl: string; model: string; temperature: number; topP: number; maxTokens: number; persistKey: boolean; }}
   */
  loadSettings() {
    try {
      const raw = window.localStorage.getItem(this.settingsKey);
      if (raw) {
        const parsed = JSON.parse(raw);
        return {
          baseUrl: typeof parsed.baseUrl === 'string' ? parsed.baseUrl : DEFAULT_SETTINGS.baseUrl,
          model: typeof parsed.model === 'string' ? parsed.model : DEFAULT_SETTINGS.model,
          temperature: Number.isFinite(parsed.temperature) ? parsed.temperature : DEFAULT_SETTINGS.temperature,
          topP: Number.isFinite(parsed.topP) ? parsed.topP : DEFAULT_SETTINGS.topP,
          maxTokens: Number.isFinite(parsed.maxTokens) ? parsed.maxTokens : DEFAULT_SETTINGS.maxTokens,
          persistKey: Boolean(parsed.persistKey),
          systemPrompt: typeof parsed.systemPrompt === 'string' ? parsed.systemPrompt : DEFAULT_SETTINGS.systemPrompt,
        };
      }
    } catch (error) {
      console.warn('Failed to parse settings', error);
    }
    return { ...DEFAULT_SETTINGS };
  }

  /**
   * @returns {{ baseUrl: string; model: string; temperature: number; topP: number; maxTokens: number; persistKey: boolean; systemPrompt: string; }}
   */
  getSettings() {
    return { ...this.settings };
  }

  /**
   * @param {Partial<ReturnType<Store['getSettings']>>} updates
   */
  saveSettings(updates) {
    this.settings = { ...this.settings, ...updates };
    try {
      window.localStorage.setItem(this.settingsKey, JSON.stringify(this.settings));
    } catch (error) {
      console.warn('Failed to persist settings', error);
    }
  }

  restoreSessionKey() {
    try {
      const stored = window.sessionStorage.getItem(this.sessionKeyKey);
      if (stored) {
        this.apiKey = stored;
        this.settings.persistKey = true;
      }
    } catch (error) {
      console.warn('Unable to access sessionStorage', error);
    }
  }

  /**
   * @returns {string}
   */
  getApiKey() {
    return this.apiKey;
  }

  /**
   * @param {string} key
   * @param {boolean} persist
   */
  setApiKey(key, persist) {
    this.apiKey = key;
    this.settings.persistKey = persist;
    try {
      if (persist && key) {
        window.sessionStorage.setItem(this.sessionKeyKey, key);
      } else {
        window.sessionStorage.removeItem(this.sessionKeyKey);
      }
    } catch (error) {
      console.warn('Unable to update sessionStorage', error);
    }
  }

  clearKey() {
    this.apiKey = '';
    this.settings.persistKey = false;
    try {
      window.sessionStorage.removeItem(this.sessionKeyKey);
    } catch (error) {
      console.warn('Unable to clear sessionStorage', error);
    }
  }
}

/**
 * SSE 解析器。
 */
class SseParser {
  constructor() {
    this.decoder = new TextDecoder();
  }

  /**
   * @param {ReadableStream<Uint8Array>} stream
   * @param {{ onMessage: (data: unknown) => void; onDone: () => void; onError?: (error: Error) => void; }} handlers
   * @returns {Promise<void>}
   */
  async parse(stream, handlers) {
    const reader = stream.getReader();
    let buffer = '';
    /** @type {string[]} */
    let eventLines = [];
    const flushEvent = () => {
      if (!eventLines.length) {
        return false;
      }
      const dataLines = eventLines
        .filter((line) => line.startsWith('data:'))
        .map((line) => line.slice(5).trimStart());
      eventLines = [];
      if (!dataLines.length) {
        return false;
      }
      const payload = dataLines.join('\n');
      if (payload === '[DONE]') {
        handlers.onDone();
        return true;
      }
      try {
        const parsed = JSON.parse(payload);
        handlers.onMessage(parsed);
      } catch (error) {
        if (handlers.onError) {
          handlers.onError(error instanceof Error ? error : new Error(String(error)));
        } else {
          console.error('Failed to parse SSE payload', error);
        }
      }
      return false;
    };

    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        if (buffer.length) {
          buffer += '\n';
        }
      } else if (value) {
        buffer += this.decoder.decode(value, { stream: true });
      }
      let newlineIndex = buffer.indexOf('\n');
      while (newlineIndex !== -1) {
        const rawLine = buffer.slice(0, newlineIndex);
        buffer = buffer.slice(newlineIndex + 1);
        const line = rawLine.replace(/\r$/, '');
        if (line === '') {
          const terminated = flushEvent();
          if (terminated) {
            await reader.cancel();
            return;
          }
        } else {
          eventLines.push(line);
        }
        newlineIndex = buffer.indexOf('\n');
      }
      if (done) {
        flushEvent();
        handlers.onDone();
        return;
      }
    }
  }
}

/**
 * API 客户端。
 */
class ApiClient {
  /**
   * @param {Store} store
   * @param {Metrics} metrics
   */
  constructor(store, metrics) {
    this.store = store;
    this.metrics = metrics;
    this.parser = new SseParser();
  }

  /**
   * 构建完整 URL。
   * @param {string} path
   * @returns {string}
   */
  buildUrl(path) {
    const base = this.store.getSettings().baseUrl.replace(/\/$/, '');
    return `${base}${path}`;
  }

  /**
   * @param {Headers} headers
   * @returns {string|null}
   */
  extractRequestId(headers) {
    return (
      headers.get('x-request-id') ||
      headers.get('request-id') ||
      headers.get('openai-request-id') ||
      null
    );
  }

  /**
   * @returns {boolean}
   */
  isStreamCapable() {
    return typeof ReadableStream !== 'undefined';
  }

  /**
   * 列出模型。
   * @param {AbortSignal=} signal
   * @returns {Promise<{ models: string[]; requestId: string|null; }>}
   */
  async listModels(signal) {
    const url = this.buildUrl('/v1/models');
    const key = this.store.getApiKey();
    const headers = new Headers({ Accept: 'application/json' });
    if (key) {
      headers.set('Authorization', `Bearer ${key}`);
    }
    const response = await fetch(url, { method: 'GET', headers, signal });
    const requestId = this.extractRequestId(response.headers);
    if (!response.ok) {
      let message = `HTTP ${response.status}`;
      try {
        const body = await response.json();
        if (body && body.error && typeof body.error.message === 'string') {
          message = body.error.message;
        }
      } catch (error) {
        // ignore
      }
      throw new ApiError(message, response.status, undefined, requestId || undefined);
    }
    const data = await response.json();
    const models = Array.isArray(data?.data)
      ? data.data
          .map((item) => (item && typeof item.id === 'string' ? item.id : null))
          .filter((id) => typeof id === 'string')
      : [];
    return { models, requestId };
  }

  /**
   * 发送 chat.completions 请求。
   * @param {{ model: string; messages: ChatHistoryItem[]; temperature: number; topP: number; maxTokens: number; stream: boolean; }} payload
   * @param {{
   *   signal: AbortSignal;
   *   onChunk: (text: string) => void;
   *   onUsage: (usage: TokenUsage|null) => void;
   *   onFinish: (result: { finishReason: string|null; usage: TokenUsage|null; stream: boolean; }) => void;
   *   onFirstToken: (ms: number) => void;
   *   onRequestId: (id: string|null) => void;
   * }} hooks
   * @returns {Promise<{ requestId: string|null; stream: boolean; }>}
   */
  async chat(payload, hooks) {
    const url = this.buildUrl('/v1/chat/completions');
    const key = this.store.getApiKey();
    if (!key) {
      throw new ApiError('请先在设置中填写 API Key。', 401);
    }
    const headers = new Headers({
      'Content-Type': 'application/json',
      Accept: 'application/json',
      Authorization: `Bearer ${key}`,
    });
    const preferStream = Boolean(payload.stream);
    const canStream = preferStream && this.isStreamCapable();
    const requestBody = {
      model: payload.model,
      messages: payload.messages,
      temperature: payload.temperature,
      top_p: payload.topP,
      max_tokens: payload.maxTokens,
      stream: canStream,
    };
    let response;
    try {
      response = await fetch(url, {
        method: 'POST',
        headers,
        body: JSON.stringify(requestBody),
        signal: hooks.signal,
      });
    } catch (error) {
      throw new ApiError('网络错误或请求被阻断。', 0);
    }
    const requestId = this.extractRequestId(response.headers);
    hooks.onRequestId(requestId);
    if (!response.ok) {
      let message = `HTTP ${response.status}`;
      let retryAfterMs;
      const retryAfter = response.headers.get('Retry-After');
      if (retryAfter) {
        const seconds = Number(retryAfter);
        if (Number.isFinite(seconds)) {
          retryAfterMs = seconds * 1000;
        }
      }
      try {
        const body = await response.json();
        if (body && body.error && typeof body.error.message === 'string') {
          message = body.error.message;
        }
      } catch (error) {
        // ignore body parse errors
      }
      throw new ApiError(message, response.status, retryAfterMs, requestId || undefined);
    }

    const startedAt = performance.now();
    let firstTokenReported = false;
    let finishReason = null;
    let usage = null;
    const reportFirstToken = () => {
      if (!firstTokenReported) {
        firstTokenReported = true;
        hooks.onFirstToken(performance.now() - startedAt);
      }
    };

    if (canStream && response.body) {
      await this.parser.parse(response.body, {
        onMessage: (data) => {
          const chunk = /** @type {{ choices?: any[]; usage?: TokenUsage; }} */ (data);
          if (chunk.usage) {
            usage = chunk.usage;
            hooks.onUsage(chunk.usage);
          }
          const choice = Array.isArray(chunk.choices) ? chunk.choices[0] : null;
          if (choice) {
            if (choice.delta && typeof choice.delta.content === 'string') {
              const text = choice.delta.content;
              if (text) {
                hooks.onChunk(text);
                reportFirstToken();
              }
            }
            if (choice.finish_reason) {
              finishReason = choice.finish_reason;
            }
          }
        },
        onDone: () => {
          hooks.onFinish({ finishReason, usage, stream: true });
        },
        onError: (error) => {
          console.error('SSE parse error', error);
        },
      });
      return { requestId, stream: true };
    }

    try {
      const body = await response.json();
      const choice = Array.isArray(body?.choices) ? body.choices[0] : null;
      const text = choice && choice.message && typeof choice.message.content === 'string' ? choice.message.content : '';
      if (text) {
        hooks.onChunk(text);
      }
      if (body && body.usage) {
        usage = body.usage;
        hooks.onUsage(body.usage);
      } else {
        hooks.onUsage(null);
      }
      reportFirstToken();
      finishReason = choice ? choice.finish_reason || null : null;
      hooks.onFinish({ finishReason, usage, stream: false });
    } catch (error) {
      throw new ApiError('解析响应失败。', 500, undefined, requestId || undefined);
    }
    return { requestId, stream: false };
  }
}

/**
 * UI 渲染与交互。
 */
class UI {
  constructor() {
    const chatList = /** @type {HTMLDivElement|null} */ (document.getElementById('chat-list'));
    const queueIndicator = /** @type {HTMLDivElement|null} */ (document.getElementById('queue-indicator'));
    const toastContainer = /** @type {HTMLDivElement|null} */ (document.getElementById('toast-container'));
    const monitorPanel = /** @type {HTMLDivElement|null} */ (document.getElementById('monitor-panel'));
    const monitorToggle = /** @type {HTMLButtonElement|null} */ (document.getElementById('monitor-toggle'));
    const monitorBody = /** @type {HTMLElement|null} */ (document.getElementById('monitor-body'));
    const metricPrompt = /** @type {HTMLSpanElement|null} */ (document.getElementById('metric-prompt'));
    const metricCompletion = /** @type {HTMLSpanElement|null} */ (document.getElementById('metric-completion'));
    const metricTotal = /** @type {HTMLSpanElement|null} */ (document.getElementById('metric-total'));
    const metric429 = /** @type {HTMLSpanElement|null} */ (document.getElementById('metric-429'));
    const metricRetries = /** @type {HTMLSpanElement|null} */ (document.getElementById('metric-retries'));
    const metricTtftP95 = /** @type {HTMLSpanElement|null} */ (document.getElementById('metric-ttft-p95'));
    const metricTtftList = /** @type {HTMLUListElement|null} */ (document.getElementById('metric-ttft-list'));
    const metricFailures = /** @type {HTMLUListElement|null} */ (document.getElementById('metric-failures'));
    const metricsExport = /** @type {HTMLButtonElement|null} */ (document.getElementById('metrics-export'));
    const settingsModal = /** @type {HTMLDivElement|null} */ (document.getElementById('settings-modal'));
    const settingsForm = /** @type {HTMLFormElement|null} */ (document.getElementById('settings-form'));
    const baseUrlInput = /** @type {HTMLInputElement|null} */ (document.getElementById('setting-base-url'));
    const apiKeyInput = /** @type {HTMLInputElement|null} */ (document.getElementById('setting-api-key'));
    const persistCheckbox = /** @type {HTMLInputElement|null} */ (document.getElementById('setting-persist'));
    const modelInput = /** @type {HTMLInputElement|null} */ (document.getElementById('setting-model'));
    const modelOptions = /** @type {HTMLDataListElement|null} */ (document.getElementById('model-options'));
    const temperatureInput = /** @type {HTMLInputElement|null} */ (document.getElementById('setting-temperature'));
    const topPInput = /** @type {HTMLInputElement|null} */ (document.getElementById('setting-top-p'));
    const maxTokensInput = /** @type {HTMLInputElement|null} */ (document.getElementById('setting-max-tokens'));
    const systemPromptInput = /** @type {HTMLTextAreaElement|null} */ (document.getElementById('setting-system-prompt'));
    const modelsStatus = /** @type {HTMLParagraphElement|null} */ (document.getElementById('models-status'));
    const modelsRefresh = /** @type {HTMLButtonElement|null} */ (document.getElementById('models-refresh'));
    const connectionTest = /** @type {HTMLButtonElement|null} */ (document.getElementById('connection-test'));
    const saveSettings = /** @type {HTMLButtonElement|null} */ (document.getElementById('save-settings'));
    const clearKeyButton = /** @type {HTMLButtonElement|null} */ (document.getElementById('clear-key'));
    const settingsButton = /** @type {HTMLButtonElement|null} */ (document.getElementById('settings-button'));
    const settingsClose = /** @type {HTMLButtonElement|null} */ (document.getElementById('settings-close'));
    const clearButton = /** @type {HTMLButtonElement|null} */ (document.getElementById('clear-button'));
    const exportButton = /** @type {HTMLButtonElement|null} */ (document.getElementById('export-button'));
    const sendButton = /** @type {HTMLButtonElement|null} */ (document.getElementById('send-button'));
    const textarea = /** @type {HTMLTextAreaElement|null} */ (document.getElementById('chat-input'));

    if (!chatList || !queueIndicator || !toastContainer || !monitorPanel || !monitorToggle || !monitorBody ||
      !metricPrompt || !metricCompletion || !metricTotal || !metric429 || !metricRetries || !metricTtftP95 ||
      !metricTtftList || !metricFailures || !metricsExport || !settingsModal || !settingsForm || !baseUrlInput ||
      !apiKeyInput || !persistCheckbox || !modelInput || !modelOptions || !temperatureInput || !topPInput || !maxTokensInput || !systemPromptInput ||
      !modelsStatus || !modelsRefresh || !connectionTest || !saveSettings || !clearKeyButton || !settingsButton || !settingsClose ||
      !clearButton || !exportButton || !sendButton || !textarea) {
      throw new Error('关键元素缺失，无法初始化 UI');
    }

    this.chatList = chatList;
    this.queueIndicator = queueIndicator;
    this.toastContainer = toastContainer;
    this.monitorPanel = monitorPanel;
    this.monitorToggle = monitorToggle;
    this.monitorBody = monitorBody;
    this.metricPrompt = metricPrompt;
    this.metricCompletion = metricCompletion;
    this.metricTotal = metricTotal;
    this.metric429 = metric429;
    this.metricRetries = metricRetries;
    this.metricTtftP95 = metricTtftP95;
    this.metricTtftList = metricTtftList;
    this.metricFailures = metricFailures;
    this.metricsExportButton = metricsExport;
    this.settingsModal = settingsModal;
    this.settingsForm = settingsForm;
    this.baseUrlInput = baseUrlInput;
    this.apiKeyInput = apiKeyInput;
    this.persistCheckbox = persistCheckbox;
    this.modelInput = modelInput;
    this.modelOptions = modelOptions;
    this.temperatureInput = temperatureInput;
    this.topPInput = topPInput;
    this.maxTokensInput = maxTokensInput;
    this.systemPromptInput = systemPromptInput;
    this.modelsStatus = modelsStatus;
    this.modelsRefreshButton = modelsRefresh;
    this.connectionTestButton = connectionTest;
    this.saveSettingsButton = saveSettings;
    this.clearKeyButton = clearKeyButton;
    this.settingsButton = settingsButton;
    this.settingsCloseButton = settingsClose;
    this.clearButton = clearButton;
    this.exportButton = exportButton;
    this.sendButton = sendButton;
    this.textarea = textarea;

    /** @type {Map<string, { element: HTMLElement; contentEl: HTMLElement; metaEl: HTMLElement; data: MessageRecord; cancelButton: HTMLButtonElement|null; retryButton: HTMLButtonElement|null; requestCopyButton: HTMLButtonElement|null; contentCopyButton: HTMLButtonElement|null; }>} */
    this.messageMap = new Map();

    this.monitorToggle.addEventListener('click', () => {
      const collapsed = this.monitorPanel.classList.toggle('collapsed');
      this.monitorToggle.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
      this.monitorToggle.textContent = collapsed ? '展开' : '收起';
      if (!collapsed) {
        this.monitorBody.focus?.();
      }
    });

    this.settingsModal.addEventListener('click', (event) => {
      if (event.target === this.settingsModal) {
        this.closeSettings();
      }
    });

    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && this.isSettingsOpen()) {
        this.closeSettings();
      }
    });
  }

  scrollToBottom() {
    this.chatList.scrollTop = this.chatList.scrollHeight;
  }

  /**
   * 添加消息气泡。
   * @param {MessageRecord} record
   */
  addMessage(record, position = 'end') {
    const article = document.createElement('article');
    article.className = `message message-${record.role}`;
    if (record.streaming) {
      article.classList.add('streaming');
    }
    if (record.error) {
      article.classList.add('message-error');
    }
    article.dataset.messageId = record.id;
    const content = document.createElement('div');
    content.className = 'message-content';
    content.textContent = record.content;
    const meta = document.createElement('div');
    meta.className = 'message-meta';
    article.append(content, meta);
    if (position === 'start') {
      this.chatList.prepend(article);
    } else {
      this.chatList.append(article);
    }
    this.messageMap.set(record.id, {
      element: article,
      contentEl: content,
      metaEl: meta,
      data: { ...record },
      cancelButton: null,
      retryButton: null,
      requestCopyButton: null,
      contentCopyButton: null,
    });
    this.renderMeta(record.id);
    if (position !== 'start') {
      this.scrollToBottom();
    }
  }

  /**
   * 更新消息数据。
   * @param {string} id
   * @param {Partial<MessageRecord>} updates
   */
  updateMessage(id, updates) {
    const entry = this.messageMap.get(id);
    if (!entry) {
      return;
    }
    entry.data = { ...entry.data, ...updates };
    if (typeof updates.content === 'string') {
      entry.contentEl.textContent = updates.content;
      this.scrollToBottom();
    }
    if (typeof updates.streaming === 'boolean') {
      entry.element.classList.toggle('streaming', updates.streaming);
    }
    if (typeof updates.error !== 'undefined') {
      entry.element.classList.toggle('message-error', Boolean(updates.error));
    }
    if (typeof updates.queued === 'boolean') {
      entry.data.queued = updates.queued;
    }
    this.renderMeta(id);
  }

  /**
   * 更新元信息区域。
   * @param {string} id
   */
  renderMeta(id) {
    const entry = this.messageMap.get(id);
    if (!entry) {
      return;
    }
    const { metaEl } = entry;
    const data = entry.data;
    metaEl.textContent = '';
    if (entry.requestCopyButton) {
      entry.requestCopyButton.remove();
      entry.requestCopyButton = null;
    }
    if (entry.contentCopyButton) {
      entry.contentCopyButton.remove();
      entry.contentCopyButton = null;
    }
    const badges = [];
    if (data.isAuto) {
      badges.push('自动续写');
    }
    if (data.streaming) {
      badges.push('生成中');
    }
    if (data.queued) {
      badges.push('队列中…');
    }
    if (data.note) {
      badges.push(data.note);
    }
    if (data.finishReason) {
      badges.push(`finish_reason: ${data.finishReason}`);
    }
    if (data.usage) {
      const usage = data.usage;
      const prompt = typeof usage.prompt_tokens === 'number' ? usage.prompt_tokens : '-';
      const completion = typeof usage.completion_tokens === 'number' ? usage.completion_tokens : '-';
      const total = typeof usage.total_tokens === 'number' ? usage.total_tokens : '-';
      badges.push(`usage p:${prompt} c:${completion} t:${total}`);
    }
    if (data.error) {
      badges.push(`错误: ${data.error}`);
    }
    for (const badge of badges) {
      const span = document.createElement('span');
      span.textContent = badge;
      metaEl.append(span);
    }
    if (data.content && data.content.length > 0) {
      const copyContentButton = document.createElement('button');
      copyContentButton.type = 'button';
      copyContentButton.className = 'ghost';
      copyContentButton.textContent = '复制内容';
      copyContentButton.addEventListener('click', () => {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(data.content).then(() => {
            this.showToast('消息内容已复制。', 'success', { duration: 2000 });
          }).catch(() => {
            this.showToast('复制失败，请手动选择文本。', 'error', { duration: 2500 });
          });
        } else {
          this.showToast('此浏览器不支持剪贴板 API。', 'error', { duration: 2500 });
        }
      });
      metaEl.append(copyContentButton);
      entry.contentCopyButton = copyContentButton;
    }
    if (data.requestId) {
      const span = document.createElement('span');
      span.textContent = `request_id: ${data.requestId}`;
      metaEl.append(span);
      const copyButton = document.createElement('button');
      copyButton.type = 'button';
      copyButton.className = 'ghost';
      copyButton.textContent = '复制 request_id';
      copyButton.addEventListener('click', () => {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(data.requestId).then(() => {
            this.showToast('已复制 request_id。', 'success', { duration: 2000 });
          }).catch(() => {
            this.showToast('复制失败，请手动选择文本。', 'error', { duration: 2500 });
          });
        } else {
          this.showToast('此浏览器不支持剪贴板 API。', 'error', { duration: 2500 });
        }
      });
      metaEl.append(copyButton);
      entry.requestCopyButton = copyButton;
    }
    if (entry.cancelButton) {
      metaEl.append(entry.cancelButton);
    }
    if (entry.retryButton) {
      metaEl.append(entry.retryButton);
    }
  }

  /**
   * 设置消息备注。
   * @param {string} id
   * @param {string|undefined} note
   */
  setMessageNote(id, note) {
    this.updateMessage(id, { note });
  }

  /**
   * 添加取消按钮。
   * @param {string} id
   * @param {() => void} handler
   */
  addCancelButton(id, handler) {
    const entry = this.messageMap.get(id);
    if (!entry) {
      return;
    }
    if (entry.cancelButton) {
      entry.cancelButton.remove();
    }
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'ghost';
    button.textContent = '取消';
    button.addEventListener('click', handler);
    entry.cancelButton = button;
    entry.metaEl.append(button);
  }

  removeCancelButton(id) {
    const entry = this.messageMap.get(id);
    if (entry && entry.cancelButton) {
      entry.cancelButton.remove();
      entry.cancelButton = null;
      this.renderMeta(id);
    }
  }

  addRetryButton(id, handler) {
    const entry = this.messageMap.get(id);
    if (!entry) {
      return;
    }
    if (entry.retryButton) {
      entry.retryButton.remove();
    }
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'ghost';
    button.textContent = '立即重试';
    button.addEventListener('click', handler);
    entry.retryButton = button;
    entry.metaEl.append(button);
  }

  removeRetryButton(id) {
    const entry = this.messageMap.get(id);
    if (entry && entry.retryButton) {
      entry.retryButton.remove();
      entry.retryButton = null;
      this.renderMeta(id);
    }
  }

  /**
   * 清理所有消息。
   */
  clearChat() {
    this.chatList.textContent = '';
    this.messageMap.clear();
  }

  removeMessage(id) {
    const entry = this.messageMap.get(id);
    if (!entry) {
      return;
    }
    entry.element.remove();
    this.messageMap.delete(id);
  }

  /**
   * 更新队列状态。
   * @param {number} queued
   * @param {number} active
   */
  updateQueueStatus(queued, active) {
    if (queued === 0 && active === 0) {
      this.queueIndicator.textContent = '空闲';
    } else {
      this.queueIndicator.textContent = `进行中 ${active} | 排队 ${queued}`;
    }
  }

  /**
   * 展示提示。
   * @param {string} message
   * @param {'info'|'success'|'error'} type
   * @param {{ duration?: number; action?: { label: string; onClick: () => void; }; }} [options]
   * @returns {() => void}
   */
  showToast(message, type = 'info', options) {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    const span = document.createElement('span');
    span.textContent = message;
    toast.append(span);
    let dismissed = false;
    const dismiss = () => {
      if (dismissed) {
        return;
      }
      dismissed = true;
      toast.remove();
    };
    if (options && options.action) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'ghost';
      button.textContent = options.action.label;
      button.addEventListener('click', () => {
        options.action.onClick();
        dismiss();
      });
      toast.append(button);
    }
    this.toastContainer.append(toast);
    const duration = options && typeof options.duration === 'number' ? options.duration : 4000;
    if (duration > 0) {
      window.setTimeout(dismiss, duration);
    }
    return dismiss;
  }

  /**
   * 更新指标显示。
   * @param {{ prompt: number; completion: number; total: number; ttftList: number[]; ttftP95: number|null; code429: number; retries: number; failures: [string, number][]; }} snapshot
   */
  updateMetrics(snapshot) {
    this.metricPrompt.textContent = String(snapshot.prompt);
    this.metricCompletion.textContent = String(snapshot.completion);
    this.metricTotal.textContent = String(snapshot.total);
    this.metric429.textContent = String(snapshot.code429);
    this.metricRetries.textContent = String(snapshot.retries);
    this.metricTtftP95.textContent = snapshot.ttftP95 != null ? `${snapshot.ttftP95} ms` : '--';
    this.metricTtftList.textContent = '';
    snapshot.ttftList.forEach((value, index) => {
      const li = document.createElement('li');
      li.textContent = `${index + 1}. ${value} ms`;
      this.metricTtftList.append(li);
    });
    this.metricFailures.textContent = '';
    snapshot.failures.forEach(([type, count]) => {
      const li = document.createElement('li');
      li.textContent = `${type}: ${count}`;
      this.metricFailures.append(li);
    });
  }

  setModelsStatus(message) {
    this.modelsStatus.textContent = message;
  }

  setModelOptions(models, selected) {
    this.modelOptions.textContent = '';
    const unique = Array.from(new Set(models));
    unique.forEach((model) => {
      const option = document.createElement('option');
      option.value = model;
      this.modelOptions.append(option);
    });
    if (selected) {
      this.modelInput.value = selected;
    }
  }

  fillSettingsForm(settings, apiKey) {
    this.baseUrlInput.value = settings.baseUrl;
    this.modelInput.value = settings.model;
    this.temperatureInput.value = String(settings.temperature);
    this.topPInput.value = String(settings.topP);
    this.maxTokensInput.value = String(settings.maxTokens);
    this.persistCheckbox.checked = Boolean(settings.persistKey);
    this.apiKeyInput.value = apiKey;
    this.systemPromptInput.value = settings.systemPrompt || '';
  }

  /**
   * 获取设置表单值。
   * @returns {{ baseUrl: string; apiKey: string; persist: boolean; model: string; temperature: number; topP: number; maxTokens: number; systemPrompt: string; }}
   */
  readSettingsForm() {
    return {
      baseUrl: this.baseUrlInput.value.trim(),
      apiKey: this.apiKeyInput.value.trim(),
      persist: this.persistCheckbox.checked,
      model: this.modelInput.value.trim(),
      temperature: parseNumberInput(this.temperatureInput.value, DEFAULT_SETTINGS.temperature, 0, 2),
      topP: parseNumberInput(this.topPInput.value, DEFAULT_SETTINGS.topP, 0, 1),
      maxTokens: Math.round(parseNumberInput(this.maxTokensInput.value, DEFAULT_SETTINGS.maxTokens, 16, 4096)),
      systemPrompt: this.systemPromptInput.value.trim(),
    };
  }

  openSettings() {
    this.settingsModal.classList.remove('hidden');
    this.settingsModal.setAttribute('aria-hidden', 'false');
    window.setTimeout(() => {
      this.baseUrlInput.focus();
    }, 0);
  }

  closeSettings() {
    this.settingsModal.classList.add('hidden');
    this.settingsModal.setAttribute('aria-hidden', 'true');
  }

  isSettingsOpen() {
    return !this.settingsModal.classList.contains('hidden');
  }

  resetInput() {
    this.textarea.value = '';
    this.autoResizeTextarea();
  }

  focusInput() {
    this.textarea.focus();
  }

  autoResizeTextarea() {
    this.textarea.style.height = 'auto';
    const maxHeight = 240;
    const next = Math.min(maxHeight, this.textarea.scrollHeight + 2);
    this.textarea.style.height = `${next}px`;
  }

  /**
   * 下载文本文件。
   * @param {string} filename
   * @param {string} content
   * @param {string} mime
   */
  downloadText(filename, content, mime) {
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = filename;
    anchor.rel = 'noopener';
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 5000);
  }
}

/** @typedef {{ id: string; assistantMessage: MessageRecord; history: ChatHistoryItem[]; settings: { model: string; temperature: number; topP: number; maxTokens: number; }; preferStream: boolean; autoAttempt: number; }} ChatJob */

/**
 * 单次对话请求。
 */
class ChatRequest {
  /**
   * @param {{ job: ChatJob; apiClient: ApiClient; metrics: Metrics; ui: UI; }} options
   */
  constructor(options) {
    this.job = options.job;
    this.apiClient = options.apiClient;
    this.metrics = options.metrics;
    this.ui = options.ui;
    this.controller = new AbortController();
    this.currentContent = '';
    this.finishReason = null;
    this.usage = null;
    this.requestId = null;
  }

  cancel() {
    this.controller.abort();
  }

  /**
   * 执行请求。
   * @returns {Promise<{ success: boolean; canceled?: boolean; content: string; finishReason?: string|null; usage?: TokenUsage|null; requestId?: string|null; autoAttempt: number; }>} 
   */
  async start() {
    const { assistantMessage } = this.job;
    this.ui.updateMessage(assistantMessage.id, { queued: false, streaming: true, error: undefined, note: undefined, content: assistantMessage.content });
    this.ui.addCancelButton(assistantMessage.id, () => this.cancel());
    try {
      const result = await this.execute();
      return result;
    } finally {
      this.ui.removeCancelButton(assistantMessage.id);
    }
  }

  async execute() {
    const maxRetries = 4;
    let attempt = 0;
    const baseSettings = this.job.settings;
    while (attempt <= maxRetries) {
      attempt += 1;
      this.currentContent = '';
      this.finishReason = null;
      this.usage = null;
      try {
        await this.apiClient.chat({
          model: baseSettings.model,
          messages: this.job.history.map((item) => ({ role: item.role, content: item.content })),
          temperature: baseSettings.temperature,
          topP: baseSettings.topP,
          maxTokens: baseSettings.maxTokens,
          stream: this.job.preferStream,
        }, {
          signal: this.controller.signal,
          onChunk: (text) => {
            this.currentContent += text;
            this.ui.updateMessage(this.job.assistantMessage.id, { content: this.currentContent });
          },
          onUsage: (usage) => {
            if (usage) {
              this.usage = usage;
            }
          },
          onFinish: (info) => {
            this.finishReason = info.finishReason;
            if (info.usage) {
              this.usage = info.usage;
            }
          },
          onFirstToken: (ms) => {
            this.metrics.recordTtft(ms);
          },
          onRequestId: (id) => {
            this.requestId = id;
            if (id) {
              this.ui.updateMessage(this.job.assistantMessage.id, { requestId: id });
            }
          },
        });
        if (this.usage) {
          this.metrics.recordUsage(this.usage);
        }
        return {
          success: true,
          content: this.currentContent,
          finishReason: this.finishReason,
          usage: this.usage,
          requestId: this.requestId,
          autoAttempt: this.job.autoAttempt,
        };
      } catch (error) {
        if (error instanceof DOMException && error.name === 'AbortError') {
          return {
            success: false,
            canceled: true,
            content: this.currentContent,
            requestId: this.requestId,
            autoAttempt: this.job.autoAttempt,
          };
        }
        if (!(error instanceof ApiError)) {
          throw error;
        }
        if (error.status === 400 || error.status === 401) {
          throw error;
        }
        if (error.status === 429) {
          this.metrics.record429();
          this.metrics.recordRetry();
          if (attempt > maxRetries) {
            throw error;
          }
          const waitMs = error.retryAfterMs || Math.max(1000, Backoff.exponential(attempt));
          await this.waitForRetry(waitMs, '触发限流，等待后自动重试。');
          continue;
        }
        if (error.status >= 500 || error.status === 0) {
          this.metrics.recordRetry();
          if (attempt > maxRetries) {
            throw error;
          }
          const waitMs = Backoff.exponential(attempt);
          await this.waitForRetry(waitMs, '服务器暂时不可用，等待后自动重试。');
          continue;
        }
        throw error;
      }
    }
    throw new ApiError('超过最大重试次数。', 500);
  }

  async waitForRetry(delay, message) {
    const seconds = (delay / 1000).toFixed(1);
    this.ui.setMessageNote(this.job.assistantMessage.id, `等待 ${seconds}s 后重试`);
    await new Promise((resolve, reject) => {
      let resolved = false;
      const dismiss = this.ui.showToast(`${message}（${seconds}s）`, 'info', {
        duration: delay + 2000,
        action: {
          label: '立即重试',
          onClick: () => {
            if (!resolved) {
              resolved = true;
              cleanup();
              resolve(undefined);
            }
          },
        },
      });
      const timer = window.setTimeout(() => {
        if (!resolved) {
          resolved = true;
          cleanup();
          resolve(undefined);
        }
      }, delay);
      const onAbort = () => {
        if (!resolved) {
          resolved = true;
          cleanup();
          reject(new DOMException('Aborted', 'AbortError'));
        }
      };
      const cleanup = () => {
        window.clearTimeout(timer);
        dismiss();
        this.controller.signal.removeEventListener('abort', onAbort);
        this.ui.setMessageNote(this.job.assistantMessage.id, undefined);
      };
      this.controller.signal.addEventListener('abort', onAbort, { once: true });
    });
  }
}

/**
 * 应用控制器。
 */
class ChatApp {
  constructor(store, apiClient, metrics, ui) {
    this.store = store;
    this.apiClient = apiClient;
    this.metrics = metrics;
    this.ui = ui;
    /** @type {ChatHistoryItem[]} */
    this.chatHistory = [];
    /** @type {MessageRecord[]} */
    this.messageLog = [];
    /** @type {MessageRecord|null} */
    this.systemMessage = null;
    /** @type {ChatJob[]} */
    this.queue = [];
    /** @type {Map<string, ChatRequest>} */
    this.activeRequests = new Map();
    this.maxConcurrent = 2;
    this.textareaKeyHandler = (event) => this.handleTextareaKey(event);
  }

  initialize() {
    const settings = this.store.getSettings();
    this.ui.fillSettingsForm(settings, this.store.getApiKey());
    this.updateMetricsDisplay();
    if (settings.systemPrompt) {
      this.applySystemPrompt(settings.systemPrompt, { newSession: true });
    }
    this.showWelcomeMessage();
    this.updateQueueStatus();
    updateCspConnectSrc(settings.baseUrl);

    this.ui.sendButton.addEventListener('click', () => this.handleSend());
    this.ui.textarea.addEventListener('keydown', this.textareaKeyHandler);
    this.ui.textarea.addEventListener('input', () => this.ui.autoResizeTextarea());
    this.ui.autoResizeTextarea();

    this.ui.settingsButton.addEventListener('click', () => this.openSettings());
    this.ui.settingsCloseButton.addEventListener('click', () => this.ui.closeSettings());
    this.ui.settingsForm.addEventListener('submit', (event) => {
      event.preventDefault();
      this.handleSaveSettings();
    });
    this.ui.clearKeyButton.addEventListener('click', () => this.handleClearKey());
    this.ui.modelsRefreshButton.addEventListener('click', () => this.fetchModels());
    this.ui.connectionTestButton.addEventListener('click', () => this.handleConnectionTest());
    this.ui.clearButton.addEventListener('click', () => this.clearConversation());
    this.ui.exportButton.addEventListener('click', () => this.exportConversation());
    this.ui.metricsExportButton.addEventListener('click', () => this.exportMetrics());

    if (this.store.getApiKey()) {
      this.fetchModels({ quiet: true });
    } else {
      this.ui.setModelsStatus('尚未加载模型列表。请先在设置中填写密钥。');
    }
  }

  showWelcomeMessage() {
    const welcome = {
      id: createId('assistant'),
      role: 'assistant',
      content: '👋 欢迎使用 Gradient Serverless Chat！请先点击右上角“设置 ⚙️”填写 API Key（可选勾选 sessionStorage 保存，但存在泄露风险）。',
      createdAt: Date.now(),
      streaming: false,
      note: '提示',
    };
    this.messageLog.push(welcome);
    this.ui.addMessage(welcome);
  }

  /**
   * 应用系统提示。
   * @param {string} prompt
   * @param {{ newSession?: boolean; announce?: boolean; }} [options]
   */
  applySystemPrompt(prompt, options) {
    const opts = options || {};
    const newSession = Boolean(opts.newSession);
    const announce = Boolean(opts.announce);
    const trimmed = (prompt || '').trim();
    const previous = this.systemMessage ? this.systemMessage.content : '';

    this.chatHistory = this.chatHistory.filter((item) => item.role !== 'system');
    if (trimmed) {
      this.chatHistory.unshift({ role: 'system', content: trimmed });
    }

    if (!trimmed) {
      if (this.systemMessage) {
        this.messageLog = this.messageLog.filter((msg) => msg.id !== this.systemMessage.id);
        this.ui.removeMessage(this.systemMessage.id);
        if (announce && previous) {
          this.ui.showToast('已移除系统提示，将立即应用于后续请求。', 'info');
        }
        this.systemMessage = null;
      } else if (announce) {
        this.ui.showToast('当前未设置系统提示。', 'info');
      }
      return;
    }

    if (this.systemMessage) {
      if (this.systemMessage.content !== trimmed) {
        this.systemMessage.content = trimmed;
        this.systemMessage.createdAt = Date.now();
        this.systemMessage.note = '系统提示';
        this.ui.updateMessage(this.systemMessage.id, { content: trimmed, note: '系统提示' });
        if (announce) {
          this.ui.showToast('系统提示已更新，将在下一次提问生效。', 'success');
        }
      } else if (announce) {
        this.ui.showToast('系统提示保持不变。', 'info');
      }
      return;
    }

    this.messageLog = this.messageLog.filter((msg) => msg.role !== 'system');
    const systemRecord = {
      id: createId('system'),
      role: 'system',
      content: trimmed,
      createdAt: Date.now(),
      note: '系统提示',
    };
    this.systemMessage = systemRecord;
    this.messageLog.unshift(systemRecord);
    this.ui.addMessage(systemRecord, 'start');
    if (announce) {
      this.ui.showToast('系统提示已设置，将在下一次提问生效。', 'success');
    }
    if (!newSession) {
      this.ui.scrollToBottom();
    }
  }

  handleTextareaKey(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.handleSend();
    }
  }

  handleSend() {
    const raw = this.ui.textarea.value;
    const text = raw.trim();
    if (!text) {
      return;
    }
    if (!this.store.getApiKey()) {
      this.ui.showToast('请先在设置中填写 API Key。', 'error');
      this.openSettings();
      return;
    }
    const userMessage = {
      id: createId('user'),
      role: 'user',
      content: text,
      createdAt: Date.now(),
    };
    this.chatHistory.push({ role: 'user', content: text });
    this.messageLog.push(userMessage);
    this.ui.addMessage(userMessage);
    this.ui.resetInput();
    const settings = this.store.getSettings();
    this.enqueueAssistant({
      model: settings.model,
      temperature: settings.temperature,
      topP: settings.topP,
    }, settings.maxTokens, 0, false);
  }

  enqueueAssistant(baseSettings, maxTokens, autoAttempt, isAuto) {
    const assistantMessage = {
      id: createId('assistant'),
      role: 'assistant',
      content: '',
      createdAt: Date.now(),
      streaming: false,
      queued: true,
      isAuto: Boolean(isAuto),
    };
    this.messageLog.push(assistantMessage);
    this.ui.addMessage(assistantMessage);
    const job = {
      id: assistantMessage.id,
      assistantMessage,
      history: this.chatHistory.map((item) => ({ role: item.role, content: item.content })),
      settings: {
        model: baseSettings.model,
        temperature: baseSettings.temperature,
        topP: baseSettings.topP,
        maxTokens,
      },
      preferStream: true,
      autoAttempt,
    };
    this.queue.push(job);
    this.processQueue();
    this.updateQueueStatus();
    return assistantMessage;
  }

  processQueue() {
    while (this.activeRequests.size < this.maxConcurrent && this.queue.length) {
      const job = this.queue.shift();
      if (!job) {
        break;
      }
      this.startJob(job);
    }
    this.updateQueueStatus();
  }

  startJob(job) {
    const request = new ChatRequest({ job, apiClient: this.apiClient, metrics: this.metrics, ui: this.ui });
    this.activeRequests.set(job.id, request);
    request.start().then((result) => {
      if (result.canceled) {
        this.handleCanceled(job, result);
      } else if (result.success) {
        this.handleSuccess(job, result);
      }
    }).catch((error) => {
      this.handleFailure(job, error);
    }).finally(() => {
      this.activeRequests.delete(job.id);
      this.updateQueueStatus();
      this.processQueue();
    });
  }

  handleSuccess(job, result) {
    const note = result.usage ? undefined : 'usage 未返回';
    Object.assign(job.assistantMessage, {
      content: result.content,
      streaming: false,
      queued: false,
      finishReason: result.finishReason || undefined,
      usage: result.usage || undefined,
      requestId: result.requestId || undefined,
      error: undefined,
      note,
    });
    this.ui.removeRetryButton(job.assistantMessage.id);
    this.ui.updateMessage(job.assistantMessage.id, {
      content: result.content,
      streaming: false,
      queued: false,
      finishReason: result.finishReason || undefined,
      usage: result.usage || undefined,
      requestId: result.requestId || undefined,
      error: undefined,
      note,
    });
    this.chatHistory.push({ role: 'assistant', content: result.content });
    this.updateMetricsDisplay();
    if (result.finishReason === 'length' && job.autoAttempt < 3) {
      this.scheduleAutoContinue(job);
    }
  }

  handleCanceled(job, result) {
    job.assistantMessage.streaming = false;
    job.assistantMessage.note = '请求已取消';
    this.ui.removeRetryButton(job.assistantMessage.id);
    this.ui.updateMessage(job.assistantMessage.id, {
      streaming: false,
      note: '请求已取消',
    });
  }

  handleFailure(job, error) {
    let message = '请求失败';
    if (error instanceof ApiError) {
      message = error.message;
      if (error.requestId) {
        job.assistantMessage.requestId = error.requestId;
      }
      this.metrics.recordFailure(`status-${error.status}`);
      if (error.status === 401) {
        this.ui.showToast('密钥无效或已过期，请重新设置。', 'error');
        this.openSettings();
      } else if (error.status === 429) {
        this.ui.showToast('多次触发限流，请稍后再试。', 'error');
      } else if (error.status >= 500 || error.status === 0) {
        this.ui.showToast('服务器暂时不可用，请稍后重试。', 'error');
      }
    } else {
      this.metrics.recordFailure('unknown');
      message = error instanceof Error ? error.message : String(error);
      this.ui.showToast('未知错误，请查看控制台。', 'error');
      console.error(error);
    }
    Object.assign(job.assistantMessage, {
      streaming: false,
      queued: false,
      error: message,
      note: undefined,
    });
    this.ui.updateMessage(job.assistantMessage.id, {
      streaming: false,
      queued: false,
      error: message,
      requestId: job.assistantMessage.requestId,
      note: undefined,
    });
    this.ui.addRetryButton(job.assistantMessage.id, () => this.retryJob(job));
    this.updateMetricsDisplay();
  }

  retryJob(job) {
    this.ui.removeRetryButton(job.assistantMessage.id);
    Object.assign(job.assistantMessage, {
      content: '',
      error: undefined,
      streaming: false,
      queued: true,
      finishReason: undefined,
      usage: undefined,
      requestId: undefined,
      note: '重新发送中…',
    });
    this.ui.updateMessage(job.assistantMessage.id, {
      content: '',
      streaming: false,
      queued: true,
      error: undefined,
      finishReason: undefined,
      usage: undefined,
      requestId: undefined,
      note: '重新发送中…',
    });
    const newJob = {
      id: job.assistantMessage.id,
      assistantMessage: job.assistantMessage,
      history: this.chatHistory.map((item) => ({ role: item.role, content: item.content })),
      settings: { ...job.settings },
      preferStream: job.preferStream,
      autoAttempt: job.autoAttempt,
    };
    this.queue.unshift(newJob);
    this.processQueue();
  }

  scheduleAutoContinue(job) {
    const nextAttempt = job.autoAttempt + 1;
    if (nextAttempt > 3) {
      return;
    }
    const autoMessage = {
      id: createId('user'),
      role: 'user',
      content: 'continue',
      createdAt: Date.now(),
      isAuto: true,
      note: '自动续写',
    };
    this.messageLog.push(autoMessage);
    this.ui.addMessage(autoMessage);
    this.chatHistory.push({ role: 'user', content: 'continue' });
    const nextMax = Math.max(32, Math.floor(job.settings.maxTokens * 0.8));
    this.enqueueAssistant({
      model: job.settings.model,
      temperature: job.settings.temperature,
      topP: job.settings.topP,
    }, nextMax, nextAttempt, true);
  }

  updateQueueStatus() {
    this.ui.updateQueueStatus(this.queue.length, this.activeRequests.size);
  }

  updateMetricsDisplay() {
    this.ui.updateMetrics(this.metrics.getSnapshot());
  }

  openSettings() {
    this.ui.fillSettingsForm(this.store.getSettings(), this.store.getApiKey());
    this.ui.openSettings();
  }

  handleSaveSettings() {
    const values = this.ui.readSettingsForm();
    try {
      new URL(values.baseUrl);
    } catch (error) {
      this.ui.showToast('Base URL 格式不合法。', 'error');
      return;
    }
    if (!values.model) {
      this.ui.showToast('模型 ID 不能为空。', 'error');
      return;
    }
    this.store.saveSettings({
      baseUrl: values.baseUrl,
      model: values.model,
      temperature: values.temperature,
      topP: values.topP,
      maxTokens: values.maxTokens,
      persistKey: values.persist,
      systemPrompt: values.systemPrompt,
    });
    this.store.setApiKey(values.apiKey, values.persist);
    updateCspConnectSrc(values.baseUrl);
    this.applySystemPrompt(values.systemPrompt, { announce: true });
    this.ui.showToast('设置已保存。', 'success');
    this.ui.closeSettings();
  }

  handleClearKey() {
    this.store.clearKey();
    this.store.saveSettings({ persistKey: false });
    this.ui.apiKeyInput.value = '';
    this.ui.persistCheckbox.checked = false;
    this.ui.showToast('已清除保存在前端的 API Key。', 'info');
  }

  async fetchModels(options) {
    const quiet = options && options.quiet;
    this.ui.setModelsStatus('加载模型列表中…');
    let attempt = 0;
    const maxAttempts = 3;
    while (attempt < maxAttempts) {
      attempt += 1;
      try {
        const result = await this.apiClient.listModels();
        const models = result.models;
        this.ui.setModelOptions(models, this.store.getSettings().model);
        this.ui.setModelsStatus(`已加载 ${models.length} 个模型。`);
        if (!quiet) {
          this.ui.showToast('模型列表已刷新。', 'success');
        }
        return;
      } catch (error) {
        if (error instanceof ApiError) {
          if (error.status === 401) {
            this.ui.setModelsStatus('认证失败，无法加载模型。');
            this.ui.showToast('API Key 无效，无法加载模型。', 'error');
            this.metrics.recordFailure('models-401');
            this.updateMetricsDisplay();
            return;
          }
          if (error.status >= 500 || error.status === 429 || error.status === 0) {
            if (attempt < maxAttempts) {
              await Backoff.wait(Backoff.exponential(attempt));
              continue;
            }
          }
          this.ui.setModelsStatus(`加载失败：${error.message}`);
          this.metrics.recordFailure(`models-${error.status}`);
          this.updateMetricsDisplay();
          return;
        }
        if (attempt < maxAttempts) {
          await Backoff.wait(Backoff.exponential(attempt));
          continue;
        }
        this.ui.setModelsStatus('网络异常，无法加载模型。');
        this.metrics.recordFailure('models-network');
        this.updateMetricsDisplay();
        return;
      }
    }
    this.ui.setModelsStatus('离线模式，可手动输入模型 ID。');
  }

  async handleConnectionTest() {
    this.ui.setModelsStatus('连接测试中…');
    try {
      const result = await this.apiClient.listModels();
      this.ui.setModelOptions(result.models, this.store.getSettings().model);
      this.ui.setModelsStatus(`连接成功，获取到 ${result.models.length} 个模型。`);
      this.ui.showToast('连接测试通过。', 'success');
    } catch (error) {
      const message = error instanceof ApiError ? error.message : '网络错误';
      this.ui.setModelsStatus(`连接测试失败：${message}`);
      this.ui.showToast('连接测试失败。', 'error');
    }
  }

  clearConversation() {
    this.queue = [];
    for (const request of this.activeRequests.values()) {
      request.cancel();
    }
    this.chatHistory = [];
    this.messageLog = [];
    this.systemMessage = null;
    this.ui.clearChat();
    const prompt = this.store.getSettings().systemPrompt;
    if (prompt) {
      this.applySystemPrompt(prompt, { newSession: true });
    }
    this.showWelcomeMessage();
    this.updateQueueStatus();
  }

  exportConversation() {
    const snapshot = {
      exported_at: new Date().toISOString(),
      base_url: this.store.getSettings().baseUrl,
      model: this.store.getSettings().model,
      messages: this.messageLog.map((msg) => ({
        id: msg.id,
        role: msg.role,
        content: msg.content,
        created_at: new Date(msg.createdAt).toISOString(),
        finish_reason: msg.finishReason || null,
        usage: msg.usage || null,
        request_id: msg.requestId || null,
        note: msg.note || null,
        error: msg.error || null,
        auto: Boolean(msg.isAuto),
      })),
    };
    this.ui.downloadText(`gradient-chat-${Date.now()}.json`, JSON.stringify(snapshot, null, 2), 'application/json');
  }

  exportMetrics() {
    this.ui.downloadText(`gradient-metrics-${Date.now()}.ndjson`, this.metrics.exportNdjson(), 'application/x-ndjson');
  }
}

/**
 * 更新 CSP connect-src。
 * @param {string} baseUrl
 */
function updateCspConnectSrc(baseUrl) {
  const meta = document.querySelector('meta[http-equiv="Content-Security-Policy"]');
  if (!meta) {
    return;
  }
  let origin = '';
  try {
    origin = new URL(baseUrl).origin;
  } catch (error) {
    console.warn('无法解析 Base URL：', error);
  }
  const defaultOrigins = new Set(["'self'", 'https://inference.do-ai.run']);
  if (origin) {
    defaultOrigins.add(origin);
  }
  const content = meta.getAttribute('content') || '';
  const directives = content
    .split(';')
    .map((segment) => segment.trim())
    .filter((segment) => segment.length > 0);
  const connectIndex = directives.findIndex((directive) => directive.startsWith('connect-src'));
  const values = Array.from(defaultOrigins);
  if (connectIndex >= 0) {
    const parts = directives[connectIndex].split(/\s+/).slice(1);
    parts.forEach((part) => values.push(part));
    const merged = Array.from(new Set(values));
    directives[connectIndex] = `connect-src ${merged.join(' ')}`;
  } else {
    directives.push(`connect-src ${values.join(' ')}`);
  }
  meta.setAttribute('content', directives.join('; '));
}

/**
 * 运行最小化单元测试。
 */
function runSelfTests() {
  const results = [];
  const assertEqual = (name, actual, expected) => {
    if (actual !== expected) {
      throw new Error(`${name} 失败：${actual} !== ${expected}`);
    }
  };
  try {
    assertEqual('Backoff#1', Backoff.exponential(1), 1000);
    assertEqual('Backoff#5', Backoff.exponential(5), 8000);
    const metrics = new Metrics();
    metrics.recordUsage({ prompt_tokens: 10, completion_tokens: 20, total_tokens: 35 });
    const snapshot = metrics.getSnapshot();
    assertEqual('Metrics prompt', snapshot.prompt, 10);
    assertEqual('Metrics completion', snapshot.completion, 20);
    metrics.recordTtft(100);
    metrics.recordTtft(200);
    const p95 = metrics.getSnapshot().ttftP95;
    if (p95 === null) {
      throw new Error('TTFT p95 未计算');
    }
    results.push('自检通过');
    console.info('自检通过');
  } catch (error) {
    console.error('自检失败', error);
  }
}

function init() {
  const store = new Store();
  const metrics = new Metrics();
  const ui = new UI();
  const apiClient = new ApiClient(store, metrics);
  const app = new ChatApp(store, apiClient, metrics, ui);
  app.initialize();
}

runSelfTests();

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}

