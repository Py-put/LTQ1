'use strict';

/**
 * @typedef {Object} UsageMetrics
 * @property {number} [prompt_tokens]
 * @property {number} [completion_tokens]
 * @property {number} [total_tokens]
 */

/**
 * @typedef {Object} SettingsState
 * @property {string} baseUrl
 * @property {string} model
 * @property {number} temperature
 * @property {number} top_p
 * @property {number} max_tokens
 * @property {boolean} rememberKey
 */

/**
 * @typedef {Object} ChatMessage
 * @property {string} id
 * @property {'system'|'user'|'assistant'} role
 * @property {string} content
 * @property {'queued'|'pending'|'streaming'|'done'|'error'|'cancelled'} status
 * @property {string=} finishReason
 * @property {UsageMetrics=} usage
 * @property {string=} requestId
 * @property {string=} error
 * @property {number} createdAt
 * @property {number=} attempt
 */

/**
 * @typedef {Object} ChatRequestOptions
 * @property {AbortSignal} signal
 * @property {(delta: string) => void} onDelta
 * @property {(info: { requestId: string | null }) => void=} onStart
 * @property {(usage: UsageMetrics) => void=} onUsage
 * @property {(summary: { finishReason: string | null; requestId: string | null }) => void=} onFinish
 * @property {(retry: RetryInfo) => void=} onRetry
 */

/**
 * @typedef {Object} RetryInfo
 * @property {number} attempt
 * @property {number} delay
 * @property {number} status
 * @property {string} reason
 * @property {() => void} resume
 */

/**
 * @typedef {Object} ChatCompletionResponse
 * @property {string} text
 * @property {string | null} finishReason
 * @property {UsageMetrics | null} usage
 * @property {string | null} requestId
 */

/**
 * @typedef {{ ts: number, value: number }} TimedValue
 */

const DEFAULT_SETTINGS = /** @type {SettingsState} */ ({
  baseUrl: 'https://inference.do-ai.run',
  model: 'llama3.3-70b-instruct',
  temperature: 0.2,
  top_p: 0.95,
  max_tokens: 512,
  rememberKey: false,
});

const FIVE_MINUTES = 5 * 60 * 1000;
const MAX_CONCURRENT = 2;
const AUTO_CONTINUE_LIMIT = 3;

/**
 * 简易 UUID 生成，优先使用 crypto.randomUUID。
 * @returns {string}
 */
function createId() {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID();
  }
  return 'id-' + Math.random().toString(16).slice(2) + Date.now().toString(16);
}

/**
 * 简易延迟，支持取消。
 * @param {number} ms
 * @param {AbortSignal} signal
 * @returns {Promise<void>}
 */
function delay(ms, signal) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      cleanup();
      resolve();
    }, ms);
    const cleanup = () => {
      clearTimeout(timer);
      signal.removeEventListener('abort', onAbort);
    };
    const onAbort = () => {
      cleanup();
      reject(new DOMException('Aborted', 'AbortError'));
    };
    if (signal.aborted) {
      cleanup();
      reject(new DOMException('Aborted', 'AbortError'));
      return;
    }
    signal.addEventListener('abort', onAbort);
  });
}

/**
 * Backoff 辅助类，提供指数退避时间。
 */
class Backoff {
  /**
   * @param {number[]} scheduleMs
   */
  constructor(scheduleMs) {
    this.schedule = scheduleMs;
  }

  /**
   * @param {number} attempt
   * @returns {number | null}
   */
  next(attempt) {
    if (attempt < 0 || attempt >= this.schedule.length) {
      return null;
    }
    return this.schedule[attempt];
  }
}

/**
 * 监控指标收集。
 */
class Metrics {
  constructor() {
    this.promptTokens = 0;
    this.completionTokens = 0;
    this.totalTokens = 0;
    /** @type {TimedValue[]} */
    this.ttft = [];
    /** @type {number[]} */
    this.retries = [];
    /** @type {number[]} */
    this.rateLimitTimestamps = [];
    /** @type {Map<string, number>} */
    this.failures = new Map();
    /** @type {Array<{ts:number,type:string,data:unknown}>} */
    this.events = [];
  }

  /**
   * @param {UsageMetrics | null | undefined} usage
   */
  recordUsage(usage) {
    if (!usage) {
      return;
    }
    if (typeof usage.prompt_tokens === 'number') {
      this.promptTokens += usage.prompt_tokens;
    }
    if (typeof usage.completion_tokens === 'number') {
      this.completionTokens += usage.completion_tokens;
    }
    if (typeof usage.total_tokens === 'number') {
      this.totalTokens += usage.total_tokens;
    }
    this.events.push({ ts: Date.now(), type: 'usage', data: usage });
  }

  /**
   * @param {number} value
   */
  recordTTFT(value) {
    const entry = { ts: Date.now(), value };
    this.ttft.push(entry);
    if (this.ttft.length > 20) {
      this.ttft.shift();
    }
    this.events.push({ ts: Date.now(), type: 'ttft', data: value });
  }

  /**
   * @returns {number[]}
   */
  getTTFTList() {
    return this.ttft.map((item) => Math.round(item.value));
  }

  /**
   * @returns {number | null}
   */
  getTTFTP95() {
    if (this.ttft.length === 0) {
      return null;
    }
    const values = this.ttft.map((item) => item.value).sort((a, b) => a - b);
    const index = Math.ceil(values.length * 0.95) - 1;
    return Math.round(values[Math.max(0, Math.min(values.length - 1, index))]);
  }

  /**
   * @param {number} status
   */
  recordRetry(status) {
    const now = Date.now();
    this.retries.push(now);
    this.events.push({ ts: now, type: 'retry', data: { status } });
    this.trimRecent();
  }

  trimRecent() {
    const cutoff = Date.now() - FIVE_MINUTES;
    this.retries = this.retries.filter((ts) => ts >= cutoff);
    this.rateLimitTimestamps = this.rateLimitTimestamps.filter((ts) => ts >= cutoff);
  }

  record429() {
    const now = Date.now();
    this.rateLimitTimestamps.push(now);
    this.events.push({ ts: now, type: 'rate_limit' });
    this.trimRecent();
  }

  /**
   * @param {string} kind
   */
  recordFailure(kind) {
    const prev = this.failures.get(kind) || 0;
    this.failures.set(kind, prev + 1);
    this.events.push({ ts: Date.now(), type: 'failure', data: { kind } });
  }

  /**
   * @returns {{prompt:number, completion:number, total:number}}
   */
  getTokenTotals() {
    return {
      prompt: this.promptTokens,
      completion: this.completionTokens,
      total: this.totalTokens,
    };
  }

  /**
   * @returns {number}
   */
  getRecent429() {
    this.trimRecent();
    return this.rateLimitTimestamps.length;
  }

  /**
   * @returns {number}
   */
  getRecentRetries() {
    this.trimRecent();
    return this.retries.length;
  }

  /**
   * @returns {Array<{kind: string; count: number}>}
   */
  getFailureList() {
    return Array.from(this.failures.entries()).map(([kind, count]) => ({ kind, count }));
  }

  /**
   * @returns {string}
   */
  exportNDJSON() {
    return this.events.map((event) => JSON.stringify(event)).join('\n');
  }
}

/**
 * 设置存储，使用 localStorage + sessionStorage。
 */
class Store extends EventTarget {
  constructor() {
    super();
    this.state = { ...DEFAULT_SETTINGS };
    this.apiKey = '';
    this.loadFromStorage();
  }

  loadFromStorage() {
    try {
      const raw = localStorage.getItem('gradient-settings');
      if (raw) {
        const parsed = JSON.parse(raw);
        this.state = { ...this.state, ...parsed };
      }
    } catch (error) {
      console.warn('无法读取设置：', error);
    }
    try {
      const remember = sessionStorage.getItem('gradient-remember-key');
      if (remember === '1') {
        const key = sessionStorage.getItem('gradient-api-key');
        if (key) {
          this.apiKey = key;
          this.state.rememberKey = true;
        }
      }
    } catch (error) {
      console.warn('无法读取密钥：', error);
    }
  }

  persist() {
    try {
      const { rememberKey, ...rest } = this.state;
      localStorage.setItem('gradient-settings', JSON.stringify(rest));
    } catch (error) {
      console.warn('保存设置失败：', error);
    }
    try {
      if (this.state.rememberKey && this.apiKey) {
        sessionStorage.setItem('gradient-remember-key', '1');
        sessionStorage.setItem('gradient-api-key', this.apiKey);
      } else {
        sessionStorage.removeItem('gradient-remember-key');
        sessionStorage.removeItem('gradient-api-key');
      }
    } catch (error) {
      console.warn('保存密钥失败：', error);
    }
  }

  /**
   * @returns {SettingsState}
   */
  getSettings() {
    return { ...this.state };
  }

  /**
   * @param {Partial<SettingsState>} patch
   */
  updateSettings(patch) {
    this.state = { ...this.state, ...patch };
    this.persist();
    this.dispatchEvent(new Event('change'));
  }

  /**
   * @returns {string}
   */
  getApiKey() {
    return this.apiKey;
  }

  /**
   * @param {string} value
   * @param {boolean} remember
   */
  setApiKey(value, remember) {
    this.apiKey = value;
    this.state.rememberKey = remember;
    this.persist();
    this.dispatchEvent(new Event('key-change'));
  }

  clearApiKey() {
    this.apiKey = '';
    this.state.rememberKey = false;
    this.persist();
    this.dispatchEvent(new Event('key-change'));
  }
}

/**
 * Toast 管理。
 */
class Toasts {
  /**
   * @param {HTMLElement} container
   */
  constructor(container) {
    this.container = container;
  }

  /**
   * @param {{ title: string; message?: string; kind?: 'info'|'error'|'success'; duration?: number; actions?: Array<{ label: string; action: () => void }> }} options
   */
  show(options) {
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.dataset.kind = options.kind || 'info';
    const title = document.createElement('strong');
    title.textContent = options.title;
    toast.appendChild(title);
    if (options.message) {
      const p = document.createElement('span');
      p.textContent = options.message;
      toast.appendChild(p);
    }
    if (options.actions && options.actions.length > 0) {
      const actionWrap = document.createElement('div');
      actionWrap.className = 'actions';
      options.actions.forEach((action) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.textContent = action.label;
        btn.addEventListener('click', () => {
          try {
            action.action();
          } finally {
            this.dismiss(toast);
          }
        });
        actionWrap.appendChild(btn);
      });
      toast.appendChild(actionWrap);
    }
    this.container.appendChild(toast);
    const timeout = options.duration ?? 6000;
    if (timeout > 0) {
      setTimeout(() => this.dismiss(toast), timeout);
    }
  }

  /**
   * @param {HTMLElement} toast
   */
  dismiss(toast) {
    if (toast.isConnected) {
      toast.remove();
    }
  }
}

/**
 * SSE 解析器。
 */
class SseParser {
  /**
   * @param {(data: string) => void} onMessage
   */
  constructor(onMessage) {
    this.buffer = '';
    this.onMessage = onMessage;
    /** @type {string[]} */
    this.current = [];
  }

  /**
   * @param {string} chunk
   */
  feed(chunk) {
    this.buffer += chunk;
    let idx;
    while ((idx = this.buffer.indexOf('\n')) !== -1) {
      const line = this.buffer.slice(0, idx);
      this.buffer = this.buffer.slice(idx + 1);
      this.processLine(line.replace(/\r$/, ''));
    }
  }

  /**
   * @param {string} line
   */
  processLine(line) {
    if (line.startsWith('data:')) {
      const value = line.slice(5).trimStart();
      this.current.push(value);
      return;
    }
    if (line === '') {
      const data = this.current.join('\n');
      this.current = [];
      if (data) {
        this.onMessage(data);
      }
    }
  }

  close() {
    if (this.current.length > 0) {
      const data = this.current.join('\n');
      this.current = [];
      if (data) {
        this.onMessage(data);
      }
    }
  }
}

/**
 * 简单任务队列，限制并发。
 */
class RequestQueue {
  /**
   * @param {number} limit
   */
  constructor(limit) {
    this.limit = limit;
    this.active = 0;
    /** @type {Array<QueueEntry>} */
    this.queue = [];
  }

  /**
   * @param {() => Promise<void>} task
   * @returns {QueueHandle}
   */
  enqueue(task) {
    /** @type {QueueEntry} */
    const entry = {
      task,
      started: false,
      resolve: () => {},
      reject: () => {},
    };
    const promise = new Promise((resolve, reject) => {
      entry.resolve = resolve;
      entry.reject = reject;
    });
    entry.promise = promise;
    this.queue.push(entry);
    this.process();
    return {
      promise,
      cancel: () => {
        if (entry.started) {
          return false;
        }
        const index = this.queue.indexOf(entry);
        if (index !== -1) {
          this.queue.splice(index, 1);
          entry.reject(new Error('cancelled'));
          return true;
        }
        return false;
      },
    };
  }

  process() {
    while (this.active < this.limit) {
      const entry = this.queue.find((item) => !item.started);
      if (!entry) {
        break;
      }
      entry.started = true;
      this.active += 1;
      Promise.resolve()
        .then(() => entry.task())
        .then(() => entry.resolve())
        .catch((error) => entry.reject(error))
        .finally(() => {
          this.active -= 1;
          const idx = this.queue.indexOf(entry);
          if (idx !== -1) {
            this.queue.splice(idx, 1);
          }
          this.process();
        });
    }
  }

  /**
   * @returns {number}
   */
  getPendingCount() {
    return this.queue.filter((item) => !item.started).length;
  }
}

/**
 * @typedef {Object} QueueEntry
 * @property {() => Promise<void>} task
 * @property {boolean} started
 * @property {(value?: unknown) => void} resolve
 * @property {(reason?: unknown) => void} reject
 * @property {Promise<void>=} promise
 */

/**
 * @typedef {Object} QueueHandle
 * @property {Promise<void>} promise
 * @property {() => boolean} cancel
 */

/**
 * API 错误。
 */
class ApiError extends Error {
  /**
   * @param {string} message
   * @param {{ status: number; retry?: boolean; retryAfter?: number; type?: string; body?: unknown }} info
   */
  constructor(message, info) {
    super(message);
    this.name = 'ApiError';
    this.status = info.status;
    this.retry = Boolean(info.retry);
    this.retryAfter = info.retryAfter;
    this.type = info.type || 'unknown';
    this.body = info.body;
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
    this.backoff = new Backoff([1000, 2000, 4000, 8000]);
  }

  /**
   * @param {string} path
   */
  url(path) {
    const { baseUrl } = this.store.getSettings();
    return baseUrl.replace(/\/$/, '') + path;
  }

  /**
   * @returns {HeadersInit}
   */
  headers() {
    const headers = {
      'Content-Type': 'application/json',
    };
    const key = this.store.getApiKey();
    if (key) {
      headers['Authorization'] = `Bearer ${key}`;
    }
    return headers;
  }

  /**
   * @param {AbortSignal=} signal
   * @returns {Promise<string[]>}
   */
  async listModels(signal) {
    const response = await fetch(this.url('/v1/models'), {
      method: 'GET',
      headers: this.headers(),
      signal,
    });
    if (!response.ok) {
      throw new ApiError('获取模型失败', { status: response.status, retry: false });
    }
    const payload = await response.json();
    const ids = Array.isArray(payload.data)
      ? payload.data.map((item) => item.id).filter((id) => typeof id === 'string')
      : [];
    return ids;
  }

  /**
   * @param {object} payload
   * @param {ChatRequestOptions} options
   * @returns {Promise<ChatCompletionResponse>}
   */
  async sendChat(payload, options) {
    const signal = options.signal;
    let attempt = 0;
    while (true) {
      if (signal.aborted) {
        throw new DOMException('Aborted', 'AbortError');
      }
      const attemptController = new AbortController();
      const onAbort = () => attemptController.abort();
      signal.addEventListener('abort', onAbort);
      try {
        const result = await this.performAttempt(payload, attemptController.signal, options, attempt + 1);
        signal.removeEventListener('abort', onAbort);
        return result;
      } catch (error) {
        signal.removeEventListener('abort', onAbort);
        if (error instanceof DOMException && error.name === 'AbortError') {
          throw error;
        }
        if (!(error instanceof ApiError)) {
          throw error;
        }
        if (error.status === 401 || error.status === 400) {
          throw error;
        }
        if (error.status === 429) {
          this.metrics.record429();
        }
        if (!error.retry) {
          throw error;
        }
        const delayMs = error.retryAfter ? error.retryAfter * 1000 : this.backoff.next(attempt);
        if (delayMs == null) {
          throw error;
        }
        attempt += 1;
        this.metrics.recordRetry(error.status);
        let resume = () => {};
        const resumePromise = new Promise((resolve) => {
          resume = resolve;
        });
        options.onRetry?.({
          attempt,
          delay: delayMs,
          status: error.status,
          reason: error.type || 'retryable',
          resume,
        });
        try {
          await Promise.race([delay(delayMs, signal), resumePromise]);
        } catch (err) {
          if (err instanceof DOMException && err.name === 'AbortError') {
            throw err;
          }
          throw err;
        }
      }
    }
  }

  /**
   * @param {object} payload
   * @param {AbortSignal} signal
   * @param {ChatRequestOptions} options
   * @param {number} attempt
   * @returns {Promise<ChatCompletionResponse>}
   */
  async performAttempt(payload, signal, options, attempt) {
    const response = await fetch(this.url('/v1/chat/completions'), {
      method: 'POST',
      headers: this.headers(),
      body: JSON.stringify({ ...payload, stream: true }),
      signal,
    });
    const requestId = response.headers.get('x-request-id') || response.headers.get('X-Request-Id') || null;
    options.onStart?.({ requestId });
    if (response.status === 401) {
      throw new ApiError('密钥无效或过期', { status: 401, retry: false, type: 'unauthorized' });
    }
    if (response.status === 400) {
      const body = await this.safeJson(response);
      throw new ApiError(body?.error?.message || '请求无效', { status: 400, retry: false, type: 'bad_request', body });
    }
    if (response.status === 429) {
      const retryAfterHeader = response.headers.get('retry-after');
      const retryAfter = retryAfterHeader ? Number(retryAfterHeader) : undefined;
      const body = await this.safeJson(response);
      throw new ApiError(body?.error?.message || '达到速率限制', {
        status: 429,
        retry: true,
        retryAfter: Number.isFinite(retryAfter) ? Number(retryAfter) : undefined,
        type: 'rate_limit',
        body,
      });
    }
    if (response.status >= 500) {
      const body = await this.safeJson(response);
      throw new ApiError('服务器错误', {
        status: response.status,
        retry: true,
        type: 'server_error',
        body,
      });
    }
    if (!response.ok) {
      const body = await this.safeJson(response);
      throw new ApiError('请求失败', { status: response.status, retry: false, body });
    }
    if (!response.body || typeof response.body.getReader !== 'function' || typeof ReadableStream === 'undefined') {
      return this.nonStream(payload, signal, options, requestId);
    }
    return this.consumeStream(response, signal, options, requestId, attempt);
  }

  async safeJson(response) {
    try {
      return await response.clone().json();
    } catch (error) {
      return null;
    }
  }

  /**
   * @param {Response} response
   * @param {AbortSignal} signal
   * @param {ChatRequestOptions} options
   * @param {string | null} requestId
   * @param {number} attempt
   * @returns {Promise<ChatCompletionResponse>}
   */
  async consumeStream(response, signal, options, requestId, attempt) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let text = '';
    let finishReason = null;
    /** @type {UsageMetrics | null} */
    let usage = null;
    let recordedTTFT = false;
    const startedAt = performance.now();
    const parser = new SseParser((data) => {
      if (data === '[DONE]') {
        return;
      }
      try {
        const json = JSON.parse(data);
        const choice = Array.isArray(json.choices) ? json.choices[0] : undefined;
        if (choice && choice.delta && typeof choice.delta.content === 'string') {
          if (!recordedTTFT) {
            recordedTTFT = true;
            const ttft = performance.now() - startedAt;
            this.metrics.recordTTFT(ttft);
          }
          text += choice.delta.content;
          options.onDelta(choice.delta.content);
        }
        if (choice && choice.finish_reason) {
          finishReason = choice.finish_reason;
        }
        if (json.usage) {
          usage = json.usage;
        }
      } catch (error) {
        console.warn('解析流失败', error, data);
      }
    });
    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }
      if (signal.aborted) {
        reader.cancel();
        throw new DOMException('Aborted', 'AbortError');
      }
      parser.feed(decoder.decode(value, { stream: true }));
    }
    parser.close();
    options.onUsage?.(usage || {});
    options.onFinish?.({ finishReason, requestId });
    this.metrics.recordUsage(usage);
    if (!recordedTTFT) {
      const ttft = performance.now() - startedAt;
      this.metrics.recordTTFT(ttft);
    }
    return { text, finishReason, usage, requestId };
  }

  /**
   * @param {object} payload
   * @param {AbortSignal} signal
   * @param {ChatRequestOptions} options
   * @param {string | null} requestId
   * @returns {Promise<ChatCompletionResponse>}
   */
  async nonStream(payload, signal, options, requestId) {
    const response = await fetch(this.url('/v1/chat/completions'), {
      method: 'POST',
      headers: this.headers(),
      body: JSON.stringify({ ...payload, stream: false }),
      signal,
    });
    if (!response.ok) {
      const body = await this.safeJson(response);
      throw new ApiError('请求失败', { status: response.status, retry: false, body });
    }
    const json = await response.json();
    const choice = Array.isArray(json.choices) ? json.choices[0] : undefined;
    const content = choice && choice.message && typeof choice.message.content === 'string' ? choice.message.content : '';
    const usage = json.usage || null;
    const finishReason = choice && choice.finish_reason ? choice.finish_reason : null;
    options.onDelta(content);
    options.onUsage?.(usage || {});
    options.onFinish?.({ finishReason, requestId });
    this.metrics.recordUsage(usage);
    return { text: content, finishReason, usage, requestId };
  }
}

/**
 * 聊天界面主控制器。
 */
class ChatApp {
  /**
   * @param {Store} store
   * @param {ApiClient} api
   * @param {Metrics} metrics
   * @param {Toasts} toasts
   */
  constructor(store, api, metrics, toasts) {
    this.store = store;
    this.api = api;
    this.metrics = metrics;
    this.toasts = toasts;
    /** @type {ChatMessage[]} */
    this.messages = [];
    this.queue = new RequestQueue(MAX_CONCURRENT);
    this.activeControllers = new Map();
    this.continueMap = new Map();

    this.elements = {
      messageList: document.getElementById('message-list'),
      textarea: /** @type {HTMLTextAreaElement} */ (document.getElementById('input-text')),
      composer: /** @type {HTMLFormElement} */ (document.getElementById('composer')),
      sendBtn: /** @type {HTMLButtonElement} */ (document.getElementById('send-btn')),
      cancelBtn: /** @type {HTMLButtonElement} */ (document.getElementById('cancel-btn')),
      clearBtn: /** @type {HTMLButtonElement} */ (document.getElementById('clear-btn')),
      exportBtn: /** @type {HTMLButtonElement} */ (document.getElementById('export-btn')),
      exportMetricsBtn: /** @type {HTMLButtonElement} */ (document.getElementById('export-metrics-btn')),
      queueIndicator: document.getElementById('queue-indicator'),
      metricsToggle: /** @type {HTMLButtonElement} */ (document.getElementById('metrics-toggle')),
      metricsPanel: /** @type {HTMLElement} */ (document.getElementById('metrics-panel')),
      metricPrompt: document.getElementById('metric-prompt'),
      metricCompletion: document.getElementById('metric-completion'),
      metricTotal: document.getElementById('metric-total'),
      metricTTFTList: document.getElementById('metric-ttft-list'),
      metricTTFTP95: document.getElementById('metric-ttft-p95'),
      metric429: document.getElementById('metric-429'),
      metricRetries: document.getElementById('metric-retries'),
      metricFailures: document.getElementById('metric-failures'),
      settingsBtn: /** @type {HTMLButtonElement} */ (document.getElementById('settings-btn')),
      settingsModal: /** @type {HTMLElement} */ (document.getElementById('settings-modal')),
      settingsClose: /** @type {HTMLButtonElement} */ (document.getElementById('settings-close')),
      settingsForm: /** @type {HTMLFormElement} */ (document.getElementById('settings-form')),
      baseUrlInput: /** @type {HTMLInputElement} */ (document.getElementById('setting-base-url')),
      apiKeyInput: /** @type {HTMLInputElement} */ (document.getElementById('setting-api-key')),
      rememberKeyCheckbox: /** @type {HTMLInputElement} */ (document.getElementById('setting-remember-key')),
      modelSelect: /** @type {HTMLSelectElement} */ (document.getElementById('setting-model')),
      refreshModelsBtn: /** @type {HTMLButtonElement} */ (document.getElementById('refresh-models')),
      testConnectionBtn: /** @type {HTMLButtonElement} */ (document.getElementById('test-connection')),
      settingsClearKey: /** @type {HTMLButtonElement} */ (document.getElementById('settings-clear-key')),
      temperatureInput: /** @type {HTMLInputElement} */ (document.getElementById('setting-temperature')),
      topPInput: /** @type {HTMLInputElement} */ (document.getElementById('setting-top-p')),
      maxTokensInput: /** @type {HTMLInputElement} */ (document.getElementById('setting-max-tokens')),
    };
  }

  init() {
    this.bindEvents();
    this.renderWelcome();
    this.autosizeTextarea();
    this.updateMetricsToggleLabel();
    this.syncSettingsToForm();
    this.refreshMetrics();
    this.refreshQueueIndicator();
  }

  renderWelcome() {
    this.addMessage({
      id: createId(),
      role: 'assistant',
      content: '欢迎使用 Gradient Serverless Chat！请点击右上角“设置 ⚙️”填写 API Key 后开始对话。',
      status: 'done',
      createdAt: Date.now(),
    });
  }

  bindEvents() {
    this.elements.composer.addEventListener('submit', (event) => {
      event.preventDefault();
      this.handleSubmit();
    });
    this.elements.textarea.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        this.handleSubmit();
      }
    });
    this.elements.textarea.addEventListener('input', () => {
      this.autosizeTextarea();
    });
    this.elements.cancelBtn.addEventListener('click', () => {
      this.cancelActive();
    });
    this.elements.clearBtn.addEventListener('click', () => {
      this.clearMessages();
    });
    this.elements.exportBtn.addEventListener('click', () => {
      this.exportConversation();
    });
    this.elements.exportMetricsBtn.addEventListener('click', () => {
      this.exportMetrics();
    });
    this.store.addEventListener('change', () => this.syncSettingsToForm());
    this.store.addEventListener('key-change', () => this.syncSettingsToForm());
    this.elements.metricsToggle.addEventListener('click', () => {
      if (!this.elements.metricsPanel) return;
      const collapsed = this.elements.metricsPanel.classList.toggle('collapsed');
      this.elements.metricsToggle.setAttribute('aria-expanded', String(!collapsed));
      this.updateMetricsToggleLabel();
    });
    this.elements.settingsBtn.addEventListener('click', () => this.openSettings());
    this.elements.settingsClose.addEventListener('click', () => this.closeSettings());
    this.elements.settingsModal.addEventListener('click', (event) => {
      if (event.target === this.elements.settingsModal) {
        this.closeSettings();
      }
    });
    this.elements.settingsForm.addEventListener('submit', (event) => {
      event.preventDefault();
      this.saveSettings();
    });
    this.elements.refreshModelsBtn.addEventListener('click', () => {
      this.fetchModels();
    });
    this.elements.testConnectionBtn.addEventListener('click', () => {
      this.testConnection();
    });
    this.elements.settingsClearKey?.addEventListener('click', () => {
      this.store.clearApiKey();
      if (this.elements.apiKeyInput) this.elements.apiKeyInput.value = '';
      this.toasts.show({ title: '密钥已清除', kind: 'info' });
    });
  }

  /**
   * @param {ChatMessage} message
   */
  addMessage(message) {
    this.messages.push(message);
    const el = this.renderMessage(message);
    this.elements.messageList?.appendChild(el);
    el.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }

  clearMessages() {
    this.messages = [];
    if (this.elements.messageList) {
      this.elements.messageList.innerHTML = '';
    }
    this.renderWelcome();
  }

  exportConversation() {
    const data = this.messages.map(({ id, role, content, finishReason, usage, createdAt, requestId }) => ({
      id,
      role,
      content,
      finish_reason: finishReason || null,
      usage: usage || null,
      created_at: createdAt,
      request_id: requestId || null,
    }));
    const blob = new Blob([JSON.stringify({
      exported_at: new Date().toISOString(),
      items: data,
    }, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `gradient-chat-${Date.now()}.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  exportMetrics() {
    const blob = new Blob([this.metrics.exportNDJSON()], { type: 'application/x-ndjson' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `gradient-metrics-${Date.now()}.ndjson`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  cancelActive() {
    for (const controller of this.activeControllers.values()) {
      controller.abort();
    }
    this.toasts.show({ title: '已请求取消当前任务', kind: 'info', duration: 3000 });
  }

  renderMessage(message) {
    const item = document.createElement('div');
    item.className = `message ${message.role}`;
    item.dataset.messageId = message.id;
    item.setAttribute('role', 'listitem');
    if (message.status === 'queued' || message.status === 'pending' || message.status === 'streaming') {
      item.classList.add('is-streaming');
    }
    if (message.status === 'error') {
      item.classList.add('has-error');
    }
    if (message.status === 'cancelled') {
      item.classList.add('is-cancelled');
    }
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = message.role === 'assistant' ? 'AI' : '我';
    const bubble = document.createElement('div');
    bubble.className = 'bubble';

    const contentEl = document.createElement('div');
    contentEl.className = 'content';
    contentEl.textContent = message.content;
    bubble.appendChild(contentEl);

    if (message.role === 'assistant') {
      const actions = document.createElement('div');
      actions.className = 'actions';
      const retryBtn = document.createElement('button');
      retryBtn.type = 'button';
      retryBtn.textContent = '立即重试';
      retryBtn.addEventListener('click', () => this.retryMessage(message.id));
      actions.appendChild(retryBtn);
      const copyBtn = document.createElement('button');
      copyBtn.type = 'button';
      copyBtn.textContent = '复制';
      copyBtn.addEventListener('click', async () => {
        try {
          await navigator.clipboard.writeText(message.content);
          this.toasts.show({ title: '已复制到剪贴板', kind: 'success', duration: 2000 });
        } catch (error) {
          this.toasts.show({ title: '复制失败', kind: 'error' });
        }
      });
      actions.appendChild(copyBtn);
      bubble.appendChild(actions);
    }
    const meta = document.createElement('div');
    meta.className = 'meta';
    this.updateMeta(meta, message);
    bubble.appendChild(meta);
    item.appendChild(avatar);
    item.appendChild(bubble);
    return item;
  }

  updateMeta(metaEl, message) {
    const parts = [];
    parts.push(new Date(message.createdAt).toLocaleTimeString());
    if (message.status === 'queued') {
      parts.push('排队中');
    }
    if (message.status === 'pending' || message.status === 'streaming') {
      parts.push('生成中');
    }
    if (message.status === 'cancelled') {
      parts.push('已取消');
    }
    if (message.finishReason) {
      parts.push(`finish_reason: ${message.finishReason}`);
    }
    if (message.error) {
      parts.push(`错误：${message.error}`);
    }
    if (message.usage) {
      const usageParts = [];
      if (typeof message.usage.prompt_tokens === 'number') usageParts.push(`prompt ${message.usage.prompt_tokens}`);
      if (typeof message.usage.completion_tokens === 'number') usageParts.push(`completion ${message.usage.completion_tokens}`);
      if (typeof message.usage.total_tokens === 'number') usageParts.push(`total ${message.usage.total_tokens}`);
      if (usageParts.length) {
        parts.push('usage ' + usageParts.join('/'));
      }
    }
    if (message.requestId) {
      parts.push(`request_id: ${message.requestId}`);
    }
    metaEl.textContent = parts.join(' · ');
  }

  findMessageElement(id) {
    return this.elements.messageList?.querySelector(`[data-message-id="${id}"]`);
  }

  autosizeTextarea() {
    const textarea = this.elements.textarea;
    if (!textarea) return;
    textarea.style.height = 'auto';
    const minHeight = 96;
    const maxHeight = Math.max(window.innerHeight * 0.5, minHeight);
    const next = Math.min(Math.max(textarea.scrollHeight, minHeight), maxHeight);
    textarea.style.height = `${next}px`;
  }

  handleSubmit() {
    const textarea = this.elements.textarea;
    if (!textarea) return;
    const content = textarea.value.trim();
    if (!content) {
      return;
    }
    textarea.value = '';
    this.autosizeTextarea();
    textarea.focus();
    const userMessage = {
      id: createId(),
      role: 'user',
      content,
      status: 'done',
      createdAt: Date.now(),
    };
    this.addMessage(userMessage);
    this.scheduleAssistantReply();
  }

  scheduleAssistantReply(maxTokensOverride) {
    const assistantId = createId();
    const assistantMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      status: 'queued',
      createdAt: Date.now(),
    };
    this.addMessage(assistantMessage);
    const controller = new AbortController();
    this.activeControllers.set(assistantId, controller);

    const settings = this.store.getSettings();
    const messagesPayload = this.messages
      .filter((msg) => {
        if (msg.role === 'assistant' && msg.status !== 'done') {
          return false;
        }
        if (msg.role === 'assistant' && !msg.content) {
          return false;
        }
        if (msg.role === 'system' && msg.content.trim() === '') {
          return false;
        }
        return true;
      })
      .map((msg) => ({ role: msg.role, content: msg.content }));

    const payload = {
      model: settings.model,
      messages: messagesPayload,
      temperature: settings.temperature,
      top_p: settings.top_p,
      max_tokens: typeof maxTokensOverride === 'number' ? maxTokensOverride : settings.max_tokens,
    };

    const handle = this.queue.enqueue(async () => {
      this.updateAssistantStatus(assistantId, 'pending');
      try {
        const result = await this.api.sendChat(payload, {
          signal: controller.signal,
          onDelta: (delta) => {
            this.appendAssistantContent(assistantId, delta);
          },
          onStart: ({ requestId }) => {
            this.setRequestId(assistantId, requestId);
          },
          onUsage: (usage) => {
            this.setUsage(assistantId, usage);
            this.refreshMetrics();
          },
          onFinish: ({ finishReason, requestId }) => {
            this.setFinish(assistantId, finishReason, requestId);
            this.refreshMetrics();
          },
          onRetry: ({ delay, status, resume }) => {
            this.toasts.show({
              title: status === 429 ? '触发限流' : '请求将重试',
              message: `将在 ${Math.round(delay / 1000)}s 后重试 (HTTP ${status})`,
              actions: [
                {
                  label: '立即重试',
                  action: resume,
                },
                {
                  label: '取消',
                  action: () => controller.abort(),
                },
              ],
            });
          },
        });
        this.updateAssistantStatus(assistantId, 'done');
        if (result.finishReason === 'length') {
          const continueCount = (this.continueMap.get(assistantId) || 0) + 1;
          if (continueCount <= AUTO_CONTINUE_LIMIT) {
            this.continueMap.set(assistantId, continueCount);
            this.toasts.show({
              title: '自动续写',
              message: `检测到长度截断，正在进行第 ${continueCount} 次续写。`,
            });
            this.addMessage({
              id: createId(),
              role: 'user',
              content: 'continue',
              status: 'done',
              createdAt: Date.now(),
            });
            const previous = typeof payload.max_tokens === 'number' ? payload.max_tokens : settings.max_tokens;
            const nextMax = Math.max(64, Math.floor(previous * 0.8));
            this.scheduleAssistantReply(nextMax);
          } else {
            this.toasts.show({
              title: '续写已达到上限',
              kind: 'info',
            });
          }
        }
      } catch (error) {
        if (error instanceof DOMException && error.name === 'AbortError') {
          this.updateAssistantStatus(assistantId, 'cancelled');
          this.toasts.show({ title: '生成已取消', kind: 'info', duration: 2000 });
        } else if (error instanceof ApiError) {
          this.updateAssistantStatus(assistantId, 'error', error.message);
          this.metrics.recordFailure(String(error.status));
          this.refreshMetrics();
          this.toasts.show({ title: '请求失败', message: error.message, kind: 'error' });
        } else {
          console.error(error);
          this.updateAssistantStatus(assistantId, 'error', '未知错误');
          this.metrics.recordFailure('unknown');
          this.refreshMetrics();
          this.toasts.show({ title: '未知错误', kind: 'error' });
        }
      } finally {
        this.activeControllers.delete(assistantId);
        this.refreshQueueIndicator();
      }
    });

    handle.promise.catch(() => {});
    this.refreshQueueIndicator();
  }

  updateAssistantStatus(id, status, error) {
    const message = this.messages.find((msg) => msg.id === id);
    if (!message) return;
    message.status = status;
    if (error) {
      message.error = error;
    } else if (status !== 'error') {
      delete message.error;
    }
    const element = this.findMessageElement(id);
    if (element) {
      const isGenerating = status === 'queued' || status === 'pending' || status === 'streaming';
      element.classList.toggle('is-streaming', isGenerating);
      element.classList.toggle('has-error', status === 'error');
      element.classList.toggle('is-cancelled', status === 'cancelled');
      const meta = element.querySelector('.meta');
      if (meta) {
        this.updateMeta(meta, message);
      }
    }
  }

  appendAssistantContent(id, delta) {
    const message = this.messages.find((msg) => msg.id === id);
    if (!message) return;
    message.status = 'streaming';
    message.content += delta;
    const element = this.findMessageElement(id);
    if (element) {
      element.classList.add('is-streaming');
      const bubble = element.querySelector('.bubble');
      if (bubble) {
        const contentEl = bubble.querySelector('.content');
        if (contentEl) {
          contentEl.textContent = message.content;
        }
      }
      const meta = element.querySelector('.meta');
      if (meta) {
        this.updateMeta(meta, message);
      }
      element.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }
  }

  setFinish(id, finishReason, requestId) {
    const message = this.messages.find((msg) => msg.id === id);
    if (!message) return;
    if (finishReason) {
      message.finishReason = finishReason;
    }
    if (requestId) {
      message.requestId = requestId;
    }
    const element = this.findMessageElement(id);
    if (element) {
      const meta = element.querySelector('.meta');
      if (meta) {
        this.updateMeta(meta, message);
      }
    }
  }

  setRequestId(id, requestId) {
    const message = this.messages.find((msg) => msg.id === id);
    if (!message) return;
    if (requestId) {
      message.requestId = requestId;
    }
  }

  setUsage(id, usage) {
    const message = this.messages.find((msg) => msg.id === id);
    if (!message) return;
    message.usage = usage;
    const element = this.findMessageElement(id);
    if (element) {
      const meta = element.querySelector('.meta');
      if (meta) {
        this.updateMeta(meta, message);
      }
    }
  }

  retryMessage(id) {
    const message = this.messages.find((msg) => msg.id === id);
    if (!message) return;
    if (message.status !== 'error') {
      this.toasts.show({ title: '该消息无需重试', kind: 'info', duration: 2000 });
      return;
    }
    message.status = 'cancelled';
    this.scheduleAssistantReply();
    const element = this.findMessageElement(id);
    if (element) {
      element.classList.remove('has-error');
      element.classList.remove('is-streaming');
      const meta = element.querySelector('.meta');
      if (meta) {
        this.updateMeta(meta, message);
      }
    }
  }

  refreshQueueIndicator() {
    const pending = this.queue.getPendingCount();
    if (this.elements.queueIndicator) {
      this.elements.queueIndicator.hidden = pending === 0;
      if (pending > 0) {
        this.elements.queueIndicator.textContent = pending === 1 ? '正在生成中…' : `队列中还有 ${pending} 个请求`;
      } else {
        this.elements.queueIndicator.textContent = '';
      }
    }
    this.elements.cancelBtn.disabled = this.activeControllers.size === 0;
  }

  updateMetricsToggleLabel() {
    const toggle = this.elements.metricsToggle;
    const panel = this.elements.metricsPanel;
    if (!toggle || !panel) return;
    const collapsed = panel.classList.contains('collapsed');
    toggle.textContent = collapsed ? '展开监控' : '折叠监控';
    toggle.setAttribute('aria-expanded', String(!collapsed));
  }

  refreshMetrics() {
    const totals = this.metrics.getTokenTotals();
    if (this.elements.metricPrompt) this.elements.metricPrompt.textContent = String(totals.prompt);
    if (this.elements.metricCompletion) this.elements.metricCompletion.textContent = String(totals.completion);
    if (this.elements.metricTotal) this.elements.metricTotal.textContent = String(totals.total);
    if (this.elements.metricTTFTList) this.elements.metricTTFTList.textContent = this.metrics
      .getTTFTList()
      .slice(-10)
      .join(', ') || '-';
    const p95 = this.metrics.getTTFTP95();
    if (this.elements.metricTTFTP95) this.elements.metricTTFTP95.textContent = p95 != null ? String(p95) : '-';
    if (this.elements.metric429) this.elements.metric429.textContent = String(this.metrics.getRecent429());
    if (this.elements.metricRetries) this.elements.metricRetries.textContent = String(this.metrics.getRecentRetries());
    if (this.elements.metricFailures) {
      const failures = this.metrics.getFailureList();
      this.elements.metricFailures.innerHTML = '';
      if (failures.length === 0) {
        const li = document.createElement('li');
        li.textContent = '无记录';
        this.elements.metricFailures.appendChild(li);
      } else {
        failures.forEach((failure) => {
          const li = document.createElement('li');
          li.textContent = `${failure.kind}: ${failure.count}`;
          this.elements.metricFailures.appendChild(li);
        });
      }
    }
  }

  openSettings() {
    this.syncSettingsToForm();
    this.elements.settingsModal?.removeAttribute('hidden');
    this.elements.settingsModal?.focus();
  }

  closeSettings() {
    this.elements.settingsModal?.setAttribute('hidden', '');
  }

  syncSettingsToForm() {
    const settings = this.store.getSettings();
    if (this.elements.baseUrlInput) this.elements.baseUrlInput.value = settings.baseUrl;
    if (this.elements.modelSelect) {
      if (!Array.from(this.elements.modelSelect.options).some((opt) => opt.value === settings.model)) {
        const option = document.createElement('option');
        option.value = settings.model;
        option.textContent = settings.model;
        this.elements.modelSelect.appendChild(option);
      }
      this.elements.modelSelect.value = settings.model;
    }
    if (this.elements.apiKeyInput) this.elements.apiKeyInput.value = this.store.getApiKey();
    if (this.elements.rememberKeyCheckbox) this.elements.rememberKeyCheckbox.checked = settings.rememberKey;
    if (this.elements.temperatureInput) this.elements.temperatureInput.value = String(settings.temperature);
    if (this.elements.topPInput) this.elements.topPInput.value = String(settings.top_p);
    if (this.elements.maxTokensInput) this.elements.maxTokensInput.value = String(settings.max_tokens);
  }

  saveSettings() {
    const baseUrl = this.elements.baseUrlInput?.value.trim() || DEFAULT_SETTINGS.baseUrl;
    const model = this.elements.modelSelect?.value.trim() || DEFAULT_SETTINGS.model;
    const temperature = Number(this.elements.temperatureInput?.value) || DEFAULT_SETTINGS.temperature;
    const topP = Number(this.elements.topPInput?.value) || DEFAULT_SETTINGS.top_p;
    const maxTokens = Number(this.elements.maxTokensInput?.value) || DEFAULT_SETTINGS.max_tokens;
    const rememberKey = Boolean(this.elements.rememberKeyCheckbox?.checked);
    const apiKey = this.elements.apiKeyInput?.value.trim() || '';
    this.store.updateSettings({ baseUrl, model, temperature, top_p: topP, max_tokens: maxTokens, rememberKey });
    if (apiKey) {
      this.store.setApiKey(apiKey, rememberKey);
    } else {
      this.store.clearApiKey();
    }
    this.closeSettings();
    this.toasts.show({ title: '设置已保存', kind: 'success' });
  }

  async fetchModels() {
    try {
      const ids = await this.api.listModels(new AbortController().signal);
      if (this.elements.modelSelect) {
        this.elements.modelSelect.innerHTML = '';
        ids.forEach((id) => {
          const option = document.createElement('option');
          option.value = id;
          option.textContent = id;
          this.elements.modelSelect.appendChild(option);
        });
      }
      if (ids.length === 0) {
        this.toasts.show({ title: '未返回模型，已进入离线模式', kind: 'error' });
        const inputOption = document.createElement('option');
        inputOption.value = this.store.getSettings().model;
        inputOption.textContent = this.store.getSettings().model;
        this.elements.modelSelect?.appendChild(inputOption);
      } else {
        this.toasts.show({ title: '模型列表已刷新', kind: 'success' });
        this.store.updateSettings({ model: ids[0] });
      }
    } catch (error) {
      console.warn(error);
      this.toasts.show({ title: '获取模型失败', kind: 'error', message: '请检查网络或密钥' });
      const option = document.createElement('option');
      option.value = this.store.getSettings().model;
      option.textContent = this.store.getSettings().model + ' (手动)';
      this.elements.modelSelect?.appendChild(option);
    }
  }

  async testConnection() {
    const controller = new AbortController();
    setTimeout(() => controller.abort(), 8000);
    try {
      await this.api.listModels(controller.signal);
      this.toasts.show({ title: '连接正常', kind: 'success' });
    } catch (error) {
      if (error instanceof ApiError && error.status === 401) {
        this.toasts.show({ title: '密钥无效或过期', kind: 'error' });
      } else {
        this.toasts.show({ title: '连接失败', kind: 'error' });
      }
    }
  }
}

function runSelfTests() {
  const results = [];
  const assert = (condition, message) => {
    if (!condition) {
      throw new Error(message);
    }
  };

  const backoff = new Backoff([1000, 2000, 4000, 8000]);
  assert(backoff.next(0) === 1000, 'Backoff step 0 应为 1000');
  assert(backoff.next(3) === 8000, 'Backoff step 3 应为 8000');
  assert(backoff.next(4) === null, 'Backoff 超出范围返回 null');
  results.push('Backoff ok');

  const collected = [];
  const parser = new SseParser((data) => collected.push(data));
  parser.feed('data: hello\n');
  parser.feed('\n');
  parser.feed('data: {"foo":"bar"}\n\n');
  parser.close();
  assert(collected.length === 2, 'SSE 解析应获得两个事件');
  assert(collected[0] === 'hello', 'SSE 第一个事件为 hello');
  results.push('SseParser ok');

  const metrics = new Metrics();
  [10, 20, 30, 40, 50].forEach((value) => metrics.recordTTFT(value));
  const p95 = metrics.getTTFTP95();
  assert(p95 != null && p95 >= 10 && p95 <= 50, 'TTFT p95 范围正确');
  results.push('Metrics ok');

  console.info('自测通过:', results.join(', '));
}

runSelfTests();

const toastContainer = document.getElementById('toast-container');
if (!toastContainer) {
  throw new Error('未找到 toast 容器');
}
const toasts = new Toasts(toastContainer);
const store = new Store();
const metrics = new Metrics();
const api = new ApiClient(store, metrics);
const app = new ChatApp(store, api, metrics, toasts);
app.init();

