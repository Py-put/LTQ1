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
const PIN_STORAGE_KEY = 'gradient-pins';
const PROMPT_HISTORY_KEY = 'gradient-prompts';
const MAX_PROMPT_HISTORY = 6;
const MAX_PINNED = 30;

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

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (char) => {
    switch (char) {
      case '&':
        return '&amp;';
      case '<':
        return '&lt;';
      case '>':
        return '&gt;';
      case '"':
        return '&quot;';
      case "'":
        return '&#39;';
      default:
        return char;
    }
  });
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\$&');
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
   */
  constructor(store) {
    this.store = store;
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
        if (!error.retry) {
          throw error;
        }
        const delayMs = error.retryAfter ? error.retryAfter * 1000 : this.backoff.next(attempt);
        if (delayMs == null) {
          throw error;
        }
        attempt += 1;
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
    const parser = new SseParser((data) => {
      if (data === '[DONE]') {
        return;
      }
      try {
        const json = JSON.parse(data);
        const choice = Array.isArray(json.choices) ? json.choices[0] : undefined;
        if (choice && choice.delta && typeof choice.delta.content === 'string') {
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
   * @param {Toasts} toasts
   */
  constructor(store, api, toasts) {
    this.store = store;
    this.api = api;
    this.toasts = toasts;
    /** @type {ChatMessage[]} */
    this.messages = [];
    this.queue = new RequestQueue(MAX_CONCURRENT);
    this.activeControllers = new Map();
    this.continueMap = new Map();
    this.searchMatches = [];
    this.highlightedMessageId = null;
    this.highlightTimer = 0;
    this.pinnedMessages = this.loadPinnedMessages();
    this.promptHistory = this.loadPromptHistory();

    this.elements = {
      messageList: document.getElementById('message-list'),
      textarea: /** @type {HTMLTextAreaElement} */ (document.getElementById('input-text')),
      composer: /** @type {HTMLFormElement} */ (document.getElementById('composer')),
      sendBtn: /** @type {HTMLButtonElement} */ (document.getElementById('send-btn')),
      cancelBtn: /** @type {HTMLButtonElement} */ (document.getElementById('cancel-btn')),
      clearBtn: /** @type {HTMLButtonElement} */ (document.getElementById('clear-btn')),
      exportBtn: /** @type {HTMLButtonElement} */ (document.getElementById('export-btn')),
      queueIndicator: document.getElementById('queue-indicator'),
      searchBtn: /** @type {HTMLButtonElement} */ (document.getElementById('search-btn')),
      searchModal: /** @type {HTMLElement} */ (document.getElementById('search-modal')),
      searchClose: /** @type {HTMLButtonElement} */ (document.getElementById('search-close')),
      searchForm: /** @type {HTMLFormElement} */ (document.getElementById('search-form')),
      searchInput: /** @type {HTMLInputElement} */ (document.getElementById('search-input')),
      searchResults: /** @type {HTMLElement} */ (document.getElementById('search-results')),
      pinsBtn: /** @type {HTMLButtonElement} */ (document.getElementById('pins-btn')),
      pinsModal: /** @type {HTMLElement} */ (document.getElementById('pins-modal')),
      pinsClose: /** @type {HTMLButtonElement} */ (document.getElementById('pins-close')),
      pinsClear: /** @type {HTMLButtonElement} */ (document.getElementById('pins-clear')),
      pinsList: /** @type {HTMLElement} */ (document.getElementById('pins-list')),
      promptHistory: document.getElementById('prompt-history'),
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
    this.renderPromptHistory();
    this.renderPins();
    this.refreshPinButtons();
    this.syncSettingsToForm();
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
    this.store.addEventListener('change', () => this.syncSettingsToForm());
    this.store.addEventListener('key-change', () => this.syncSettingsToForm());
    this.elements.searchBtn?.addEventListener('click', () => this.openSearch());
    this.elements.searchClose?.addEventListener('click', () => this.closeSearch());
    this.elements.searchModal?.addEventListener('click', (event) => {
      if (event.target === this.elements.searchModal) {
        this.closeSearch();
      }
    });
    this.elements.searchInput?.addEventListener('input', () => {
      const value = this.elements.searchInput?.value || '';
      this.updateSearchResults(value);
    });
    this.elements.searchInput?.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        this.closeSearch();
      }
    });
    this.elements.searchForm?.addEventListener('submit', (event) => {
      event.preventDefault();
      this.selectSearchResult(0);
    });
    this.elements.pinsBtn?.addEventListener('click', () => this.openPins());
    this.elements.pinsClose?.addEventListener('click', () => this.closePins());
    this.elements.pinsModal?.addEventListener('click', (event) => {
      if (event.target === this.elements.pinsModal) {
        this.closePins();
      }
    });
    this.elements.pinsClear?.addEventListener('click', () => {
      this.clearPins();
    });
    this.elements.pinsList?.addEventListener('click', (event) => {
      const target = event.target;
      if (!(target instanceof HTMLElement)) return;
      const action = target.dataset.action;
      const id = target.dataset.messageId;
      if (!action || !id) return;
      if (action === 'copy') {
        const pin = this.pinnedMessages.find((item) => item.id === id);
        if (!pin) return;
        if (!navigator.clipboard || typeof navigator.clipboard.writeText !== 'function') {
          this.toasts.show({ title: '当前环境不支持复制', kind: 'error' });
          return;
        }
        navigator.clipboard
          .writeText(pin.content)
          .then(() => {
            this.toasts.show({ title: '已复制到剪贴板', kind: 'success', duration: 2000 });
          })
          .catch(() => {
            this.toasts.show({ title: '复制失败', kind: 'error' });
          });
      } else if (action === 'fill') {
        const pin = this.pinnedMessages.find((item) => item.id === id);
        if (!pin || !this.elements.textarea) return;
        this.elements.textarea.value = pin.content;
        this.autosizeTextarea();
        this.elements.textarea.focus();
      } else if (action === 'remove') {
        this.togglePin(id);
      } else if (action === 'locate') {
        this.highlightMessage(id);
        this.closePins();
      }
    });
    this.elements.promptHistory?.addEventListener('click', (event) => {
      const target = event.target;
      if (!(target instanceof HTMLElement)) return;
      const action = target.dataset.action;
      const indexAttr = target.dataset.index;
      if (!action || indexAttr == null) return;
      const index = Number(indexAttr);
      if (Number.isNaN(index)) return;
      if (action === 'fill') {
        const prompt = this.promptHistory[index];
        if (prompt && this.elements.textarea) {
          this.elements.textarea.value = prompt;
          this.autosizeTextarea();
          this.elements.textarea.focus();
        }
      } else if (action === 'remove') {
        this.promptHistory.splice(index, 1);
        this.savePromptHistory();
        this.renderPromptHistory();
      }
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
    window.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') {
        const shouldCloseSearch = this.elements.searchModal ? !this.elements.searchModal.hasAttribute('hidden') : false;
        const shouldClosePins = this.elements.pinsModal ? !this.elements.pinsModal.hasAttribute('hidden') : false;
        const shouldCloseSettings = this.elements.settingsModal ? !this.elements.settingsModal.hasAttribute('hidden') : false;
        if (shouldCloseSearch) {
          event.preventDefault();
          this.closeSearch();
          return;
        }
        if (shouldClosePins) {
          event.preventDefault();
          this.closePins();
          return;
        }
        if (shouldCloseSettings) {
          event.preventDefault();
          this.closeSettings();
        }
      }
    });
    window.addEventListener('keydown', (event) => {
      if ((event.key === 'k' || event.key === 'K') && (event.metaKey || event.ctrlKey)) {
        const target = event.target;
        if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement) {
          return;
        }
        if (target && typeof target.getAttribute === 'function' && target.getAttribute('contenteditable') === 'true') {
          return;
        }
        event.preventDefault();
        this.openSearch();
      }
    });
  }

  openSearch() {
    this.elements.searchModal?.removeAttribute('hidden');
    const input = this.elements.searchInput;
    if (input) {
      input.focus();
      input.select();
      this.updateSearchResults(input.value);
    }
  }

  closeSearch() {
    this.elements.searchModal?.setAttribute('hidden', '');
    const input = this.elements.searchInput;
    if (input) {
      input.blur();
    }
    this.elements.textarea?.focus();
  }

  updateSearchResults(rawTerm) {
    const container = this.elements.searchResults;
    if (!container) return;
    const term = rawTerm.trim();
    if (term.length === 0) {
      this.searchMatches = [];
      this.renderSearchResults([], { rawTerm: term, normalizedTerm: '', tooShort: false });
      return;
    }
    if (term.length < 2) {
      this.searchMatches = [];
      this.renderSearchResults([], { rawTerm: term, normalizedTerm: '', tooShort: true });
      return;
    }
    const normalized = term.toLowerCase();
    const matches = this.messages
      .filter((message) => message.role !== 'system' && message.content && message.content.toLowerCase().includes(normalized))
      .map((message) => ({
        message,
        index: message.content.toLowerCase().indexOf(normalized),
      }))
      .sort((a, b) => b.message.createdAt - a.message.createdAt);
    this.searchMatches = matches.map((entry) => entry.message);
    this.renderSearchResults(matches, { rawTerm: term, normalizedTerm: normalized, tooShort: false });
  }

  renderSearchResults(matches, options) {
    const container = this.elements.searchResults;
    if (!container) return;
    container.innerHTML = '';
    const { rawTerm, normalizedTerm, tooShort } = options;
    if (!rawTerm) {
      const info = document.createElement('div');
      info.className = 'search-empty';
      info.textContent = '输入关键词开始查找。';
      container.appendChild(info);
      return;
    }
    if (tooShort) {
      const info = document.createElement('div');
      info.className = 'search-empty';
      info.textContent = '请输入至少 2 个字符。';
      container.appendChild(info);
      return;
    }
    if (matches.length === 0) {
      const info = document.createElement('div');
      info.className = 'search-empty';
      info.textContent = `未找到包含 “${rawTerm}” 的消息。`;
      container.appendChild(info);
      return;
    }
    matches.forEach((entry, index) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'search-result';
      button.setAttribute('role', 'listitem');
      button.dataset.messageId = entry.message.id;
      const meta = document.createElement('div');
      meta.className = 'meta';
      const roleLabel = entry.message.role === 'assistant' ? 'AI' : '我';
      meta.textContent = `${roleLabel} · ${new Date(entry.message.createdAt).toLocaleString()}`;
      const snippetEl = document.createElement('div');
      snippetEl.className = 'snippet';
      const snippetText = entry.message.content || '';
      const matchIndex = entry.index >= 0 ? entry.index : 0;
      const radius = 60;
      const start = Math.max(0, matchIndex - radius);
      const end = Math.min(snippetText.length, matchIndex + normalizedTerm.length + radius);
      let snippet = snippetText.slice(start, end);
      const prefix = start > 0 ? '…' : '';
      const suffix = end < snippetText.length ? '…' : '';
      const escaped = escapeHtml(snippet);
      const highlightRegex = new RegExp(`(${escapeRegExp(normalizedTerm)})`, 'gi');
      const highlighted = normalizedTerm ? escaped.replace(highlightRegex, '<mark>$1</mark>') : escaped;
      snippetEl.innerHTML = `${prefix}${highlighted}${suffix}`;
      button.appendChild(meta);
      button.appendChild(snippetEl);
      button.addEventListener('click', () => {
        this.selectSearchResult(index);
      });
      container.appendChild(button);
    });
  }

  selectSearchResult(index) {
    const target = this.searchMatches[index];
    if (!target) {
      return;
    }
    this.highlightMessage(target.id);
    this.closeSearch();
  }

  highlightMessage(id) {
    const element = this.findMessageElement(id);
    if (!element) return;
    if (this.highlightedMessageId && this.highlightedMessageId !== id) {
      const previous = this.findMessageElement(this.highlightedMessageId);
      previous?.classList.remove('is-highlighted');
    }
    element.classList.add('is-highlighted');
    element.scrollIntoView({ behavior: 'smooth', block: 'center' });
    if (this.highlightTimer) {
      clearTimeout(this.highlightTimer);
    }
    this.highlightedMessageId = id;
    this.highlightTimer = window.setTimeout(() => {
      element.classList.remove('is-highlighted');
      if (this.highlightedMessageId === id) {
        this.highlightedMessageId = null;
      }
    }, 2200);
  }

  openPins() {
    this.elements.pinsModal?.removeAttribute('hidden');
    this.renderPins();
    this.elements.pinsClose?.focus();
  }

  closePins() {
    this.elements.pinsModal?.setAttribute('hidden', '');
    this.elements.textarea?.focus();
  }

  loadPinnedMessages() {
    try {
      const raw = localStorage.getItem(PIN_STORAGE_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) {
        return parsed
          .filter((item) => item && typeof item.id === 'string' && typeof item.content === 'string')
          .slice(0, MAX_PINNED);
      }
    } catch (error) {
      console.warn('无法读取收藏夹', error);
    }
    return [];
  }

  savePinnedMessages() {
    try {
      localStorage.setItem(PIN_STORAGE_KEY, JSON.stringify(this.pinnedMessages.slice(0, MAX_PINNED)));
    } catch (error) {
      console.warn('无法保存收藏夹', error);
    }
  }

  renderPins() {
    const container = this.elements.pinsList;
    if (!container) return;
    container.innerHTML = '';
    if (this.pinnedMessages.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'pin-empty';
      empty.textContent = '暂无收藏，点击消息右上角的“收藏”即可保存重点内容。';
      container.appendChild(empty);
      if (this.elements.pinsClear) {
        this.elements.pinsClear.disabled = true;
      }
      return;
    }
    if (this.elements.pinsClear) {
      this.elements.pinsClear.disabled = false;
    }
    this.pinnedMessages.forEach((pin) => {
      const item = document.createElement('div');
      item.className = 'pin-item';
      item.setAttribute('role', 'listitem');
      const meta = document.createElement('div');
      meta.className = 'meta';
      const roleLabel = pin.role === 'assistant' ? 'AI' : '我';
      meta.textContent = `${roleLabel} · ${new Date(pin.createdAt).toLocaleString()}`;
      const snippet = document.createElement('div');
      snippet.className = 'pin-snippet';
      snippet.textContent = pin.content;
      const actions = document.createElement('div');
      actions.className = 'pin-actions';

      const locateBtn = document.createElement('button');
      locateBtn.type = 'button';
      locateBtn.textContent = '定位';
      locateBtn.dataset.action = 'locate';
      locateBtn.dataset.messageId = pin.id;
      actions.appendChild(locateBtn);

      const fillBtn = document.createElement('button');
      fillBtn.type = 'button';
      fillBtn.textContent = '填入输入框';
      fillBtn.dataset.action = 'fill';
      fillBtn.dataset.messageId = pin.id;
      actions.appendChild(fillBtn);

      const copyBtn = document.createElement('button');
      copyBtn.type = 'button';
      copyBtn.textContent = '复制内容';
      copyBtn.dataset.action = 'copy';
      copyBtn.dataset.messageId = pin.id;
      actions.appendChild(copyBtn);

      const removeBtn = document.createElement('button');
      removeBtn.type = 'button';
      removeBtn.textContent = '取消收藏';
      removeBtn.dataset.action = 'remove';
      removeBtn.dataset.messageId = pin.id;
      actions.appendChild(removeBtn);

      item.appendChild(meta);
      item.appendChild(snippet);
      item.appendChild(actions);
      container.appendChild(item);
    });
  }

  clearPins() {
    if (this.pinnedMessages.length === 0) {
      return;
    }
    this.pinnedMessages = [];
    this.savePinnedMessages();
    this.renderPins();
    this.refreshPinButtons();
    this.toasts.show({ title: '收藏夹已清空', kind: 'info' });
  }

  togglePin(id) {
    const existingIndex = this.pinnedMessages.findIndex((item) => item.id === id);
    if (existingIndex >= 0) {
      this.pinnedMessages.splice(existingIndex, 1);
      this.savePinnedMessages();
      this.renderPins();
      this.refreshPinButtons();
      this.toasts.show({ title: '已取消收藏', kind: 'info', duration: 2000 });
      return;
    }
    const message = this.messages.find((msg) => msg.id === id);
    if (!message || !message.content) {
      this.toasts.show({ title: '暂无可收藏的内容', kind: 'error', duration: 2000 });
      return;
    }
    this.pinnedMessages.unshift({
      id: message.id,
      role: message.role,
      content: message.content,
      createdAt: message.createdAt,
    });
    if (this.pinnedMessages.length > MAX_PINNED) {
      this.pinnedMessages = this.pinnedMessages.slice(0, MAX_PINNED);
    }
    this.savePinnedMessages();
    this.renderPins();
    this.refreshPinButtons();
    this.toasts.show({ title: '已收藏', kind: 'success', duration: 2000 });
  }

  isPinned(id) {
    return this.pinnedMessages.some((item) => item.id === id);
  }

  updatePinButton(button, id) {
    const message = this.messages.find((msg) => msg.id === id);
    const hasContent = !!(message && message.content && message.content.trim().length > 0);
    button.textContent = this.isPinned(id) ? '取消收藏' : '收藏';
    button.dataset.action = 'pin';
    button.dataset.messageId = id;
    button.disabled = !hasContent;
  }

  refreshPinButtons() {
    const buttons = this.elements.messageList?.querySelectorAll('[data-pin-for]');
    buttons?.forEach((button) => {
      if (button instanceof HTMLButtonElement) {
        const id = button.dataset.pinFor;
        if (id) {
          this.updatePinButton(button, id);
        }
      }
    });
  }

  loadPromptHistory() {
    try {
      const raw = localStorage.getItem(PROMPT_HISTORY_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) {
        return parsed.filter((item) => typeof item === 'string').slice(0, MAX_PROMPT_HISTORY);
      }
    } catch (error) {
      console.warn('无法读取提示历史', error);
    }
    return [];
  }

  savePromptHistory() {
    try {
      localStorage.setItem(PROMPT_HISTORY_KEY, JSON.stringify(this.promptHistory.slice(0, MAX_PROMPT_HISTORY)));
    } catch (error) {
      console.warn('无法保存提示历史', error);
    }
  }

  renderPromptHistory() {
    const container = this.elements.promptHistory;
    if (!container) return;
    container.innerHTML = '';
    if (this.promptHistory.length === 0) {
      container.setAttribute('hidden', '');
      return;
    }
    container.removeAttribute('hidden');
    this.promptHistory.forEach((entry, index) => {
      const chip = document.createElement('div');
      chip.className = 'prompt-chip';
      const fill = document.createElement('button');
      fill.type = 'button';
      fill.className = 'prompt-fill';
      fill.dataset.action = 'fill';
      fill.dataset.index = String(index);
      fill.textContent = entry;
      const remove = document.createElement('button');
      remove.type = 'button';
      remove.className = 'prompt-remove';
      remove.dataset.action = 'remove';
      remove.dataset.index = String(index);
      remove.setAttribute('aria-label', '移除此提示');
      remove.textContent = '×';
      chip.appendChild(fill);
      chip.appendChild(remove);
      container.appendChild(chip);
    });
  }

  recordPrompt(prompt) {
    const value = prompt.trim();
    if (!value) return;
    this.promptHistory = this.promptHistory.filter((item) => item !== value);
    this.promptHistory.unshift(value);
    if (this.promptHistory.length > MAX_PROMPT_HISTORY) {
      this.promptHistory = this.promptHistory.slice(0, MAX_PROMPT_HISTORY);
    }
    this.savePromptHistory();
    this.renderPromptHistory();
  }

  /**
   * @param {ChatMessage} message
   */
  addMessage(message) {
    this.messages.push(message);
    const el = this.renderMessage(message);
    this.elements.messageList?.appendChild(el);
    el.scrollIntoView({ behavior: 'smooth', block: 'end' });
    if (this.elements.searchModal && !this.elements.searchModal.hasAttribute('hidden')) {
      const value = this.elements.searchInput?.value || '';
      this.updateSearchResults(value);
    }
    this.refreshPinButtons();
  }

  clearMessages() {
    this.messages = [];
    if (this.elements.messageList) {
      this.elements.messageList.innerHTML = '';
    }
    this.renderWelcome();
    if (this.elements.searchModal && !this.elements.searchModal.hasAttribute('hidden')) {
      const value = this.elements.searchInput?.value || '';
      this.updateSearchResults(value);
    }
    this.refreshPinButtons();
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

    const actions = document.createElement('div');
    actions.className = 'actions';
    if (message.role === 'assistant') {
      const retryBtn = document.createElement('button');
      retryBtn.type = 'button';
      retryBtn.textContent = '立即重试';
      retryBtn.addEventListener('click', () => this.retryMessage(message.id));
      actions.appendChild(retryBtn);
      const copyBtn = document.createElement('button');
      copyBtn.type = 'button';
      copyBtn.textContent = '复制';
      copyBtn.addEventListener('click', async () => {
        if (!navigator.clipboard || typeof navigator.clipboard.writeText !== 'function') {
          this.toasts.show({ title: '当前环境不支持复制', kind: 'error' });
          return;
        }
        try {
          await navigator.clipboard.writeText(message.content);
          this.toasts.show({ title: '已复制到剪贴板', kind: 'success', duration: 2000 });
        } catch (error) {
          this.toasts.show({ title: '复制失败', kind: 'error' });
        }
      });
      actions.appendChild(copyBtn);
    }
    const pinBtn = document.createElement('button');
    pinBtn.type = 'button';
    pinBtn.dataset.pinFor = message.id;
    pinBtn.addEventListener('click', () => this.togglePin(message.id));
    actions.appendChild(pinBtn);
    this.updatePinButton(pinBtn, message.id);
    bubble.appendChild(actions);
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
    this.recordPrompt(content);
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
          },
          onFinish: ({ finishReason, requestId }) => {
            this.setFinish(assistantId, finishReason, requestId);
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
          this.toasts.show({ title: '请求失败', message: error.message, kind: 'error' });
        } else {
          console.error(error);
          this.updateAssistantStatus(assistantId, 'error', '未知错误');
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
      if (this.elements.searchModal && !this.elements.searchModal.hasAttribute('hidden')) {
        const value = this.elements.searchInput?.value || '';
        this.updateSearchResults(value);
      }
      this.refreshPinButtons();
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

  console.info('自测通过:', results.join(', '));
}

runSelfTests();

const toastContainer = document.getElementById('toast-container');
if (!toastContainer) {
  throw new Error('未找到 toast 容器');
}
const toasts = new Toasts(toastContainer);
const store = new Store();
const api = new ApiClient(store);
const app = new ChatApp(store, api, toasts);
app.init();

