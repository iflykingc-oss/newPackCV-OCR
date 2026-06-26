/**
 * 工具函数
 */

/** 等待毫秒 */
export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** 超时控制 */
export async function withTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
  errorMessage: string = 'Operation timeout'
): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | null = null;
  const timeoutPromise = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error(errorMessage)), timeoutMs);
  });
  try {
    return (await Promise.race([promise, timeoutPromise])) as T;
  } finally {
    if (timer) clearTimeout(timer);
  }
}

/** 指数退避重试 */
export interface RetryOptions {
  max_retries: number;
  base_delay: number;
  max_delay: number;
  retry_on?: number[];
}

export async function retry<T>(
  fn: () => Promise<T>,
  options: RetryOptions
): Promise<T> {
  const { max_retries, base_delay, max_delay, retry_on = [408, 429, 500, 502, 503, 504] } = options;
  let lastError: unknown;

  for (let attempt = 0; attempt <= max_retries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastError = err;
      const status = (err as { status?: number })?.status ?? 0;
      if (attempt >= max_retries || !retry_on.includes(status)) {
        throw err;
      }
      // 指数退避 + 抖动
      const delay = Math.min(base_delay * 2 ** attempt, max_delay);
      const jitter = Math.random() * delay * 0.1;
      await sleep(delay + jitter);
    }
  }
  throw lastError;
}
