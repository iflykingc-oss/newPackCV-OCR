/**
 * PackCV SDK 异常体系
 */

export class PackCVError extends Error {
  public readonly status: number;
  public readonly code: string;
  public readonly details: Record<string, unknown> | undefined;

  constructor(
    message: string,
    status: number = 0,
    code: string = 'unknown_error',
    details?: Record<string, unknown>
  ) {
    super(message);
    this.name = 'PackCVError';
    this.status = status;
    this.code = code;
    this.details = details;
    Object.setPrototypeOf(this, new.target.prototype);
  }

  toString(): string {
    return `[${this.name}] ${this.code} (${this.status}): ${this.message}`;
  }
}

export class AuthenticationError extends PackCVError {
  constructor(message: string = 'Invalid or missing API key', details?: Record<string, unknown>) {
    super(message, 401, 'authentication_error', details);
    this.name = 'AuthenticationError';
  }
}

export class RateLimitError extends PackCVError {
  public readonly retry_after: number | undefined;

  constructor(
    message: string = 'Rate limit exceeded',
    retry_after?: number,
    details?: Record<string, unknown>
  ) {
    super(message, 429, 'rate_limit_error', details);
    this.name = 'RateLimitError';
    this.retry_after = retry_after;
  }
}

export class QuotaExceededError extends PackCVError {
  constructor(message: string = 'Monthly quota exceeded', details?: Record<string, unknown>) {
    super(message, 402, 'quota_exceeded', details);
    this.name = 'QuotaExceededError';
  }
}

export class ValidationError extends PackCVError {
  constructor(message: string, details?: Record<string, unknown>) {
    super(message, 400, 'validation_error', details);
    this.name = 'ValidationError';
  }
}

export class ServerError extends PackCVError {
  constructor(message: string = 'Internal server error', status: number = 500, details?: Record<string, unknown>) {
    super(message, status, 'server_error', details);
    this.name = 'ServerError';
  }
}

export class TimeoutError extends PackCVError {
  constructor(message: string = 'Request timeout', details?: Record<string, unknown>) {
    super(message, 408, 'timeout', details);
    this.name = 'TimeoutError';
  }
}

export class NetworkError extends PackCVError {
  constructor(message: string = 'Network error', details?: Record<string, unknown>) {
    super(message, 0, 'network_error', details);
    this.name = 'NetworkError';
  }
}

/** 根据状态码映射到对应异常 */
export function errorFromStatus(
  status: number,
  message: string,
  details?: Record<string, unknown>
): PackCVError {
  if (status === 401 || status === 403) {
    return new AuthenticationError(message, details);
  }
  if (status === 429) {
    const retryAfter = typeof details?.['retry_after'] === 'number' ? details['retry_after'] : undefined;
    return new RateLimitError(message, retryAfter, details);
  }
  if (status === 402) {
    return new QuotaExceededError(message, details);
  }
  if (status >= 400 && status < 500) {
    return new ValidationError(message, details);
  }
  if (status === 408) {
    return new TimeoutError(message, details);
  }
  if (status >= 500) {
    return new ServerError(message, status, details);
  }
  return new PackCVError(message, status, 'unknown_error', details);
}
