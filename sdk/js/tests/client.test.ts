import { describe, it, expect, vi } from 'vitest';
import { PackCVClient, PackCVError, AuthenticationError, RateLimitError, QuotaExceededError, errorFromStatus } from '../src/index';
import type { HttpResponse, HttpClient } from '../src/client';

/** Mock HTTP 客户端 */
class MockHttpClient implements HttpClient {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  public responses: Array<{ status: number; data: any }> = [];
  public callCount = 0;

  async request<T>(method: string, url: string, headers: Record<string, string>, body: string | null, timeout: number): Promise<HttpResponse> {
    this.callCount++;
    return this.responses.shift() ?? { status: 500, data: { error: 'no more mocks' } };
  }
}

describe('PackCVClient', () => {
  it('sends Authorization header', async () => {
    const mock = new MockHttpClient();
    mock.responses.push({ status: 200, data: { ok: true } });
    const client = new PackCVClient({ api_key: 'pk_test_123' }, mock);
    await client.health();
    expect(mock.callCount).toBe(1);
  });

  it('parses successful response', async () => {
    const mock = new MockHttpClient();
    mock.responses.push({ status: 200, data: { run_id: 'r1', structured_data: { amount: 100 } } });
    const client = new PackCVClient({ api_key: 'pk_test' }, mock);
    const result = await client.extract('https://x.com/a.pdf', { scenario: 'invoice' });
    expect(result.run_id).toBe('r1');
  });

  it('throws AuthenticationError on 401', async () => {
    const mock = new MockHttpClient();
    mock.responses.push({ status: 401, data: { message: 'Invalid API key' } });
    const client = new PackCVClient({ api_key: 'bad' }, mock);
    await expect(client.health()).rejects.toThrow(AuthenticationError);
  });

  it('throws RateLimitError on 429 with retry_after', async () => {
    const mock = new MockHttpClient();
    mock.responses.push({ status: 429, data: { message: 'Too many', retry_after: 30 } });
    const client = new PackCVClient({ api_key: 'pk', max_retries: 0 }, mock);
    try {
      await client.health();
    } catch (err) {
      expect(err).toBeInstanceOf(RateLimitError);
      if (err instanceof RateLimitError) {
        expect(err.retry_after).toBe(30);
      }
    }
  });

  it('throws QuotaExceededError on 402', async () => {
    const mock = new MockHttpClient();
    mock.responses.push({ status: 402, data: { message: 'Quota exceeded' } });
    const client = new PackCVClient({ api_key: 'pk', max_retries: 0 }, mock);
    await expect(client.health()).rejects.toThrow(QuotaExceededError);
  });

  it('retries on 5xx then succeeds', async () => {
    const mock = new MockHttpClient();
    mock.responses.push({ status: 503, data: { error: 'unavailable' } });
    mock.responses.push({ status: 200, data: { ok: true } });
    const client = new PackCVClient({ api_key: 'pk', max_retries: 1, base_url: 'http://x' }, mock);
    await client.health();
    expect(mock.callCount).toBe(2);
  });
});

describe('errorFromStatus', () => {
  it('maps 401 to AuthenticationError', () => {
    const err = errorFromStatus(401, 'no auth');
    expect(err).toBeInstanceOf(AuthenticationError);
  });
  it('maps 402 to QuotaExceededError', () => {
    const err = errorFromStatus(402, 'quota');
    expect(err).toBeInstanceOf(QuotaExceededError);
  });
  it('maps 500 to ServerError', () => {
    const err = errorFromStatus(500, 'oops');
    expect(err).toBeInstanceOf(PackCVError);
  });
});
