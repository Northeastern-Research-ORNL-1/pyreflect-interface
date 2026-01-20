// @vitest-environment node

import { describe, expect, it, vi } from 'vitest';

vi.mock('next-auth/providers/github', () => ({
  default: (config: unknown) => ({ id: 'github', config }),
}));

describe('authOptions', () => {
  it('persists github user id into token and session', async () => {
    vi.resetModules();
    process.env.GITHUB_CLIENT_ID = 'id';
    process.env.GITHUB_CLIENT_SECRET = 'secret';

    const { authOptions } = await import('@/lib/authOptions');

    const token: Record<string, unknown> = {};
    const updatedToken = await authOptions.callbacks!.jwt!({
      token,
      account: {},
      profile: { id: 123 },
    } as never);
    expect(updatedToken.id).toBe('123');

    const session = { user: {} } as { user: { id?: string } };
    const updatedSession = await authOptions.callbacks!.session!({
      session,
      token: updatedToken,
    } as never);
    expect(updatedSession.user.id).toBe('123');
  });

  it('does not overwrite token id when missing account/profile', async () => {
    vi.resetModules();
    process.env.GITHUB_CLIENT_ID = 'id';
    process.env.GITHUB_CLIENT_SECRET = 'secret';
    const { authOptions } = await import('@/lib/authOptions');

    const token: Record<string, unknown> = { id: 'existing' };
    const updatedToken = await authOptions.callbacks!.jwt!({
      token,
      account: null,
      profile: null,
    } as never);
    expect(updatedToken.id).toBe('existing');
  });
});
