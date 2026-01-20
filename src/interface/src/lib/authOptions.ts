import type { NextAuthOptions } from "next-auth";
import GitHubProvider from "next-auth/providers/github";

export const authOptions: NextAuthOptions = {
  providers: [
    GitHubProvider({
      clientId: process.env.GITHUB_CLIENT_ID!,
      clientSecret: process.env.GITHUB_CLIENT_SECRET!,
    }),
  ],
  callbacks: {
    async jwt({ token, account, profile }) {
      // Persist the GitHub login to the token.
      //
      // We use this value as `X-User-ID` when talking to the backend so the
      // allowlist can be managed using human-readable GitHub usernames.
      if (account && profile) {
        const p = profile as { id?: number; login?: string };
        token.id = p.login?.toString();
        token.githubId = p.id?.toString();
      }
      return token;
    },
    async session({ session, token }) {
      // Include the GitHub login in the session (used as `X-User-ID`).
      if (session.user) {
        (session.user as { id?: string }).id = token.id as string;
        (session.user as { githubId?: string }).githubId = token.githubId as string;
      }
      return session;
    },
  },
};
