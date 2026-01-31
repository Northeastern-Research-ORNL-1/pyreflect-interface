import type { Metadata } from 'next';
import Script from 'next/script';
import './globals.css';
import AuthProvider from '../components/AuthProvider';

export const metadata: Metadata = {
  title: 'PyReflect',
  description: 'GUI',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <Script
          src="http://localhost:3000/actuator.js"
          data-pulse-key="pk_qCvxEgPqtuRY3s9chiAHaR0G2WFpBM32"
          strategy="beforeInteractive"
        />
      </head>
      <body>
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}
