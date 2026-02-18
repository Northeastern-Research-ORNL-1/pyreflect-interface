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
          src="http://localhost:8000/actuator.js"
          data-public-key="pk_yL8jdHHZkup30OMOjlodRWzwANCuto51"
          data-api-url="http://localhost:8000"
          data-debug="true"
          data-always-record="true"
        />
        <Script
          src="https://cdn.jsdelivr.net/npm/@rrweb/record@latest/dist/record.umd.min.cjs"
        />
      </head>
      <body>
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}
