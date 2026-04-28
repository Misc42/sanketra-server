"use client";

import { QRCodeSVG } from "qrcode.react";

export default function QRCode({ value }: { value: string }) {
  return (
    <div className="inline-flex rounded-md bg-[#f4efe6] p-4">
      <QRCodeSVG value={value} size={196} bgColor="#F4EFE6" fgColor="#0F0E14" level="M" />
    </div>
  );
}
