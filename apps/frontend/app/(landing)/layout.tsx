import React from "react";
import { Outfit, Roboto } from "next/font/google";

const outf = Roboto({ subsets: ["latin"], weight: "400" });

const layout = ({ children }: { children: React.ReactNode }) => {
  return <div className={`relative w-full overflow-x-hidden select-none ${outf.className}`}>{children}</div>;
};

export default layout;
