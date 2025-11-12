import React from "react";
import Header from "@/components/ui/home/header";
import Hero from "@/components/ui/home/hero";
import { Footer } from "@/components/ui/home/footer";

const page = () => {
  return (
    <div className="flex flex-col w-full">
      <Header />
      <Hero />
      <Footer />
    </div>
  );
};

export default page;
  