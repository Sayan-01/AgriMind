
import { Poppins } from "next/font/google";
import UploadZone from "./uploadzone";

const poppins = Poppins({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700", "800", "900"],
});  

export default function Hero() {
  return (
    <section className="w-full ">
      <div className="w-full max-w-[1200px] mx-auto mt-7 px-4 pt-7 pb-10">
        <div className="grid md:grid-cols-[1.1fr_0.9fr] gap-10 items-center">
          <div className="py-2">
            <span className="inline-block bg-emerald-500/10 text-emerald-900 px-3 py-1 rounded-full text-[12px] font-bold tracking-wider uppercase mb-3">
              AI-Powered Agriculture
            </span>
            <h1 className={`max-w-3xl text-[58px] ${poppins.className} leading-[1.05] font-extrabold text-emerald-900 mb-4`}>
              Smart Crop Health Analysis at Your Fingertips
            </h1>
            <p className="text-[18px] text-slate-700 mb-6">
              Upload a photo of your crop and instantly get AI-powered analysis of plant health,
              disease detection, and treatment recommendations.
            </p>
            <div className="flex gap-3 flex-wrap items-center">
              <button className="bg-emerald-500 text-white px-5 py-3 rounded-xl font-extrabold shadow-lg shadow-emerald-500/35 transition hover:-translate-y-0.5">
                Analyze Your Crop
              </button>

              <button className="bg-white text-emerald-900 px-4 py-3 rounded-xl font-bold border border-emerald-500/25 shadow transition hover:-translate-y-0.5 hover:shadow-md">
                How It Works
              </button>
            </div>

            <div className="flex items-center gap-4 mt-6 flex-wrap">
              <div className="flex gap-1">
                {[1, 2, 3, 4, 5].map((i) => (
                  <span key={i} className="text-[18px] text-emerald-500">
                    ★
                  </span>
                ))}
              </div>
              <span className="text-slate-600 text-[14px] font-semibold">
                Loved by 4,200+ teams
              </span>
            </div>
          </div>
          <UploadZone/>
        </div>
      </div>
      <div className="w-full max-w-[1200px] mx-auto px-4 pt-2 pb-9">
        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-white rounded-[14px] p-5 shadow-lg shadow-black/5 border border-emerald-500/10">
            <p className="text-[13px] text-slate-500 font-bold uppercase tracking-wide mb-1.5">
              Monthly growth
            </p>
            <h3 className="text-[28px] font-extrabold text-emerald-900">+34%</h3>
          </div>

          <div className="bg-white rounded-[14px] p-5 shadow-lg shadow-black/5 border border-emerald-500/10">
            <p className="text-[13px] text-slate-500 font-bold uppercase tracking-wide mb-1.5">
              Churn rate
            </p>
            <h3 className="text-[28px] font-extrabold text-emerald-900">2.1%</h3>
          </div>

          <div className="bg-white rounded-[14px] p-5 shadow-lg shadow-black/5 border border-emerald-500/10">
            <p className="text-[13px] text-slate-500 font-bold uppercase tracking-wide mb-1.5">
              Active workspaces
            </p>
            <h3 className="text-[28px] font-extrabold text-emerald-900">12,480</h3>
          </div>
        </div>
      </div>{" "}
    </section>
  );
}
